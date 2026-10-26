"""Stage, upload and verify the CASTOR release on Hugging Face.

    uv run --frozen python scripts/release_049.py inventory --out runs/049/inventory.json
    uv run --frozen python scripts/release_049.py labels --set a0 --out-dir DIR [--upload]
    # weights are loaded only inside jobs (login node: no weight loads)
    shell/nchc/submit.sh --dev --name 049-pair -- uv run --frozen python scripts/release_049.py \
        models --ids ID [ID ...] --out-dir DIR --upload [--public] --verify
    uv run --frozen python scripts/release_049.py repo-cards   # rewrite every main README from the Hub
    uv run --frozen python scripts/release_049.py collection   # create it (private), add every repo on the Hub
    uv run --frozen python scripts/release_049.py manifest --out-dir DIR --out runs/049/release_manifest.json
    # the flip: refuses while card.yaml holds a TBD; rewrites every card, then makes the repos public
    uv run --frozen python scripts/release_049.py publish [--dry-run]

`models` skips ids whose manifest record already shows the requested upload and verify
(--force redoes them).
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from castor import provenance, release

STAGE_FILES = ("config.json", "generation_config.json", "special_tokens_map.json",
               "tokenizer_config.json", "tokenizer.json")
STAGE_WEIGHTS = "pytorch_model.bin"
TWINS = release.ROOT / "data/manifests/castor_twins_1B.json"
TOKEN_FILE = Path("/work/harrychang08/hf/castor_release_token")  # write token, org repos only


def _token() -> str:
    # never fall back to the HF_HOME token, which is shared with astra
    token = TOKEN_FILE.read_text().strip() if TOKEN_FILE.is_file() else ""
    if not token:
        sys.exit(f"release token missing or empty: {TOKEN_FILE}")
    return token


def _models(args) -> list[release.Model]:
    return release.inventory(include_035=args.include_035)


def _all_models() -> list[release.Model]:
    """Every model that may hold a revision, whatever --include-035 says."""
    return release.inventory(include_035=True)


def _twins() -> dict:
    return {t["revision"]: t for t in json.loads(TWINS.read_text())["entries"]}


def _prov(m: release.Model) -> tuple[dict, dict]:
    prov = json.loads((Path(m.out) / "provenance.json").read_text())
    sft = json.loads(Path(prov["parent"]["run_provenance"]).read_text())
    return prov, sft


def _shape(m: release.Model) -> dict:
    """Prompt and sample counts for the cards, read straight from the labels (no column check)."""
    with Path(m.labels).open() as fh:
        return release.label_shape(json.loads(line) for line in fh)


def cmd_inventory(args) -> None:
    models = _models(args)
    rows, missing = [], []
    for m in models:
        ck = m.checkpoint.resolve()
        weights = sorted(ck.glob("*.bin")) + sorted(ck.glob("*.safetensors"))
        stage = ck / STAGE_WEIGHTS  # the one file stage_model reads
        lab = Path(m.labels)
        rec = release.inventory_record(m) | {
            "checkpoint_resolved": str(ck),
            "weights": [w.name for w in weights],
            "bytes": stage.stat().st_size if stage.is_file() else 0,
            "labels_bytes": lab.stat().st_size if lab.exists() else None}
        if not stage.is_file() or not lab.exists():
            missing.append(m.model_id)
        rows.append(rec)
    by_prompt = {}
    for r in rows:
        by_prompt.setdefault(r["prompt"], [0, 0])
        by_prompt[r["prompt"]][0] += 1
        by_prompt[r["prompt"]][1] += r["bytes"]
    out = {"n_models": len(rows), "missing": missing,
           "bytes_total": sum(r["bytes"] for r in rows),
           "by_prompt": {p: {"n": n, "bytes": b} for p, (n, b) in sorted(by_prompt.items())},
           "repos": sorted({r["repo"] for r in rows}),
           "models": rows,
           "git_commit": provenance.git_commit(), "command": provenance.command(),
           "script": "scripts/release_049.py"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(rows)} models, {out['bytes_total'] / 1e9:.1f} GB, {len(out['repos'])} repos, "
          f"missing {missing}")


def _api():
    from huggingface_hub import HfApi

    return HfApi(token=_token())


def _record(out_dir: Path, kind: str, name: str, rec: dict) -> None:
    d = out_dir / "manifest" / kind
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(rec, indent=1) + "\n")


def _transient(e: Exception) -> bool:
    """Connection failures and 5xx; auth, 404 and programming errors surface at once."""
    import requests
    from huggingface_hub.utils import HfHubHTTPError

    if isinstance(e, HfHubHTTPError):
        return e.response is not None and e.response.status_code >= 500
    return isinstance(e, (requests.ConnectionError, requests.Timeout))


def _retry(fn, tries: int = 4):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1 or not _transient(e):
                raise
            print(f"retry {i + 1}: {e}", file=sys.stderr)
            time.sleep(30 * (i + 1))


def _put_readme(api, repo: str, readme: str, revision: str | None = None,
                repo_type: str = "model", message: str = "Model card") -> bool:
    """Upload README.md when it differs from the one on the Hub."""
    from huggingface_hub.utils import EntryNotFoundError

    try:
        current = Path(api.hf_hub_download(repo, "README.md", revision=revision,
                                           repo_type=repo_type)).read_text()
    except EntryNotFoundError:
        current = None
    if current == readme:
        return False
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md", repo_id=repo,
                    revision=revision, repo_type=repo_type, commit_message=message)
    return True


def stage_model(m: release.Model, cfg: dict, twins: dict, out_dir: Path) -> tuple[Path, dict]:
    import torch
    from safetensors.torch import save_file

    ck = m.checkpoint.resolve()
    d = out_dir / "stage" / m.model_id
    d.mkdir(parents=True, exist_ok=True)
    state = torch.load(ck / STAGE_WEIGHTS, map_location="cpu", weights_only=True)
    digest = release.tensor_digest(state)
    save_file({k: v.contiguous() for k, v in state.items()}, d / "model.safetensors",
              metadata={"format": "pt"})
    del state
    for f in STAGE_FILES:
        shutil.copy2(ck / f, d / f)
    prov, sft = _prov(m)
    shape = _shape(m)
    facts = release.card_facts(m, prov, sft, shape, twins)
    (d / "README.md").write_text(release.render_card(facts, cfg))
    rec = {"model_id": m.model_id, "repo": m.repo, "revision": m.revision,
           "checkpoint": str(ck), "bin_sha256": release.sha256_file(ck / STAGE_WEIGHTS),
           "tensor_digest": digest,
           "files": {f.name: release.sha256_file(f) for f in sorted(d.iterdir())}}
    return d, rec | {"facts": facts}


def upload_model(api, m: release.Model, d: Path, private: bool) -> str:
    api.create_repo(m.repo, repo_type="model", private=private, exist_ok=True)
    if api.repo_info(m.repo).private != private:  # exist_ok never changes visibility
        print(f"warning: {m.repo} exists with private={not private}, uploading anyway",
              file=sys.stderr)
    api.create_branch(m.repo, branch=m.revision, exist_ok=True)
    info = _retry(lambda: api.upload_folder(repo_id=m.repo, folder_path=str(d),
                                            revision=m.revision,
                                            commit_message=f"Add {m.revision}"))
    return info.oid


def live_revisions(api, repo: str, models: list[release.Model]) -> list[release.Model]:
    """The revisions of `repo` that exist on the Hub with weights."""
    mine = {m.revision: m for m in models if m.repo == repo}
    refs = {b.name for b in api.list_repo_refs(repo).branches}
    return [mine[r] for r in sorted(refs & set(mine))
            if api.file_exists(repo, "model.safetensors", revision=r)]


def update_repo_card(api, repo: str, models: list[release.Model], cfg: dict) -> None:
    """Rewrite main/README.md from the revisions that exist on the Hub (branches with weights).

    Called once per repo at the end of `models`; concurrent jobs can race, so `repo-cards`
    reruns it at the end."""
    live = live_revisions(api, repo, models)
    if not live:
        return
    twins, datasets = _twins(), []
    for m in live:
        for d in release.card_datasets(release.card_facts(m, *_prov(m), {}, twins)):
            datasets += [d] if d not in datasets else []
    shape = _shape(live[0])
    readme = release.render_repo_card(repo, [release.inventory_record(m) for m in live],
                                      datasets, shape, cfg)
    _put_readme(api, repo, readme)


def verify_model(m: release.Model, rec: dict, out_dir: Path) -> dict:
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    # a scratch copy under out_dir, not the shared HF cache: deleted after the digests
    path = out_dir / "verify" / m.model_id
    shutil.rmtree(path, ignore_errors=True)
    try:
        snapshot_download(m.repo, revision=m.revision, token=_token(), local_dir=str(path))
        digest = release.tensor_digest(load_file(path / "model.safetensors"))
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
        loaded = release.tensor_digest(model.state_dict())
        files = {f: release.sha256_file(path / f) for f in rec["files"]}
    finally:
        shutil.rmtree(path, ignore_errors=True)
    out = {"hub_tensor_digest": digest, "transformers_tensor_digest": loaded,
           "digest_match": digest == rec["tensor_digest"],
           "transformers_match": loaded == rec["tensor_digest"],
           "files_match": files == rec["files"]}
    assert all(out[k] for k in ("digest_match", "transformers_match", "files_match")), (
        m.model_id, out)
    return out


def _done(out_dir: Path, mid: str, args) -> bool:
    """A manifest record that already shows every step this run asks for."""
    p = out_dir / "manifest" / "models" / f"{mid}.json"
    if args.force or not p.is_file():
        return False
    rec = json.loads(p.read_text())
    return ((not args.upload or "hub_commit" in rec)
            and (not args.verify or rec.get("verify", {}).get("digest_match") is True))


def cmd_models(args) -> None:
    out_dir = Path(args.out_dir)
    assert out_dir.is_absolute(), "--out-dir must be absolute"
    cfg = release.card_config()
    twins = _twins()
    models = _all_models()
    by_id = {m.model_id: m for m in models}
    api = _api() if args.upload else None
    for mid in args.ids:
        if _done(out_dir, mid, args):
            print(f"{mid}: already recorded, skipped (--force redoes)")
            continue
        m = by_id[mid]
        d, rec = stage_model(m, cfg, twins, out_dir)
        rec.pop("facts")
        if args.upload:
            rec["hub_commit"] = upload_model(api, m, d, private=not args.public)
        if args.verify:
            rec["verify"] = verify_model(m, rec, out_dir)
        rec |= {"git_commit": provenance.git_commit(), "command": provenance.command()}
        _record(out_dir, "models", mid, rec)
        print(json.dumps({k: rec[k] for k in rec if k != "files"}))
        if args.upload and not args.keep_stage:
            (d / "model.safetensors").unlink()
    if args.upload:  # after every record is written, so a card failure loses none
        for repo in sorted({by_id[mid].repo for mid in args.ids}):
            _retry(lambda r=repo: update_repo_card(api, r, models, cfg))


def cmd_labels(args) -> None:
    import pyarrow.parquet as pq

    out_dir = Path(args.out_dir)
    assert out_dir.is_absolute(), "--out-dir must be absolute"
    models = _all_models()
    sel = release.a0(models) if args.set == "a0" else [m for m in models if m.model_id in args.ids]
    assert sel, "no models selected"
    assert not (unknown := set(args.ids) - {m.model_id for m in models}), f"unknown ids {unknown}"
    shapes = []
    cfg = release.card_config()
    d = out_dir / "labels"
    api = _api() if args.upload else None
    if api:
        api.create_repo(release.LABELS_REPO, repo_type="dataset", private=True, exist_ok=True)
    for m in sel:
        rows = list(release.label_rows(m.labels))
        shapes.append(release.label_shape(rows))
        f = d / "data" / m.prompt / f"{m.model_id}.parquet"
        f.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(release.label_table(m, rows), f, compression="zstd")
        meta = json.loads(Path(m.labels).with_name("labels.meta.json").read_text())
        rec = {"model_id": m.model_id, "repo": m.repo, "revision": m.revision,
               "path_in_repo": str(f.relative_to(d)), "parquet_sha256": release.sha256_file(f),
               "labels_jsonl_sha256": release.sha256_file(m.labels),
               "counts": release.label_counts(rows),
               "generations_sha256": {Path(g).name: release.sha256_file(g)
                                      for g in meta["generations"]},
               "judge_version": meta["judge_version"], "labels_commit": meta["git_commit"]}
        _record(out_dir, "labels", m.model_id, rec)
        print(m.model_id, rec["counts"])
    assert all(s == shapes[0] for s in shapes), shapes
    (d / "README.md").write_text(release.labels_readme(cfg, _n_labels(api, d), shapes[0]))
    if api:
        _retry(lambda: api.upload_folder(repo_id=release.LABELS_REPO, repo_type="dataset",
                                         folder_path=str(d),
                                         commit_message=f"Add labels ({args.set or 'ids'})"))
    if args.verify:
        verify_labels(sel, out_dir)


def _n_labels(api, d: Path) -> int:
    """Models the dataset holds after this upload: parquets on the Hub plus the staged ones."""
    staged = {str(f.relative_to(d)) for f in (d / "data").glob("*/*.parquet")}
    hub = set()
    if api and api.repo_exists(release.LABELS_REPO, repo_type="dataset"):
        hub = {f for f in api.list_repo_files(release.LABELS_REPO, repo_type="dataset")
               if f.startswith("data/") and f.endswith(".parquet")}
    return len(staged | hub)


def verify_labels(sel: list[release.Model], out_dir: Path) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    for m in sel:
        rec_path = out_dir / "manifest" / "labels" / f"{m.model_id}.json"
        rec = json.loads(rec_path.read_text())
        f = hf_hub_download(release.LABELS_REPO, rec["path_in_repo"], repo_type="dataset",
                            token=_token())
        hub = pq.read_table(f).to_pylist()
        meta = json.loads(Path(m.labels).with_name("labels.meta.json").read_text())
        check = {"parquet_match": release.sha256_file(f) == rec["parquet_sha256"],
                 "counts_match": release.label_counts(hub) == release.label_counts(
                     release.label_rows(m.labels)),
                 "generations_match": {Path(g).name: release.sha256_file(g)
                                       for g in meta["generations"]} == rec["generations_sha256"]}
        rec["verify"] = check
        rec_path.write_text(json.dumps(rec, indent=1) + "\n")
        assert all(check.values()), (m.model_id, check)
        print(m.model_id, check)


def cmd_repo_cards(args) -> None:
    api, cfg, models = _api(), release.card_config(), _all_models()
    for repo in sorted({m.repo for m in models}):
        if api.repo_exists(repo):
            update_repo_card(api, repo, models, cfg)
            print(repo)


def sync_collection(api, cfg: dict, models: list[release.Model]) -> str:
    """Create the collection (private) if absent and add every repo on the Hub, in order.

    Items already present keep their place, so repos created later are appended."""
    title = cfg["collection_title"]
    c = next((c for c in api.list_collections(owner=release.ORG, limit=100)
              if c.title == title), None)
    if c is None:
        c = api.create_collection(title, namespace=release.ORG,
                                  description=cfg["collection_description"], private=True)
    have = {i.item_id for i in api.get_collection(c.slug).items}
    repos = [m.repo for m in models]
    for item, kind in release.collection_order(sorted(set(repos))):
        if item not in have and api.repo_exists(item, repo_type=kind):
            api.add_collection_item(c.slug, item, kind, exists_ok=True)
            print(f"{c.slug}: added {item}")
    return c.slug


def cmd_collection(args) -> None:
    api, cfg = _api(), release.card_config()
    slug = sync_collection(api, cfg, _all_models())
    c = api.get_collection(slug)
    print(f"https://huggingface.co/collections/{slug}  private={c.private}  items={len(c.items)}")


def cmd_publish(args) -> None:
    """The Nov 20 flip: every card rewritten with the filled paper fields, then public."""
    cfg = release.card_config()
    if (tbd := release.unfilled(cfg)):
        sys.exit(f"not published: {release.CARD_CONFIG} still holds TBD in {tbd}")
    api, models, twins = _api(), _all_models(), _twins()
    repos = [r for r in sorted({m.repo for m in models}) if api.repo_exists(r)]
    for repo in repos:
        live = live_revisions(api, repo, models)
        for m in live:
            prov, sft = _prov(m)
            card = release.render_card(release.card_facts(m, prov, sft, _shape(m), twins), cfg)
            if not args.dry_run:
                _put_readme(api, repo, card, revision=m.revision)
        if not args.dry_run:
            update_repo_card(api, repo, models, cfg)
            api.update_repo_settings(repo, private=False)
        print(f"{repo}: {len(live)} revisions, {'would be' if args.dry_run else 'now'} public")
    labels = release.LABELS_REPO
    if api.repo_exists(labels, repo_type="dataset"):
        files = api.list_repo_files(labels, repo_type="dataset")
        n = sum(f.startswith("data/") and f.endswith(".parquet") for f in files)
        first = next(m for m in models if m.prompt == "008")  # one shape for every model
        readme = release.labels_readme(cfg, n, _shape(first))
        if not args.dry_run:
            _put_readme(api, labels, readme, repo_type="dataset", message="Dataset card")
            api.update_repo_settings(labels, private=False, repo_type="dataset")
        print(f"{labels}: {n} models, {'would be' if args.dry_run else 'now'} public")
    # last, so no item is still private when the collection shows
    slug = sync_collection(api, cfg, models) if not args.dry_run else None
    if slug:
        desc = cfg["collection_description"].strip()
        desc += "" if release.arxiv_pending(cfg) else f" Paper: arXiv:{cfg['arxiv_id']}."
        api.update_collection_metadata(slug, description=desc[:150], private=False)
    print(f"collection: {'would be' if args.dry_run else 'now'} public")


def cmd_manifest(args) -> None:
    out_dir = Path(args.out_dir)
    recs = {kind: [json.loads(p.read_text())
                   for p in sorted((out_dir / "manifest" / kind).glob("*.json"))]
            for kind in ("models", "labels")}
    out = recs | {"org": release.ORG, "labels_repo": release.LABELS_REPO,
                  "git_commit": provenance.git_commit(), "command": provenance.command()}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print({k: len(v) for k, v in recs.items()})


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-035", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inventory")
    p.add_argument("--out", required=True)
    p = sub.add_parser("models")
    p.add_argument("--ids", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--upload", action="store_true")
    p.add_argument("--public", action="store_true", help="create new repos public")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--keep-stage", action="store_true")
    p.add_argument("--force", action="store_true", help="redo ids already recorded")
    p = sub.add_parser("labels")
    p.add_argument("--set", choices=["a0"])
    p.add_argument("--ids", nargs="*", default=[])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--upload", action="store_true")
    p.add_argument("--verify", action="store_true")
    sub.add_parser("repo-cards")
    sub.add_parser("collection")
    p = sub.add_parser("publish")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("manifest")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    {"inventory": cmd_inventory, "models": cmd_models, "labels": cmd_labels,
     "repo-cards": cmd_repo_cards, "collection": cmd_collection, "publish": cmd_publish,
     "manifest": cmd_manifest}[args.cmd](args)


if __name__ == "__main__":
    main()
