"""Convert one of our OLMo pretraining checkpoints (via its hf_olmo copy) to LlamaForCausalLM with
the released twins' converter, and register it.

    uv run --frozen python scripts/convert_castor_ckpt.py \
        --ref castor/DataDecide-dclm-baseline-1B@init-default-data-aux2 \
        --hf-olmo /work/harrychang08/castor/runs/003/eval/ours-1B/hf/castor-init-default-data-aux2-step69369 \
        --run castor-init-default-data-aux2 --step 69369 --init-seed 6198 --data-seed 4 \
        --register data/manifests/castor_twins_1B.json

The hf_olmo copy is the one `shell/olmo/eval_ours.sh` / `to_hf.sh` made for OLMES (its
provenance names the unsharded checkpoint and the sha256 of its model.pt; both are checked
here against the run directory). Output: castor.convert.converted_dir(repo, revision), same
files and provenance fields as scripts/convert_datadecide.py (`source`, `source_sha` = sha256
of model.pt, `converter_sha256`), plus the run, step and seeds. Skip-if-done keyed on
provenance (same source sha and converter). `--out-root` redirects the output (smoke tests).
Validate afterwards with scripts/validate_conversion.py --source-dir <hf-olmo>.
"""

import argparse
import json
import shutil
from pathlib import Path

from safetensors.torch import save_file
from transformers import GenerationConfig

from castor import convert, localckpt, provenance

RUNS = Path("/work/harrychang08/castor/runs/003")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="castor/DataDecide-<recipe>-<size>@<cell>")
    ap.add_argument("--hf-olmo", type=Path, required=True)
    ap.add_argument("--run", required=True, help="run name under runs/003/")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--init-seed", type=int, required=True)
    ap.add_argument("--data-seed", type=int, required=True)
    ap.add_argument("--register", type=Path, default=None, help="manifest to add the entry to")
    ap.add_argument("--out-root", type=Path, default=None)
    a = ap.parse_args()
    commit, command = provenance.git_commit(), provenance.command()
    repo, rev = localckpt.parse_ref(a.ref)

    ckpt = RUNS / a.run / "ckpt" / f"step{a.step}-unsharded"
    hprov = json.loads((a.hf_olmo / "provenance.json").read_text())
    assert Path(hprov["source"]).resolve() == ckpt.resolve(), (hprov["source"], ckpt)
    sha = localckpt.sha256_file(ckpt / "model.pt")
    assert sha == hprov["model_pt_sha256"], "hf_olmo copy is not of this model.pt"
    changes = json.loads((RUNS / a.run / "config.yaml.changes.json").read_text())
    size = repo.rsplit("-", 1)[1]
    assert changes["ladder_args"]["model"] == size, (changes["ladder_args"]["model"], size)
    assert changes["ladder_args"]["seed"] == a.init_seed, (changes["ladder_args"]["seed"], a.init_seed)
    assert localckpt.data_seed(changes) == a.data_seed, (localckpt.data_seed(changes), a.data_seed)

    dst = convert.converted_dir(repo, rev)
    if a.out_root:
        dst = a.out_root / dst.relative_to(convert.OUT)
    converter = convert.converter_sha256()
    if (dst / "provenance.json").exists():
        p = json.loads((dst / "provenance.json").read_text())
        if p.get("source_sha") == sha and p.get("converter_sha256") == converter:
            print(f"skip {a.ref} (current)")
            return
    src_cfg, filled = localckpt.load_config(a.hf_olmo)
    cfg = convert.llama_config(src_cfg)
    state = convert.convert_state(localckpt.load_state(a.hf_olmo), cfg)
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    cfg.save_pretrained(tmp)
    GenerationConfig(eos_token_id=cfg.eos_token_id, pad_token_id=cfg.pad_token_id).save_pretrained(tmp)
    save_file(state, str(tmp / "model.safetensors"), metadata={"format": "pt"})
    shutil.copy(a.hf_olmo / "tokenizer.json", tmp / "tokenizer.json")
    shutil.copy(a.hf_olmo / "special_tokens_map.json", tmp / "special_tokens_map.json")
    tok_cfg = dict(convert.TOKENIZER_CONFIG, model_max_length=src_cfg["max_sequence_length"])
    (tmp / "tokenizer_config.json").write_text(json.dumps(tok_cfg, indent=1))
    prov = {
        "source": a.ref, "source_sha": sha, "source_checkpoint": str(ckpt),
        "hf_olmo_copy": str(a.hf_olmo), "hf_olmo_converter": hprov["converter"],
        "config_fields_filled": {k: localckpt.FILL[k] for k in filled},
        "run": a.run, "step": a.step, "init_seed": a.init_seed, "data_seed": a.data_seed,
        "tokenizer_json_sha256": localckpt.sha256_file(a.hf_olmo / "tokenizer.json"),
        "converter": "castor/convert.py", "converter_sha256": converter,
        "git_commit": commit, "command": command,
    }
    (tmp / "provenance.json").write_text(json.dumps(prov, indent=1))
    if dst.exists():
        shutil.rmtree(dst)
    tmp.rename(dst)
    print(f"ok {a.ref} -> {dst} (filled {filled})")

    if a.register:
        m = json.loads(a.register.read_text()) if a.register.exists() else {"entries": []}
        entry = {"repo": repo, "revision": rev, "sha": sha, "run": a.run, "step": a.step,
                 "init_seed": a.init_seed, "data_seed": a.data_seed,
                 "source_checkpoint": str(ckpt), "local_path": str(dst)}
        m["entries"] = [e for e in m["entries"] if (e["repo"], e["revision"]) != (repo, rev)]
        m["entries"].append(entry)
        m["entries"].sort(key=lambda e: (e["repo"], e["revision"]))
        a.register.write_text(json.dumps(m, indent=1) + "\n")
        print(f"registered {a.ref} in {a.register}")


if __name__ == "__main__":
    main()
