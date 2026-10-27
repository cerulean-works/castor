import json

import pytest
import torch

from castor import release


def test_naming():
    assert release.repo_name("dclm-baseline", "1B") == "cerulean-works/castor-dclm-baseline-1B"
    assert release.revision_name("default", 1) == "sdefault-p1"
    assert release.revision_name("large-aux-2", 3) == "slarge-aux-2-p3"
    assert release.revision_name("replicate-i6198-d6198", 2) == "sdefault-r1-p2"
    assert release.revision_name("replicate2-i6198-d6198", 4) == "sdefault-r2-p4"
    assert release.revision_name("init-aux2-data-default", 1) == "sinit-aux2-data-default-p1"


def test_inventory_counts_and_slots():
    models = release.inventory()
    assert len(models) == 156 + 24 + 8 + 4 + 12
    by = {}
    for m in models:
        by[m.prompt] = by.get(m.prompt, 0) + 1
    assert by == {"008": 156, "009": 8, "011": 4, "016": 24, "080": 12}
    swap = [m for m in models if m.prompt == "080"]
    assert {m.revision for m in swap if m.recipe == "dclm-baseline"} == {
        f"s{s}-p{k}-swap080" for s in ("default", "large-aux-2", "large-aux-3") for k in (1, 2)}
    assert all("/swap080/" in m.out and m.model_id.endswith("-swap080") for m in swap)
    assert len({(m.repo, m.revision) for m in models}) == len(models)
    assert len({m.repo for m in models}) == 25 + 2
    a0 = release.a0(models)
    assert {m.revision for m in a0} == {f"s{s}-p{k}" for s in ("default", "large-aux-2",
                                                               "large-aux-3")
                                        for k in (1, 2, 3, 4)}
    with_035 = release.inventory(include_035=True)
    assert {m.revision for m in with_035 if m.prompt == "035"} == {
        f"sdefault-r2-p{k}" for k in (1, 2, 3, 4)}


def test_source_of():
    assert release.source_of("harmbench_humanjb-00012") == "harmbench_humanjb"
    assert release.source_of("xstest-00001") == "xstest"
    with pytest.raises(AssertionError):
        release.source_of("nodigits")


def _row(model, pid, s, v):
    r = dict.fromkeys(release.LABEL_COLUMNS)
    r |= {"model": model, "prompt_id": pid, "sample": s, "coherent": True, "wg_vulnerable": v,
          "wg_harmful": 0, "q3g_vulnerable": v, "q3g_harmful": 0}
    return r


def test_label_table_refuses_text(tmp_path):
    m = release.inventory()[0]
    p = tmp_path / "labels.jsonl"
    rows = [_row(m.model_id, "harmbench-00001", 0, 1), _row(m.model_id, "xstest-00002", -1, 0)]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    t = release.label_table(m, release.label_rows(p))
    assert t.column_names[:4] == ["repo", "revision", "model_id", "source"]
    assert "response" not in t.column_names and "prompt" not in t.column_names
    assert t.column("source").to_pylist() == ["harmbench", "xstest"]
    assert release.label_counts(t.to_pylist()) == release.label_counts(rows)
    assert release.label_counts(rows)["wg_vulnerable"] == 1
    p.write_text(json.dumps(rows[0] | {"response": "text"}) + "\n")
    with pytest.raises(ValueError):
        list(release.label_rows(p))
    p.write_text(json.dumps(rows[0] | {"truncated_to_fit": False}) + "\n")
    assert list(release.label_rows(p)) == [rows[0]]
    p.write_text(json.dumps(rows[0] | {"truncated_to_fit": True}) + "\n")
    with pytest.raises(ValueError):
        list(release.label_rows(p))
    (mid, pid, smp), = release.TRUNCATED
    cut = rows[0] | {"model": mid, "prompt_id": pid, "sample": smp}
    p.write_text(json.dumps(cut | {"truncated_to_fit": True}) + "\n")
    assert list(release.label_rows(p)) == [cut]
    p.write_text(json.dumps({k: v for k, v in rows[0].items() if k != "refusal_agree"}) + "\n")
    with pytest.raises(ValueError):
        list(release.label_rows(p))


def test_tensor_digest_format_invariant(tmp_path):
    from safetensors.torch import load_file, save_file

    g = torch.Generator().manual_seed(0)
    state = {"b": torch.randn(3, 4, generator=g).to(torch.bfloat16),
             "a": torch.randn(5, generator=g), "scale": torch.tensor(0.5, dtype=torch.bfloat16),
             "empty": torch.zeros(0)}
    torch.save(state, tmp_path / "m.bin")
    save_file(state, tmp_path / "m.safetensors")
    d = release.tensor_digest(state)
    assert d == release.tensor_digest(torch.load(tmp_path / "m.bin", weights_only=True))
    assert d == release.tensor_digest(load_file(tmp_path / "m.safetensors"))
    state["a"][0] += 1
    assert d != release.tensor_digest(state)


def _prov():
    return ({"oi_seed": 111, "config": "configs/posttrain/dpo.yaml", "git_commit": "c" * 40,
             "parent": {"oi_seed": 1},
             "dataset": {"repo": "allenai/pref", "revision": "r" * 40},
             "open_instruct": {"commit": "o" * 40}},
            {"config": "configs/posttrain/sft.yaml",
             "dataset": {"repo": "allenai/sft", "revision": "s" * 40}})


SHAPE = {"n_prompts": 6226, "n_samples": 20}


def test_label_shape():
    rows = [{"prompt_id": p, "sample": k} for p in ("a-1", "b-2") for k in (-1, 0, 1)]
    assert release.label_shape(rows) == {"n_prompts": 2, "n_samples": 2}
    with pytest.raises(AssertionError):
        release.label_shape(rows[:-1])
    with pytest.raises(AssertionError):
        release.label_shape([r for r in rows if r != {"prompt_id": "b-2", "sample": -1}])


def test_cards():
    cfg = release.card_config()
    models = release.inventory(include_035=True)
    m = next(x for x in models if x.model_id == "dd-dclm-baseline-1b-default-dpo-seed1")
    f = release.card_facts(m, *_prov(), SHAPE)
    card = release.render_card(f, cfg)
    for s in ("license: apache-2.0", "base_model: allenai/DataDecide-dclm-baseline-1B",
              "step69369-seed-default", "Post-training seed:** 1", "--seed` 1 for SFT, 111",
              "allenai/sft", "allenai/pref", release.CODE_URL, release.LABELS_REPO,
              "answered 6,226 public", "20 sampled", "## Intended use", "## Safety"):
        assert s in card, s
    assert "\u2014" not in card and "{" not in card.split("```bibtex")[0]
    r = next(x for x in models if x.prompt == "011")
    twins = {"replicate-i6198-d6198": {"init_seed": 6198, "data_seed": 6198, "step": 1234,
                                       "sha": "f" * 64}}
    rc = release.render_card(release.card_facts(r, *_prov(), SHAPE, twins), cfg)
    assert "base_model" not in rc and "initialisation seed 6198" in rc and "step 1,234." in rc
    with pytest.raises(AssertionError):
        release.card_facts(r, *_prov(), SHAPE, {})


def test_swap_card():
    cfg = release.card_config()
    m = next(x for x in release.inventory() if x.model_id.endswith("default-dpo-seed1-swap080")
             and x.recipe == "dclm-baseline")
    prov, sft = _prov()
    f = release.card_facts(m, prov, sft | {"dataset": {"repo": "x (080 swap, castor)",
                                                       "revision": None}}, SHAPE)
    card = release.render_card(f, cfg)
    assert "revision `sdefault-p1-swap080`" in card and "(080 swap" not in card
    for s in ("Safety data replaced", "110,800 rows", "nvidia/Nemotron-SFT-Safety-v2",
              "Anthropic/hh-rlhf", "coconot_converted", "runs/080/swap_mix.json",
              "allenai/tulu-3-sft-olmo-2-mixture-0225", "The DPO stage is unchanged"):
        assert s in card, s
    assert sum(f["swap_removed"].values()) == f["swap_rows"] == 110800
    assert "- PKU-Alignment/BeaverTails" in card.split("---")[1]
    sib = [release.inventory_record(x) for x in release.inventory() if x.repo == m.repo]
    rc = release.render_repo_card(m.repo, sib, release.card_datasets(f), SHAPE, cfg)
    assert "Revisions ending in `-swap080`" in rc


def test_released_tables_have_no_free_text():
    from tests.test_export_049 import E

    assert not [c for c in release.LABEL_COLUMNS if E.FREE_TEXT.match(c)]


def test_repo_card_independent_of_uploader():
    cfg = release.card_config()
    models = release.inventory(include_035=True)
    sib = [release.inventory_record(x) for x in models if x.repo == release.a0(models)[0].repo]
    assert {s["prompt"] for s in sib} >= {"008", "011", "035"}
    ds = ["allenai/sft", "allenai/pref"]
    card = release.render_repo_card(sib[0]["repo"], sib, ds, SHAPE, cfg)
    assert card == release.render_repo_card(sib[0]["repo"], sib[::-1], ds, SHAPE, cfg)
    assert all(f"`{x['revision']}`" in card for x in sib)
    assert "base_model: allenai/DataDecide-dclm-baseline-1B" in card
    own = [s for s in sib if not s["base_ref"].startswith("allenai/")]
    assert "base_model" not in release.render_repo_card(sib[0]["repo"], own, ds, SHAPE, cfg)
    with pytest.raises(AssertionError):
        release.render_repo_card("cerulean-works/other", sib, ds, SHAPE, cfg)


def test_labels_readme_counts():
    text = release.labels_readme(release.card_config(), 12, SHAPE)
    assert "12 models." in text and "6,226 public" in text and "(20 samples" in text
    assert "wildjailbreak-00287" in text and "JUDGE-ERROR" in text


def test_paper_and_citation_placeholders():
    live = release.card_config()
    assert release.unfilled(live) == [] and live["arxiv_id"] == "2610.34074"
    cfg = live | {"arxiv_id": "TBD"}
    assert release.arxiv_pending(cfg)
    assert release.unfilled(cfg | {"authors": "TBD"}) == ["authors"]
    cite = "\n".join(release.citation(cfg))
    assert "ARXIV_ID" not in cite and "eprint" not in cite and "archivePrefix" not in cite
    assert "  year          = {2026}\n}" in cite and "primaryClass" not in cite
    assert release.paper_line(cfg).endswith("(arXiv link to follow)")
    filled = cfg | {"arxiv_id": "2611.01234"}
    cite = "\n".join(release.citation(filled))
    assert "eprint        = {2611.01234}" in cite and "ARXIV_ID" not in cite
    assert release.paper_line(filled) == "**Paper:** [arXiv:2611.01234](https://arxiv.org/abs/2611.01234)"


def test_collection_order():
    repos = ["cerulean-works/castor-c4-1B", "cerulean-works/castor-dclm-baseline-300M",
             "cerulean-works/castor-dclm-baseline-1B"]
    assert release.collection_order(repos) == [
        ("cerulean-works/castor-dclm-baseline-1B", "model"),
        ("cerulean-works/castor-dclm-baseline-300M", "model"),
        (release.LABELS_REPO, "dataset"),
        ("cerulean-works/castor-c4-1B", "model")]
    cfg = release.card_config()
    assert len(cfg["collection_title"]) <= 60 and len(cfg["collection_description"]) <= 150
