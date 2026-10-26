"""Convert downloaded DataDecide checkpoints to LlamaForCausalLM (see castor/convert.py).

    uv run --frozen python scripts/convert_datadecide.py \
        --manifest data/manifests/datadecide_1B.json [--only dclm-baseline-1B] [--workers 4]

Output: /work/harrychang08/castor/models/datadecide-llama/<repo name>/<revision>/ with
config.json, model.safetensors (float32, as released), tokenizer files, and provenance.json
(source repo, revision, commit sha, converter hash, git commit, command). An existing output
is skipped only if its provenance records the same source sha and the same converter code;
otherwise it is converted again, so every twin is built by one converter.
"""

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from transformers import GenerationConfig

from castor import convert, provenance


def is_current(dst: Path, sha: str, converter: str) -> bool:
    prov = dst / "provenance.json"
    if not prov.exists():
        return False
    p = json.loads(prov.read_text())
    return p.get("source_sha") == sha and p.get("converter_sha256") == converter


def convert_one(entry: dict, commit: str, command: str) -> str:
    repo, rev, sha = entry["repo"], entry["revision"], entry["sha"]
    dst = convert.converted_dir(repo, rev)
    converter = convert.converter_sha256()
    if is_current(dst, sha, converter):
        return f"skip {repo}@{rev} (current)"
    src = Path(snapshot_download(repo, revision=sha, ignore_patterns=["training/*"]))
    src_cfg = json.loads((src / "config.json").read_text())
    cfg = convert.llama_config(src_cfg)
    state = convert.convert_state(convert.load_olmo_state(src), cfg)
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    cfg.save_pretrained(tmp)
    GenerationConfig(eos_token_id=cfg.eos_token_id, pad_token_id=cfg.pad_token_id).save_pretrained(tmp)
    save_file(state, str(tmp / "model.safetensors"), metadata={"format": "pt"})
    shutil.copy(src / "tokenizer.json", tmp / "tokenizer.json")
    shutil.copy(src / "special_tokens_map.json", tmp / "special_tokens_map.json")
    tok_cfg = dict(convert.TOKENIZER_CONFIG, model_max_length=src_cfg["max_sequence_length"])
    (tmp / "tokenizer_config.json").write_text(json.dumps(tok_cfg, indent=1))
    (tmp / "provenance.json").write_text(json.dumps({
        "source": f"{repo}@{rev}", "source_sha": sha,
        "converter": "castor/convert.py", "converter_sha256": converter,
        "git_commit": commit, "command": command,
    }, indent=1))
    redo = dst.exists()
    if redo:
        shutil.rmtree(dst)
    tmp.rename(dst)
    return f"{'redo' if redo else 'ok'} {repo}@{rev} -> {dst}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, nargs="+")
    ap.add_argument("--only", nargs="*", default=None, help="substrings of repo@revision")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    commit, command = provenance.git_commit(), provenance.command()
    entries = [e for m in args.manifest for e in json.loads(Path(m).read_text())["entries"]]
    if args.only:
        entries = [e for e in entries
                   if any(s in f"{e['repo']}@{e['revision']}" for s in args.only)]
    n = len(entries)
    with ProcessPoolExecutor(args.workers) as ex:
        for msg in ex.map(convert_one, entries, [commit] * n, [command] * n):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
