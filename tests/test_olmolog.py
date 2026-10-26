from castor.olmolog import in_loop_evals

LOG = """\
2026-09-27 12:49:01.871	host:0	olmo.train:880	INFO	[step=1249/38147]
    train/CrossEntropyLoss=4.110
2026-09-27 12:49:01.871	host:0	olmo.train:880	INFO	[step=1250/38147]
    train/CrossEntropyLoss=4.010
    train/Perplexity=1,224
2026-09-27 12:49:05.000	host:0	olmo.train:880	INFO	all-small-ppl-validation
    eval/c4_en-validation/CrossEntropyLoss=4.091
    eval/c4_en-validation/Perplexity=59.81
    eval/dolma_common-crawl-validation/Perplexity=1,203
    eval/m2d2_s2orc-validation/Perplexity=148.2
2026-09-27 12:49:06.000	host:0	olmo.train:880	INFO	piqa
    eval/downstream/piqa_len_norm=0.5822
2026-09-27 12:49:07.000	host:0	olmo.train:880	INFO	[step=1251/38147]
    train/CrossEntropyLoss=4.000
"""


def test_parses_eval_block_under_its_step(tmp_path):
    p = tmp_path / "a.out"
    p.write_text(LOG)
    ev = in_loop_evals([p])
    assert list(ev) == [1250]
    assert ev[1250]["c4_en"] == {"CrossEntropyLoss": 4.091, "Perplexity": 59.81}
    assert ev[1250]["dolma_common-crawl"]["Perplexity"] == 1203.0
    assert ev[1250]["m2d2_s2orc"]["Perplexity"] == 148.2
    assert "piqa" not in ev[1250]


def test_later_log_overrides_repeated_step(tmp_path):
    a, b = tmp_path / "a.out", tmp_path / "b.out"
    a.write_text(LOG)
    b.write_text(LOG.replace("=59.81", "=59.90"))
    assert in_loop_evals([a, b])[1250]["c4_en"]["Perplexity"] == 59.90
