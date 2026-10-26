import pandas as pd

from castor import provenance

D35 = provenance.load_script("divergence_035")


def _tsv(pairs, steps=(0, 2500)):
    return pd.DataFrame([{"step": s, "pair": p, "rel_l2": v + s / 1e4, "cos": 1 - v}
                         for s in steps for p, v in pairs.items()])


def test_merge_takes_r_from_011_and_the_rest_from_035():
    d35 = _tsv({"ours~r1": 0.1, "ours~default": 0.2, "large-aux-2~default": 0.4,
                "ours~large-aux-2": 0.5})
    d11 = _tsv({"ours~default": 0.3, "large-aux-2~default": 0.9})
    m = D35.merge(d35, d11, "rel_l2")
    assert list(m.columns) == list(D35.LABEL)
    assert m.loc[2500].round(4).to_dict() == {"r2~r1": 0.35, "r2~default": 0.45,
                                             "r1~default": 0.55, "large-aux-2~default": 0.65}
