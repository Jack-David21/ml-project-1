import pandas as pd
from scripts.features import add_features

KEEP = ["s2"]


def make_engine():
    return pd.DataFrame({
        "unit": [1] * 12,
        "cycle": list(range(1, 13)),
        "s2": [float(i) for i in range(1, 13)],
    })


def test_first_row_has_no_history():
    out = add_features(make_engine(), KEEP, win=10)
    first = out.iloc[0]
    assert first["s2_mean"] == first["s2"]
    assert first["s2_slope"] == 0


def test_no_future_leakage():
    a = make_engine()
    b = make_engine()
    b.loc[b.index[-1], "s2"] = 999.0  # change only the last reading

    out_a = add_features(a, KEEP, win=10)
    out_b = add_features(b, KEEP, win=10)

    cols = ["s2_mean", "s2_slope"]
    # rows 1 to 11 must not notice that the last reading changed
    pd.testing.assert_frame_equal(out_a[cols].iloc[:-1], out_b[cols].iloc[:-1])
    # the last row must notice, or the test proves nothing
    assert out_a[cols].iloc[-1].tolist() != out_b[cols].iloc[-1].tolist()