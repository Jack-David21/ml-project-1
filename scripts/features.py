import pandas as pd


def add_features(df, keep, win=10):
    """Rolling mean and slope of each column in `keep`, computed per engine.

    Needs columns `unit` and `cycle`. Uses only past and current cycles.
    The same function must be used for training and for the API.
    """
    out = []
    for _, g in df.sort_values(["unit", "cycle"]).groupby("unit"):
        g = g.copy()
        x = g["cycle"].astype(float)
        y = g[keep]

        roll = lambda d: d.rolling(win, min_periods=1).sum()
        n = x.rolling(win, min_periods=1).count()
        sx, sxx = roll(x), roll(x * x)
        sy, sxy = roll(y), roll(y.mul(x, axis=0))

        denom = n * sxx - sx**2
        slope = (sxy.mul(n, axis=0) - sy.mul(sx, axis=0)).div(denom.where(denom > 0), axis=0)

        g[[f"{c}_mean" for c in keep]] = sy.div(n, axis=0).to_numpy()
        g[[f"{c}_slope" for c in keep]] = slope.fillna(0.0).to_numpy()
        out.append(g)
    return pd.concat(out, ignore_index=True)