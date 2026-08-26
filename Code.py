
import os
import sys
import glob
import argparse
import warnings

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm

import matplotlib
matplotlib.use("Agg")                      
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


SEED = 42                                  


EVENTS = {

    "Introduced S.394":   "2025-02-04",    

    "Cloture FAILS":      "2025-05-08",    
    "Cloture PASSES":     "2025-05-19",    
    "Senate passage":     "2025-06-17",    
    "Signed into law":    "2025-07-18",
}
PLACEBO_DATE = "2025-01-06"                
T0_MAY19   = "2025-05-19 21:45"            
T0_JUN17   = "2025-06-17 21:15"            
T0_JUL18   = "2025-07-18 19:00"            


W_RED, W_RES, W_REG, W_TRA, W_INT, W_GOV = 0.25, 0.25, 0.20, 0.15, 0.075, 0.075


NAVY, RED, GREY, TEAL, PURPLE = "#26436b", "#B1534A", "#8a8f98", "#3d7a80", "#6b5280"
EVENT_C = "#cf2c3a"


SHEETS = {}

def save_sheet(name, df):
    
    SHEETS[name[:31]] = df
    print(f"  [sheet] {name[:31]:35s} rows={len(df)}")


def winsor(s, lo=0.01, hi=0.99):
    
    ql, qh = s.quantile(lo), s.quantile(hi)
    return s.clip(ql, qh)

def twoway_demean(df, cols, fe1, fe2, tol=1e-10, maxiter=1000):
    
    out = df[cols].values.astype(float).copy()
    prev = out.copy()
    for _ in range(maxiter):
        tmp = pd.DataFrame(out, columns=cols)
        for fe in (fe1, fe2):
            out = out - tmp.groupby(df[fe].values).transform("mean").values
            tmp = pd.DataFrame(out, columns=cols)
        if np.abs(out - prev).max() < tol:
            break
        prev = out.copy()
    return pd.DataFrame(out, columns=cols, index=df.index)

def event_did(data, evdate, yvar, treat, win=30, nperm=1000, seed=SEED):
    
    ev = pd.Timestamp(evdate)
    s = data[(data["date"] >= ev - pd.Timedelta(days=win))
             & (data["date"] <= ev + pd.Timedelta(days=win))].dropna(subset=[yvar]).copy()
    s["post"] = (s["date"] >= ev).astype(float)
    s["TxP"] = s[treat] * s["post"]

    
    dm = twoway_demean(s, [yvar, "TxP"], "coin", "date")
    res = sm.OLS(dm[yvar].values, dm[["TxP"]].values).fit(
        cov_type="cluster", cov_kwds={"groups": s["coin"].values})
    beta = res.params[0]

    
    
    rng = np.random.default_rng(seed)
    coins = np.sort(s["coin"].unique())
    tvec = s.groupby("coin")[treat].first().reindex(coins).values  
    postv = s["post"].values
    ucoin, cidx = np.unique(s["coin"].values, return_inverse=True)
    udate, didx = np.unique(s["date"].values, return_inverse=True)
    cnt_coin = np.bincount(cidx).astype(float)
    cnt_date = np.bincount(didx).astype(float)
    ydm = dm[yvar].values
    cnt = 0
    for _ in range(nperm):
        xp = rng.permutation(tvec)[cidx] * postv            
        xp = xp - (np.bincount(cidx, xp) / cnt_coin)[cidx]  
        xp = xp - (np.bincount(didx, xp) / cnt_date)[didx]  
        bp = float(xp @ ydm) / float(xp @ xp)
        cnt += abs(bp) >= abs(beta) - 1e-12

    return {"beta": beta, "se": res.bse[0],
            "p_cluster": res.pvalues[0],
            "p_RI": (cnt + 1) / (nperm + 1),
            "n": len(s), "n_treat": int(s.loc[s[treat] > 0, "coin"].nunique())}

def star(p):
    
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""

def cronbach_alpha(X):
    
    X = np.asarray(X, float)
    k = X.shape[1]
    return k / (k - 1) * (1 - X.var(axis=0, ddof=1).sum() / X.sum(axis=1).var(ddof=1))


DIM_COLS = ["Redemption Rights", "Regulatory Status", "Reserve Quality",
            "Transparency", "International Acceptance", "Governance"]

def build_scores(data_dir):
    
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    path = os.path.join(data_dir, "stablecoin_scores_verified.xlsx")
    sc = pd.read_excel(path, sheet_name="master")
    sc = sc.rename(columns={"Stablecoin": "coin"})
    sc = sc.dropna(subset=["coin"]).copy()

    
    sc["GENIUS_index"] = (W_RED * sc["Redemption Rights"]
                          + W_RES * sc["Reserve Quality"]
                          + W_REG * sc["Regulatory Status"]
                          + W_TRA * sc["Transparency"]
                          + W_INT * sc["International Acceptance"]
                          + W_GOV * sc["Governance"])
    sc["GENIUS_z"] = (sc["GENIUS_index"] - sc["GENIUS_index"].mean()) / sc["GENIUS_index"].std()
    sc = sc.sort_values("GENIUS_index", ascending=False).reset_index(drop=True)
    sc["rank"] = np.arange(1, len(sc) + 1)
    sc["GOOD_top15"] = (sc["rank"] <= 15).astype(float)

    
    Xs = StandardScaler().fit_transform(sc[DIM_COLS].values)
    pca = PCA().fit(Xs)
    km = KMeans(n_clusters=2, n_init=50, random_state=SEED).fit(Xs)
    diag = pd.DataFrame({
        "metric": ["PC1 variance share", "PC2 variance share", "Cronbach alpha",
                   "kmeans silhouette (k=2)",
                   "PC1 loading: Redemption Rights", "PC1 loading: Regulatory Status",
                   "PC1 loading: Reserve Quality", "PC1 loading: Transparency",
                   "PC1 loading: International Acceptance", "PC1 loading: Governance"],
        "value": [pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[1],
                  cronbach_alpha(sc[DIM_COLS].values),
                  silhouette_score(Xs, km.labels_),
                  *pca.components_[0]]})
    save_sheet("Stage1_diagnostics", diag)
    save_sheet("Stage1_spearman_matrix", sc[DIM_COLS].corr(method="spearman"))
    save_sheet("Stage1_legitimacy_scores",
               sc[["rank", "coin", "GENIUS_index", "GENIUS_z", "GOOD_top15"] + DIM_COLS])
    return sc[["coin", "GENIUS_index", "GENIUS_z", "GOOD_top15", "rank"] + DIM_COLS]


def build_daily_panel(data_dir, scores):
    
    frames = []
    files = sorted(glob.glob(os.path.join(data_dir, "* Daily Data.xlsx")))
    print(f"  发现日级文件 {len(files)} 个（应为 30）")
    for f in files:
        coin = os.path.basename(f).replace(" Daily Data.xlsx", "")
        d = pd.read_excel(f, sheet_name=0)
        d = d.rename(columns={"Date": "date", "Price (USD)": "price",
                              "Market Cap (USD)": "mcap", "24h Volume (USD)": "vol24h",
                              "Change %": "chg"})
        d["coin"] = coin
        frames.append(d[["coin", "date", "price", "mcap", "vol24h"]])
    p = pd.concat(frames, ignore_index=True)
    p["date"] = pd.to_datetime(p["date"])
    for c in ["price", "mcap", "vol24h"]:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p = p.sort_values(["coin", "date"]).drop_duplicates(["coin", "date"]).reset_index(drop=True)

    
    p["dev_bps"] = (p["price"] - 1.0) * 1e4            
    p["absdev_bps"] = p["dev_bps"].abs()               
    p["lsupply"] = np.log(p["mcap"].where(p["mcap"] > 0))
    
    
    p["dsupply"] = p.groupby("coin")["lsupply"].diff() * 100
    
    p["dlogprice"] = np.log(p["price"].where(p["price"] > 0))
    p["dlogprice"] = p.groupby("coin")["dlogprice"].diff() * 100
    p["dsupply_pure"] = p["dsupply"] - p["dlogprice"]
    p["turnover"] = p["vol24h"] / p["mcap"]

    
    p["glitch"] = ((p["dsupply"].abs() > 50) | (p["dev_bps"].abs() > 500)
                   | (p["mcap"] <= 0) | p["price"].isna())
    n_gl = int(p["glitch"].sum())
    print(f"  glitch 观测 {n_gl} 条，已置为缺失")
    for c in ["dev_bps", "absdev_bps", "dsupply", "dsupply_pure", "turnover"]:
        p.loc[p["glitch"], c] = np.nan

    
    for c in ["dev_bps", "absdev_bps", "dsupply", "dsupply_pure", "turnover"]:
        p[c + "_w"] = winsor(p[c])

    
    p = p.merge(scores[["coin", "GENIUS_index", "GENIUS_z", "GOOD_top15"]],
                on="coin", how="left")
    missing = sorted(p.loc[p["GENIUS_index"].isna(), "coin"].unique())
    if missing:
        print(f"  [警告] 以下币未匹配到评分，将被排除: {missing}")
    print(f"  日级面板: {p['coin'].nunique()} 币, {len(p)} 行, "
          f"{p['date'].min().date()} ~ {p['date'].max().date()}")
    return p



HOUR_NAME_MAP = {"USDT": "USDT", "USDC": "USDC", "USDS": "USDS", "USD1": "USD1",
                 "USDE": "USDe", "USDG": "USDG", "PYUSD": "PYUSD", "USDD": "USDD",
                 "RLUDD": "RLUSD", "USDF": "Falcon USDf", "USDTB": "USDtb",
                 "GHO": "GHO", "USD0": "USD0", "TUSD": "TUSD", "FDUSD": "FDUSD",
                 "BUSD": "Binance-Peg BUSD", "FRAX": "Legacy FRAX",
                 "CRVUSD": "crvUSD", "AUSD": "AUSD", "SATUSD": "satUSD",
                 "asterUSDF": "Aster USDF", "FRXUSD": "frxUSD",
                 "AVUSD": "Avant avUSD", "USDA": "Avalon USDa", "FXUSD": "fxUSD",
                 "XDAI": "XDAI", "USDX": "USDX", "DOLA": "DOLA", "DAI": "DAI"}

def build_hourly_144h(data_dir):
    
    path = os.path.join(data_dir, "5.19 144h.xlsx")
    frames = []
    for sheet, coin in HOUR_NAME_MAP.items():
        d = pd.read_excel(path, sheet_name=sheet, header=None)
        if d.shape[0] == 0:
            continue
        if d.shape[1] >= 6:
            d = d.iloc[:, :6]
            d.columns = ["ts1", "price", "ts2", "mcap", "ts3", "vol24h"]
        else:                                   
            d = d.iloc[:, :4]
            d.columns = ["ts1", "price", "ts2", "mcap"]
            d["vol24h"] = np.nan
        d["hour"] = pd.to_datetime(d["ts1"], unit="ms", utc=True)
        d["coin"] = coin
        frames.append(d[["coin", "hour", "price", "mcap", "vol24h"]])
    h = pd.concat(frames, ignore_index=True)
    for c in ["price", "mcap", "vol24h"]:
        h[c] = pd.to_numeric(h[c], errors="coerce")
    h = h[(h["hour"] >= "2025-05-16") & (h["hour"] < "2025-05-22")]
    h = h.sort_values(["coin", "hour"]).drop_duplicates(["coin", "hour"]).reset_index(drop=True)
    h["dev_bps"] = (h["price"] - 1.0) * 1e4
    h["absdev_bps"] = h["dev_bps"].abs()
    h["lvol"] = np.log(h["vol24h"].where(h["vol24h"] > 0))
    print(f"  144h 小时面板: {h['coin'].nunique()} 币, {len(h)} 行, "
          f"{h['hour'].min()} ~ {h['hour'].max()}")
    return h

def load_hourly_long(data_dir):
    
    path = os.path.join(data_dir, "hourly_panel_clean.csv")
    if not os.path.exists(path):
        print("  [跳过] 未找到 hourly_panel_clean.csv，7/18 小时级分析跳过")
        return None
    h = pd.read_csv(path)
    h["hour"] = pd.to_datetime(h["hour"], unit="ms", utc=True)
    h = h.rename(columns=str.lower)
    h["dev_bps"] = (h["price"] - 1.0) * 1e4
    h["absdev_bps"] = h["dev_bps"].abs()
    return h

def load_bitstamp(data_dir):
    
    path = os.path.join(data_dir, "Bitstamp_USDCUSD_1h.xlsx")
    b = pd.read_excel(path, header=None)
    b = b.iloc[:, :5]
    b.columns = ["dt", "open", "high", "low", "close"]
    b["dt"] = pd.to_datetime(b["dt"], utc=True)
    for c in ["open", "high", "low", "close"]:
        b[c] = pd.to_numeric(b[c], errors="coerce")
    b = b.dropna(subset=["close"]).sort_values("dt").reset_index(drop=True)
    b["dev_bps"] = (b["close"] - 1.0) * 1e4
    print(f"  Bitstamp USDC: {len(b)} 行, {b['dt'].min()} ~ {b['dt'].max()}")
    return b


def run_headline(p, nperm):
    
    rows = []
    for yvar, ylab in [("dsupply_w", "supply growth (mcap-based)"),
                       ("dsupply_pure_w", "pure supply growth (price-adjusted)")]:
        for treat in ["GOOD_top15", "GENIUS_z"]:
            r = event_did(p, EVENTS["Cloture PASSES"], yvar, treat, win=30, nperm=nperm)
            rows.append({"outcome": ylab, "treat": treat, "window": "±30d",
                         "event": "2025-05-19 cloture passes", **r})
            print(f"    {ylab:38s} {treat:10s} β={r['beta']:+.3f} "
                  f"(p_cl={r['p_cluster']:.4f}, p_RI={r['p_RI']:.4f})")
    save_sheet("PureSupply_headline", pd.DataFrame(rows))
    return rows

def run_windows(p, nperm):
    
    rows = []
    for win in [14, 21, 30, 45]:
        for treat in ["GOOD_top15", "GENIUS_z"]:
            r = event_did(p, EVENTS["Cloture PASSES"], "dsupply_w", treat, win=win, nperm=nperm)
            rows.append({"event": "2025-05-19", "treat": treat, "window": f"±{win}d", **r})
    save_sheet("May19_windows", pd.DataFrame(rows))

def run_event_series(p, nperm):
    
    rows = []
    evs = dict(EVENTS); evs["Placebo Date"] = PLACEBO_DATE
    for name, dt in evs.items():
        r = event_did(p, dt, "dsupply_w", "GOOD_top15", win=30, nperm=nperm)
        rows.append({"event": name, "date": dt, "outcome": "dsupply_w",
                     "treat": "GOOD_top15", "window": "±30d", **r})
        print(f"    {name:22s} β={r['beta']:+.3f} (p_RI={r['p_RI']:.3f})")
    save_sheet("EventSeries_all", pd.DataFrame(rows))
    return pd.DataFrame(rows)

def run_loo(p, nperm):
    
    base_coins = p.loc[p["GOOD_top15"] == 1, "coin"].unique()
    rows = []
    for c in base_coins:
        r = event_did(p[p["coin"] != c], EVENTS["Cloture PASSES"],
                      "dsupply_w", "GOOD_top15", win=30, nperm=nperm)
        rows.append({"dropped": c, **r})
    df = pd.DataFrame(rows)
    save_sheet("May19_LOO", df)
    print(f"    LOO: β∈[{df['beta'].min():.2f},{df['beta'].max():.2f}], "
          f"max p_RI={df['p_RI'].max():.4f}")

def run_cross_section(p, nperm, seed=SEED):
    
    ev = pd.Timestamp(EVENTS["Cloture PASSES"]); win = 30
    s = p[(p["date"] >= ev - pd.Timedelta(days=win))
          & (p["date"] <= ev + pd.Timedelta(days=win))].dropna(subset=["dsupply_w"])
    s = s.assign(post=(s["date"] >= ev).astype(int))
    cnt = s.groupby(["coin", "post"])["dsupply_w"].agg(["mean", "count"]).reset_index()
    wide = cnt.pivot(index="coin", columns="post", values="mean")
    wide.columns = ["pre", "post"]
    wide["delta"] = wide["post"] - wide["pre"]
    wide = wide.dropna().reset_index()
    wide = wide.merge(p.groupby("coin")[["GOOD_top15", "GENIUS_z"]].first(), on="coin")
    rows = []
    rng = np.random.default_rng(seed)
    for treat in ["GOOD_top15", "GENIUS_z"]:
        x = sm.add_constant(wide[treat].values)
        res = sm.OLS(wide["delta"].values, x).fit(cov_type="HC1")
        beta = res.params[1]
        cntp = 0
        tv = wide[treat].values
        for _ in range(nperm):
            xp = sm.add_constant(rng.permutation(tv))
            bp, *_ = np.linalg.lstsq(xp, wide["delta"].values, rcond=None)
            cntp += abs(bp[1]) >= abs(beta) - 1e-12
        rows.append({"treat": treat, "beta": beta, "se": res.bse[1],
                     "p_HC1": res.pvalues[1], "p_perm": (cntp + 1) / (nperm + 1),
                     "n": len(wide)})
        print(f"    截面 {treat:10s} β={beta:+.3f} (p_perm={rows[-1]['p_perm']:.3f}), n={len(wide)}")
    save_sheet("CrossSection_May19", pd.DataFrame(rows))

def run_verification_grid(p, nperm):
    
    rows = []
    for yvar in ["dev_bps_w", "absdev_bps_w", "turnover_w", "dsupply_w"]:
        for treat in ["GOOD_top15", "GENIUS_z"]:
            for win in [7, 14, 21, 30, 45, 60]:
                r = event_did(p, EVENTS["Cloture PASSES"], yvar, treat,
                              win=win, nperm=nperm)
                rows.append({"outcome": yvar, "treat": treat, "window": f"±{win}d", **r})
    df = pd.DataFrame(rows)
    save_sheet("Verification_grid", df)
    sig = df.assign(sig=lambda d: (d["p_cluster"] < 0.05) | (d["p_RI"] < 0.05))
    print("    验证网格显著比例：")
    print(sig.groupby("outcome")["sig"].agg(["sum", "count"]).to_string())
    return df


def _binned_es(p, evdate, yvar, score, bin_days, leads, lags):
    
    ev = pd.Timestamp(evdate)
    s = p.dropna(subset=[yvar]).copy()
    s["k"] = ((s["date"] - ev).dt.days // bin_days)
    s = s[(s["k"] >= -leads) & (s["k"] <= lags)]
    g = s.groupby(["coin", "k"]).agg(y=(yvar, "mean"), sc=(score, "first")).reset_index()
    ks = [k for k in range(-leads, lags + 1) if k != -1]
    for k in ks:
        g[f"D{k}"] = (g["k"] == k).astype(float) * g["sc"]
    cols = [f"D{k}" for k in ks]
    dm = twoway_demean(g, ["y"] + cols, "coin", "k")
    res = sm.OLS(dm["y"].values, dm[cols].values).fit(
        cov_type="cluster", cov_kwds={"groups": g["coin"].values})
    out = pd.DataFrame({"k": ks, "beta": res.params, "se": res.bse, "p": res.pvalues})
    out = pd.concat([out, pd.DataFrame({"k": [-1], "beta": [0.0], "se": [np.nan], "p": [np.nan]})]
                    ).sort_values("k").reset_index(drop=True)
    
    lead_idx = [i for i, k in enumerate(ks) if k <= -2]
    if lead_idx:
        R = np.zeros((len(lead_idx), len(ks)))
        for r, i in enumerate(lead_idx):
            R[r, i] = 1.0
        f = res.f_test(R)
        joint_p = float(f.pvalue)
    else:
        joint_p = np.nan
    return out, joint_p

def run_dynamics(p):
    
    wk, jp_wk = _binned_es(p, EVENTS["Cloture PASSES"], "dsupply_w", "GENIUS_z",
                           bin_days=7, leads=8, lags=12)
    save_sheet("Weekly_ES_May19", wk)
    bi = _binned_es(p, EVENTS["Cloture PASSES"], "dsupply_w", "GENIUS_z",
                    bin_days=14, leads=6, lags=6)[0]
    save_sheet("Biweekly_dynamics", bi)
    print(f"    周频领先项联合检验 p={jp_wk:.4f}（不显著 ⇒ 平行趋势成立）")
    
    
    pre = p[p["date"] < "2025-01-01"]
    _, jp_pre = _binned_es(pre, "2024-11-20", "dsupply_w", "GENIUS_z",
                           bin_days=7, leads=4, lags=3)
    print(f"    纯2024安慰剂（伪事件 2024-11-20）领先联合 p={jp_pre:.4f}")
    return wk, bi

def run_unified_3regime(p):
    
    s = p[(p["date"] >= "2024-10-01") & (p["date"] <= "2025-06-16")
          ].dropna(subset=["dsupply_w"]).copy()
    s["X_ANT"] = ((s["date"] >= "2025-01-01") & (s["date"] <= "2025-05-18")).astype(float) * s["GOOD_top15"]
    s["X_POST"] = (s["date"] >= "2025-05-19").astype(float) * s["GOOD_top15"]
    dm = twoway_demean(s, ["dsupply_w", "X_ANT", "X_POST"], "coin", "date")
    res = sm.OLS(dm["dsupply_w"].values, dm[["X_ANT", "X_POST"]].values).fit(
        cov_type="cluster", cov_kwds={"groups": s["coin"].values})
    wald_p = float(res.f_test(np.array([[1.0, -1.0]])).pvalue)   
    df = pd.DataFrame({"term": ["Score×ANT(预期期)", "Score×POST(事件后)", "POST−ANT"],
                       "beta": [res.params[0], res.params[1], res.params[1] - res.params[0]],
                       "p": [res.pvalues[0], res.pvalues[1], wald_p]})
    save_sheet("DiD_unified_3regime", df)
    print(df.to_string(index=False))

def run_milestone_gradient(p):
    
    bins = {"R1_election_to_intro":   ("2024-11-05", "2025-02-03"),
            "R2_intro_to_committee":  ("2025-02-04", "2025-03-12"),
            "R3_committee_to_fail":   ("2025-03-13", "2025-05-07"),
            "R4_fail_to_pass":        ("2025-05-08", "2025-05-18"),
            "R5_pass_to_senate":      ("2025-05-19", "2025-06-16"),
            "R6_after_senate":        ("2025-06-17", "2025-08-31")}
    s = p[(p["date"] >= "2024-10-01") & (p["date"] <= "2025-08-31")
          ].dropna(subset=["dsupply_w"]).copy()
    cols = []
    for name, (a, b) in bins.items():
        s[f"X_{name}"] = ((s["date"] >= a) & (s["date"] <= b)).astype(float) * s["GOOD_top15"]
        cols.append(f"X_{name}")
    dm = twoway_demean(s, ["dsupply_w"] + cols, "coin", "date")
    res = sm.OLS(dm["dsupply_w"].values, dm[cols].values).fit(
        cov_type="cluster", cov_kwds={"groups": s["coin"].values})
    df = pd.DataFrame({"phase": list(bins.keys()), "beta": res.params,
                       "se": res.bse, "p": res.pvalues})
    save_sheet("Milestone_gradient", df)
    print(df.to_string(index=False))

def run_absdev_sensitivity(p, nperm):
    
    rows = []
    for win in [7, 14, 21, 30]:
        r = event_did(p, EVENTS["Cloture PASSES"], "absdev_bps_w", "GOOD_top15",
                      win=win, nperm=nperm)
        rows.append({"window": f"±{win}d", **r})
    
    ev = pd.Timestamp(EVENTS["Cloture PASSES"])
    s = p[(p["date"] >= ev - pd.Timedelta(days=30)) & (p["date"] <= ev + pd.Timedelta(days=30))
          ].dropna(subset=["absdev_bps"])
    med = s.assign(post=(s["date"] >= ev).astype(int)).groupby(["coin", "post"])["absdev_bps"].median().unstack()
    med["delta"] = med[1] - med[0]
    med = med.dropna().merge(p.groupby("coin")["GOOD_top15"].first(), on="coin")
    g1 = med.loc[med["GOOD_top15"] == 1, "delta"]; g0 = med.loc[med["GOOD_top15"] == 0, "delta"]
    tt = st.ttest_ind(g1, g0, equal_var=False)
    rows.append({"window": "median DiD ±30d", "beta": g1.mean() - g0.mean(),
                 "se": np.nan, "p_cluster": tt.pvalue, "p_RI": np.nan,
                 "n": len(med), "n_treat": int(med["GOOD_top15"].sum())})
    save_sheet("absdev_sensitivity", pd.DataFrame(rows))

def run_mechanism(p):
    
    ev = pd.Timestamp(EVENTS["Cloture PASSES"]); win = 30
    s = p[(p["date"] >= ev - pd.Timedelta(days=win))
          & (p["date"] <= ev + pd.Timedelta(days=win))].copy()
    s = s.sort_values(["coin", "date"])
    rows = []
    for h in range(1, 7):
        s[f"devL{h}"] = s.groupby("coin")["dev_bps_w"].shift(h)
        for yvar in ["dsupply_w", "dsupply_pure_w"]:
            d = s.dropna(subset=[yvar, f"devL{h}"])
            dm = twoway_demean(d, [yvar, f"devL{h}"], "coin", "date")
            res = sm.OLS(dm[yvar].values, dm[[f"devL{h}"]].values).fit(
                cov_type="cluster", cov_kwds={"groups": d["coin"].values})
            rows.append({"lag_days": h, "outcome": yvar,
                         "beta": res.params[0], "se": res.bse[0], "p": res.pvalues[0]})
    df = pd.DataFrame(rows)
    save_sheet("Mechanism_pure_supply", df)
    print(df.pivot(index="lag_days", columns="outcome", values="beta").to_string())


def h_es(h, t0str, yvar="absdev_bps", score_df=None, bin_h=4, nbins=12):
    
    t0 = pd.Timestamp(t0str, tz="UTC")
    s = h.dropna(subset=[yvar]).copy()
    s["k"] = ((s["hour"] - t0).dt.total_seconds() // (bin_h * 3600)).astype(int)
    s = s[(s["k"] >= -nbins) & (s["k"] < nbins)]
    if score_df is not None:
        s = s.merge(score_df[["coin", "GENIUS_z"]], on="coin", how="left")
    ks = [k for k in range(-nbins, nbins) if k != -1]
    for k in ks:
        s[f"D{k}"] = (s["k"] == k).astype(float) * s["GENIUS_z"]
    cols = [f"D{k}" for k in ks]
    d = s.dropna(subset=["GENIUS_z"])
    dm = twoway_demean(d, [yvar] + cols, "coin", "hour")
    res = sm.OLS(dm[yvar].values, dm[cols].values).fit(
        cov_type="cluster", cov_kwds={"groups": d["coin"].values})
    out = pd.DataFrame({"k": ks, "beta": res.params, "se": res.bse, "p": res.pvalues})
    out = pd.concat([out, pd.DataFrame({"k": [-1], "beta": [0.0], "se": [np.nan], "p": [np.nan]})]
                    ).sort_values("k").reset_index(drop=True)
    n_sig = int((out["p"] < 0.05).sum())
    print(f"    4h 事件研究（{t0str}）: 显著箱数 {n_sig}/{len(ks)}")
    return out


def run_dispersion(h, scores, t0str=T0_MAY19, nperm=1000, seed=SEED):
    
    t0 = pd.Timestamp(t0str, tz="UTC")
    s = h.merge(scores[["coin", "GOOD_top15", "GENIUS_z"]], on="coin", how="left").dropna(subset=["GOOD_top15"])
    s["grp"] = np.where(s["GOOD_top15"] == 1, "LEGIT", "CTRL")
    rows, coin_rows = [], []
    for hrs in [48, 72]:
        pre = s[(s["hour"] >= t0 - pd.Timedelta(hours=hrs)) & (s["hour"] < t0)]
        post = s[(s["hour"] >= t0) & (s["hour"] < t0 + pd.Timedelta(hours=hrs))]
        for gname, gpre, gpost in [("LEGIT", pre[pre["grp"] == "LEGIT"], post[post["grp"] == "LEGIT"]),
                                   ("CTRL", pre[pre["grp"] == "CTRL"], post[post["grp"] == "CTRL"])]:
            sd_pre = gpre.groupby("hour")["dev_bps"].std().mean()
            sd_post = gpost.groupby("hour")["dev_bps"].std().mean()
            rows.append({"window_h": hrs, "group": gname,
                         "dispersion_pre_bps": sd_pre, "dispersion_post_bps": sd_post,
                         "change": sd_post - sd_pre})
        
        for gname in ["LEGIT", "CTRL"]:
            sub_pre, sub_post = (pre[pre["grp"] == gname], post[post["grp"] == gname])
            a = sub_pre.groupby("coin")["absdev_bps"].mean()
            b = sub_post.groupby("coin")["absdev_bps"].mean()
            delta = (b - a).dropna()
            tt = st.ttest_1samp(delta, 0) if len(delta) > 2 else None
            coin_rows.append({"window_h": hrs, "group": gname,
                              "mean_change_bps": delta.mean(),
                              "p_1samp": tt.pvalue if tt else np.nan, "n_coins": len(delta)})
    df1 = pd.DataFrame(rows); df2 = pd.DataFrame(coin_rows)
    save_sheet("B1_dispersion_periods", df1)
    save_sheet("B2_coinlevel_absdev_change", df2)
    print(df1.to_string(index=False)); print(df2.to_string(index=False))
    return df1

def run_volume_tests(h, scores, t0str=T0_MAY19, seed=SEED):
    
    t0 = pd.Timestamp(t0str, tz="UTC")
    s = h.merge(scores[["coin", "GOOD_top15"]], on="coin", how="left").dropna(
        subset=["GOOD_top15", "lvol"])
    s["grp"] = np.where(s["GOOD_top15"] == 1, "LEGIT", "CTRL")
    rng = np.random.default_rng(seed)

    def perm_between(dv, gv, nperm=2000):
        
        g1, g0 = dv[gv == "LEGIT"], dv[gv == "CTRL"]
        obs = g1.mean() - g0.mean()
        cnt = 0
        pool = dv.values.copy()
        for _ in range(nperm):
            rp = rng.permutation(pool)
            cnt += abs(rp[:len(g1)].mean() - rp[len(g1):].mean()) >= abs(obs) - 1e-12
        return obs, (cnt + 1) / (nperm + 1)

    
    pre = s[(s["hour"] >= t0 - pd.Timedelta(hours=48)) & (s["hour"] < t0)]
    post = s[(s["hour"] >= t0) & (s["hour"] < t0 + pd.Timedelta(hours=48))]
    a, b = pre.groupby("coin")["lvol"].mean(), post.groupby("coin")["lvol"].mean()
    naive = (b - a).dropna().rename("d_lvol").reset_index()
    naive["grp"] = naive["coin"].map(s.groupby("coin")["grp"].first())

    
    d19 = s[(s["hour"] >= "2025-05-19") & (s["hour"] < "2025-05-19 21:00")]
    d20 = s[(s["hour"] >= "2025-05-20") & (s["hour"] < "2025-05-20 21:00")]
    a2, b2 = d19.groupby("coin")["lvol"].mean(), d20.groupby("coin")["lvol"].mean()
    adj = (b2 - a2).dropna().rename("d_lvol").reset_index()
    adj["grp"] = adj["coin"].map(s.groupby("coin")["grp"].first())

    rows = []
    for label, df in [("naive_±48h(前窗含周末)", naive),
                      ("weekday_adjusted(5/20 vs 5/19同时段)", adj)]:
        for gname in ["LEGIT", "CTRL"]:
            dv = df.loc[df["grp"] == gname, "d_lvol"]
            tt = st.ttest_1samp(dv, 0)
            rows.append({"comparison": label, "group": gname,
                         "mean_dlog_vol": dv.mean(), "p_1samp": tt.pvalue,
                         "n_coins": len(dv)})
        obs, pp = perm_between(df["d_lvol"], df["grp"])
        rows.append({"comparison": label, "group": "LEGIT−CTRL",
                     "mean_dlog_vol": obs, "p_1samp": pp,
                     "n_coins": len(df)})
    df = pd.DataFrame(rows)
    save_sheet("B3_volume_tests", df)
    print(df.to_string(index=False))

def run_bitstamp_events(b):
    
    rows = []
    for name, t0s in [("Cloture passes 5/19", T0_MAY19),
                      ("Senate passage 6/17", T0_JUN17),
                      ("Signed into law 7/18", T0_JUL18)]:
        t0 = pd.Timestamp(t0s, tz="UTC")
        w = b[(b["dt"] >= t0 - pd.Timedelta(hours=12)) & (b["dt"] <= t0 + pd.Timedelta(hours=12))]
        rows.append({"event": name, "t0_utc": t0s,
                     "max_absdev_bps_±12h": w["dev_bps"].abs().max(),
                     "n_hours": len(w)})
    df = pd.DataFrame(rows)
    save_sheet("Bitstamp_USDC_events", df)
    print(df.to_string(index=False))
    return df

def run_jul18_hourly(hlong, scores):
    
    if hlong is None:
        return None
    h = hlong.merge(scores[["coin", "GENIUS_z"]], on="coin", how="left")
    out = h_es(h, T0_JUL18, yvar="absdev_bps", score_df=None)
    save_sheet("Hourly_Jul18_joint", out)
    return out


def _new_ax(figsize=(9, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax

def _es_plot(df, title, xlab, path, color=NAVY):
    
    fig, ax = _new_ax()
    ax.axhline(0, color=GREY, lw=0.8)
    ax.axvline(-0.5, color=EVENT_C, lw=1.2, ls="--")
    d = df.sort_values("k")
    ax.plot(d["k"], d["beta"], "-o", color=color, ms=4, lw=1.4)
    lo, hi = d["beta"] - 1.96 * d["se"], d["beta"] + 1.96 * d["se"]
    ax.fill_between(d["k"].values, lo.values.astype(float), hi.values.astype(float),
                    color=color, alpha=0.18)
    ax.set_xlabel(xlab); ax.set_ylabel("Coefficient on GENIUS score (supply growth, %)")
    ax.set_title(title)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  [fig] {os.path.basename(path)}")

def make_figures(p, h144, b, scores, out_dir,
                 weekly_es, biweekly_es, event_series):

    ev = pd.Timestamp(EVENTS["Cloture PASSES"])
    plt.rcParams.update({'font.size': 13, 'axes.edgecolor': '#cccccc', 'axes.linewidth': 0.8})
    # fig1 长窗口双周事件研究 ------------------------------------------------
    _es_plot(biweekly_es,
             "Supply-growth response around the GENIUS cloture vote (biweekly bins)",
             "Fortnights relative to 2025-05-19 (bin -1 = reference)",
             os.path.join(out_dir, "fig1_eventstudy_long_EN.png"))

    
    fig, ax = _new_ax(figsize=(10.5,7.2))

    
    plot_df = weekly_es[(weekly_es['k'] >= -4) & (weekly_es['k'] <= 4)].copy()

    
    xw = plot_df['k'] * 7 + 3.5

    ax.axhline(0, color=GREY, lw=0.8)
    ax.axvline(0, color=EVENT_C, ls='--', lw=1.2)

    
    off = (pd.Timestamp('2025-06-17') - ev).days
    ax.axvline(off, color=PURPLE, ls='--', lw=1, alpha=0.5)
    
    ax.text(off - 1, 1.0, 'Senate passes', color=PURPLE, fontsize=14, va='top', ha='right')

    ax.text(1.5, 1.0, 'Cloture passes\n(May 19, 22:00 UTC)', color=EVENT_C, fontsize=14, va='top')

    
    ax.errorbar(xw, plot_df['beta'], yerr=1.96 * plot_df['se'].fillna(0),
                fmt='o-', capsize=3, color=NAVY, ms=4.5, lw=1.4,
                label=r'$\beta_k$: Legitimacy score $\times$ week dummies')

    ax.set_xlabel('Days relative to May 19, 2025   |   Base = week −1', fontsize=17)
    ax.set_ylabel('Differential effect on daily\nsupply growth (pp/day)', fontsize=17)

    
    ax.set_title('Weekly Dynamics Around the May 19 Cloture Vote (±1 Month)\nFlat pre-trends; effect starts in week +1',
                 fontsize=19)

    ax.legend(loc='upper left', fontsize=13, frameon=False)
    ax.grid(axis='y', alpha=0.25)
    ax.set_ylim(-0.75, 1.15)

    
    ax.set_xlim(-32, 35)

    plt.tight_layout()
    
    fig2_path = os.path.join(out_dir, "fig2_eventstudy_weekly_May19_EN.png")
    fig.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [fig] fig2_eventstudy_weekly_May19_EN.png")

    
    es = event_series.copy()
    order = ["Election (placebo)", "Introduced S.394", "Cloture FAILS",
             "Cloture PASSES", "Senate passage", "Placebo Date", "Signed into law"]
    es["event"] = pd.Categorical(es["event"], categories=order, ordered=True)
    es = es.sort_values("event").reset_index(drop=True)
    es['x'] = range(len(es))
    fig, ax = _new_ax(figsize=(10.5, 7.2))
    
    colors = [EVENT_C if e == "Cloture PASSES" else GREY for e in es["event"]]

    ax.axhline(0, color=GREY, lw=0.8)

    ax.errorbar(es['x'], es['beta'], yerr=1.96 * es['se'], fmt='none',
                ecolor=NAVY, capsize=4, lw=1.4)

    ax.scatter(es['x'], es['beta'], c=colors, s=70, zorder=3)
    ax.annotate('β = +0.68\nRI p = 0.003', xy=(2, 0.68), xytext=(1, 1.15), fontsize=14, color= EVENT_C,
                arrowprops=dict(arrowstyle='->', color= EVENT_C))
    ax.grid(axis='y', alpha=0.25)

    ax.set_xticks(es['x'])

    ax.set_xticklabels([f"{e}\n{d}" for e, d in zip(es["event"], es["date"])],
                       fontsize=13, rotation=0)
    ax.set_ylabel("DiD coefficient (supply growth, %)", fontsize=17)
    ax.set_title("Falsification across candidate event dates", fontsize=19)

    for i, (_, r) in enumerate(es.iterrows()):
        ax.text(i, r["beta"] + 0.06 * np.sign(r["beta"]), star(r["p_RI"]),
                ha="center", fontsize=14)
    ax.legend(loc='upper left', fontsize=13, frameon=False)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_eventdates_falsification_EN.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [fig] fig3_eventdates_falsification_EN.png")

    
    sc = scores.sort_values("GENIUS_index", ascending=True)

    fig, ax = _new_ax(figsize=(12.5, 9))

    ax.barh(
        sc["coin"],
        sc["GENIUS_index"],
        color=[
            NAVY if g == 1 else GREY
            for g in sc["GOOD_top15"]
        ],
        alpha=0.9,
        height=0.7
    )


    median_value = sc["GENIUS_index"].median()

    ax.axvline(
        median_value,
        color=EVENT_C,
        linestyle=":",
        linewidth=1.2
    )

    ax.text(
        median_value + 0.03,
        0.3,
        "median",
        color=EVENT_C,
        fontsize=14
    )


    ax.set_title("")
    ax.set_xlabel("")

    weights_text = (
        "Weights taken from the Act:\n"
        "• redemption 0.25\n"
        "• reserve quality 0.25\n"
        "• regulatory status 0.20\n"
        "• transparency 0.15\n"
        "• acceptance 0.075\n"
        "• governance 0.075"
    )

    
    
    box_style = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9)
    ax.text(0.998, 0.12, weights_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right',
            multialignment='left', bbox=box_style)
    ax.grid(
        axis="x",
        alpha=0.25,
        color="#ffffff"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
        color="#ffffff"
    )

    ax.set_axisbelow(True)

    fig.tight_layout(
        rect=[0.02, 0.06, 0.98, 0.95],
        pad=0.5
    )


    fig.canvas.draw()

    axes_position = ax.get_position()

    fig.text(
        0.5,
        axes_position.y1 + 0.012,
        "Legitimacy Ranking of 30 USD Stablecoins",
        ha="center",
        va="bottom",
        fontsize=19
    )

    fig.text(
        0.5,
        axes_position.y0 - 0.045,
        "GENIUS statutory-weighted legitimacy index (1–5)",
        ha="center",
        va="top",
        fontsize=17
    )

    fig.savefig(
        os.path.join(
            out_dir,
            "fig4_legitimacy_ranking_EN.png"
        ),
        dpi=300,
        facecolor=fig.get_facecolor()
    )

    plt.close(fig)

    print("  [fig] fig4_legitimacy_ranking_EN.png")

    
    d = p[(p["date"] >= "2025-04-29") & (p["date"] <= "2025-06-9")].copy()
    d["grp"] = np.where(d["GOOD_top15"] == 1, "Legitimate (top 15)", "Control")
    g = d.groupby(["grp", "date"])["dsupply_w"].mean().reset_index()
    g["roll"] = g.groupby("grp")["dsupply_w"].transform(lambda s: s.rolling(7, 1).mean())

    fig, ax = _new_ax(figsize=(12, 7.2))
    for gname, c in [("Legitimate (top 15)", NAVY), ("Control", GREY)]:
        sub = g[g["grp"] == gname]
        ax.plot(sub["date"], sub["roll"], color=c, lw=1.6, label=gname)

    ax.axvline(ev, color=EVENT_C, lw=1.2, ls="--")
    ax.text(ev, ax.get_ylim()[1], "  Cloture passes\n  (May 19, 22:00 UTC)", color=EVENT_C,
            fontsize=14, va="top")
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_ylabel("Supply growth (%, 7-day moving avg)", fontsize=17)
    ax.set_title("Daily Supply Growth by Legitimacy Group", fontsize=19)
    ax.legend(fontsize=13, frameon=False)
    fig.savefig(os.path.join(out_dir, "fig5_group_supply_growth_EN.png"),
                dpi=300, bbox_inches="tight"); plt.close(fig)
    print("  [fig] fig5_group_supply_growth_EN.png")

    
    fig, axes = plt.subplots(1, 3, figsize=(10.5,7.2), sharey=True)

    for ax, (name, t0s) in zip(axes, [("Cloture passes\n2025-05-19 21:45 UTC", T0_MAY19),
                                      ("Senate passage\n2025-06-17 21:15 UTC", T0_JUN17),
                                      ("Signed into law\n2025-07-18 19:00 UTC", T0_JUL18)]):
        t0 = pd.Timestamp(t0s, tz="UTC")
        w = b[(b["dt"] >= t0 - pd.Timedelta(hours=72)) & (b["dt"] <= t0 + pd.Timedelta(hours=72))]
        ax.plot(w["dt"], w["dev_bps"], color=NAVY, lw=1)
        ax.axhline(0, color=GREY, lw=0.8); ax.axvline(t0, color=EVENT_C, lw=1.2, ls="--")
        ax.set_title(name, fontsize=9); ax.tick_params(axis="x", labelsize=7, rotation=30)
        ax.grid(alpha=0.25);
        ax.set_ylim(-4, 4)
    axes[0].set_ylabel("USDC deviation from $1 (bps)")

    fig.suptitle("Deepest USDC Market (Bitstamp): no reaction at any milestone", y=1.02, fontsize=12)
    fig.savefig(os.path.join(out_dir, "fig6_USDC_peg_events_EN.png"),
                dpi=300, bbox_inches="tight"); plt.close(fig)
    print("  [fig] fig6_USDC_peg_events_EN.png")

    
    t0 = pd.Timestamp(T0_MAY19, tz="UTC")
    hh = h144.merge(scores[["coin", "GOOD_top15"]], on="coin", how="left").dropna(subset=["GOOD_top15"])
    hh["grp"] = np.where(hh["GOOD_top15"] == 1, "Legitimate (top 15)", "Control")
    disp = hh.groupby(["grp", "hour"])["dev_bps"].std().reset_index()
    fig, ax = _new_ax(figsize=(11.5,7.2))
    for gname, c in [("Legitimate (top 15)", NAVY), ("Control", GREY)]:
        sub = disp[disp["grp"] == gname]
        ax.plot(sub["hour"].dt.tz_localize(None), sub["dev_bps"].rolling(6, 1).mean(),
                color=c, lw=1.6, label=gname)
    ax.axvline(pd.Timestamp("2025-05-19 21:45"), color=EVENT_C, lw=1.2, ls="--")
    ax.text(pd.Timestamp("2025-05-19 21:45"), ax.get_ylim()[1], "  Cloture passes\n  (May 19, 22:00 UTC)", color=EVENT_C,
            fontsize=13, va="top", ha="left")
    ax.set_ylabel("Cross-sectional std of peg deviation (bps, 6h avg)", fontsize=17)
    ax.set_title("Price Dispersion around the Cloture Vote", fontsize=19)
    ax.legend(fontsize=13, frameon=False)
    fig.savefig(os.path.join(out_dir, "fig8_dispersion_around_vote_EN.png"),
                dpi=300, bbox_inches="tight"); plt.close(fig)
    print("  [fig] fig8_dispersion_around_vote_EN.png")

    
    vol = hh.dropna(subset=["lvol"]).groupby(["grp", "hour"])["lvol"].mean().reset_index()
    fig, ax = _new_ax(figsize=(11.5,6.8))
    for gname, c in [("Legitimate (top 15)", NAVY), ("Control", GREY)]:
        sub = vol[vol["grp"] == gname]
        ax.plot(sub["hour"].dt.tz_localize(None), sub["lvol"], color=c, lw=1.4, label=gname)
    ax.axvspan(pd.Timestamp("2025-05-17"), pd.Timestamp("2025-05-19"),
               color='#f0e6c8', alpha=0.25)
    ax.text(pd.Timestamp("2025-05-18"), ax.get_ylim()[1], "weekend", ha="center",
            va="top", fontsize=9, color='#8a7a3d')
    ax.axvline(pd.Timestamp("2025-05-19 21:45"), color=EVENT_C, lw=1.2, ls="--")
    ax.text(pd.Timestamp("2025-05-19 21:45"), ax.get_ylim()[1], "  Cloture passes\n  (May 19, 22:00 UTC)", color=EVENT_C,
            fontsize=13, va="top", ha="left")
    ax.set_ylabel("Mean log 24h volume", fontsize=17)
    ax.set_title("Trading Volume around the Vote", fontsize=19)
    ax.legend(frameon=False)
    fig.savefig(os.path.join(out_dir, "fig9_volume_around_vote_EN.png"),
                dpi=300, bbox_inches="tight"); plt.close(fig)
    print("  [fig] fig9_volume_around_vote_EN.png")


def main():
    ap = argparse.ArgumentParser(description="GENIUS 稳定币合法性溢价：完整复现")
    ap.add_argument("--data_dir", default="./data", help="原始数据文件夹")
    ap.add_argument("--out_dir", default="./results", help="结果输出文件夹")
    ap.add_argument("--nperm", type=int, default=int(os.environ.get("NPERM", 1000)),
                    help="随机化推断/置换次数（默认 1000；调小可加速试跑）")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    nperm = args.nperm
    print("=" * 72)
    print(f"GENIUS 稳定币合法性溢价 复现运行 | data={args.data_dir} | "
          f"out={args.out_dir} | nperm={nperm} | seed={SEED}")
    print("=" * 72)

    
    print("\n[S1] 合法性评分与 Stage-1 诊断")
    scores = build_scores(args.data_dir)
    print("\n[S2] 日级面板")
    p = build_daily_panel(args.data_dir, scores)
    p.to_csv(os.path.join(args.out_dir, "panel_daily.csv"), index=False)
    print("\n[S2b] 小时级面板（5/16–5/22）")
    h144 = build_hourly_144h(args.data_dir)
    h144.to_csv(os.path.join(args.out_dir, "panel_hourly_144h.csv"), index=False)
    b = load_bitstamp(args.data_dir)
    hlong = load_hourly_long(args.data_dir)

    
    print("\n[S3.1] 头条结果（5/19 ±30d，供给增长）")
    run_headline(p, nperm)
    print("\n[S3.2] 窗口稳健性")
    run_windows(p, nperm)
    print("\n[S3.3] 事件序列证伪（7 个候选日期）")
    es_df = run_event_series(p, nperm)
    print("\n[S3.4] Leave-one-out")
    run_loo(p, nperm)
    print("\n[S3.5] 币级截面 DiD")
    run_cross_section(p, nperm)
    print("\n[S3.6] 48 规格验证网格（价格/失锚/流动性/供给 × 2 处理 × 6 窗口）")
    run_verification_grid(p, nperm)
    print("\n[S3.7] 周/双周动态 + 平行趋势")
    weekly_es, biweekly_es = run_dynamics(p)
    print("\n[S3.8] 统一三阶段 DiD（PRE/预期/事件后）")
    run_unified_3regime(p)
    print("\n[S3.9] 里程碑梯度")
    run_milestone_gradient(p)
    print("\n[S3.10] absdev 敏感性")
    run_absdev_sensitivity(p, nperm)
    print("\n[S3.11] 机制检验（溢价→铸造，局部投影）")
    run_mechanism(p)

    
    print("\n[S4.1] 5/19 小时级 4h 事件研究（absdev）")
    save_sheet("Hourly_May19_4h", h_es(h144, T0_MAY19, score_df=scores))
    print("\n[S4.2] 离散度收敛（老师建议2-B1/B2）")
    run_dispersion(h144, scores, nperm=nperm)
    print("\n[S4.3] 成交量即时性（周末修正，老师建议2-B3）")
    run_volume_tests(h144, scores)
    print("\n[S4.4] Bitstamp USDC 三大事件锚定")
    run_bitstamp_events(b)
    print("\n[S4.5] 7/18 签署日小时级联合检验")
    run_jul18_hourly(hlong, scores)

    
    print("\n[S5] 生成英文图")
    make_figures(p, h144, b, scores, args.out_dir, weekly_es, biweekly_es, es_df)

    
    wb = os.path.join(args.out_dir, "复现结果工作簿.xlsx")
    with pd.ExcelWriter(wb, engine="openpyxl") as w:
        for name, df in SHEETS.items():
            df.to_excel(w, sheet_name=name, index=True if "matrix" in name else False)
    print(f"\n[输出] 结果工作簿: {wb}（{len(SHEETS)} 个 sheet）")

    
    print("\n" + "=" * 72)
    print("核对表（参考值来自已交付的结果工作簿；RI p 值随 nperm 略有波动）")
    print("=" * 72)
    refs = [("头条 top15 dsupply_w β", SHEETS["PureSupply_headline"].iloc[0]["beta"], 0.68, 0.10),
            ("头条 GENIUS_z β", SHEETS["PureSupply_headline"].iloc[1]["beta"], 0.26, 0.08),
            ("纯供给 top15 β", SHEETS["PureSupply_headline"].iloc[2]["beta"], 0.72, 0.10)]
    for name, got, exp, tol in refs:
        ok = "OK " if abs(got - exp) <= tol else "差异!"
        print(f"  [{ok}] {name}: {got:.3f}（参考 {exp}，容差 {tol}）")
    grid = SHEETS["Verification_grid"]
    
    grid = grid.assign(robust=(grid["p_cluster"] < 0.05) & (grid["p_RI"] < 0.05))
    for oc, note in [("dev_bps_w", "价格：应全不显著"), ("turnover_w", "流动性：应全不显著"),
                     ("absdev_bps_w", "失锚：应基本不显著"), ("dsupply_w", "供给：应多数显著")]:
        sub = grid[grid["outcome"] == oc]
        nsig = int(sub["robust"].sum())
        print(f"  {oc:14s} 稳健显著 {nsig}/{len(sub)}  —— {note}")
    print("\n完成。")

if __name__ == "__main__":
    main()
