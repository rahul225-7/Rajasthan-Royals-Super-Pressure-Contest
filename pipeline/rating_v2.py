"""
SupeRR Selector S2 - Round 1 - ratings (step 2 of 2). Needs balls_v2.pkl from pressure_v2.py.
Builds context-adjusted components on pressure balls, empirical-Bayes shrinkage,
reliability-based weights, robustness checks, final top-10 lists.
Run from the repo root: python pipeline/rating_v2.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = os.environ.get("SUPERR_OUT", "outputs/")           # results folder (same as pressure_v2.py)
SEED = 7
ELIG = {"bat": (60, 12), "bowl": (60, 20)}     # (pressure balls, matches with pressure balls)
W_KNOCKOUT, W_AWAY = 2.0, 1.25
W_CAPTAIN = 1.0          # captaincy tested (see tests()): no clear effect, so no weight; 1.5 is a sensitivity scenario
TACTICAL_WIDE = 0.5      # isolated wide charged at half; repeat wide / no-ball at full
PRESSURE_Q = 0.75
NB_BAT_RUNS = 1.5
NON_BOWLER = ["RunOut", "Retire Out", "Obstracting the field", "Obstructing the field"]


# ------------------------------------------------------------------ per-ball values
def prep(b, q=PRESSURE_Q, use_weights=True, phase_rel=True, w_cap=None):
    w_cap = W_CAPTAIN if w_cap is None else w_cap
    b = b.copy()
    if phase_rel:   # top 25% of leverage within each phase (PP, 7-10, 11-15, death)
        cut = b.groupby("phase", observed=True).LI.transform(lambda s: s.quantile(q))
    else:
        cut = b.LI.quantile(q)
    b["pressure"] = (b.LI >= cut).astype(int)
    b["collapse"] = ((b.wk_last12 >= 2) & b.overid.between(7, 15)).astype(int)
    ko = np.where(b.knock_out == 1, W_KNOCKOUT, 1.0)
    bat_away = (b.away == 1)
    bowl_away = (b.is_t20i == 1) & (b.away == 0)
    if use_weights:
        b["w_bat"] = ko * np.where(b.striker_cap == 1, w_cap, 1) * np.where(bat_away, W_AWAY, 1)
        b["w_bowl"] = ko * np.where(b.bowler_cap == 1, w_cap, 1) * np.where(bowl_away, W_AWAY, 1)
    else:
        b["w_bat"] = 1.0; b["w_bowl"] = 1.0

    # value of a wicket in runs, from win-probability on pressure balls
    hp = b[(b.pressure == 1) & b.dwp.notna()]
    A = np.c_[np.ones(len(hp)), hp.tot, hp.is_wicket]
    coef = np.linalg.lstsq(A, hp.dwp, rcond=None)[0]
    c_w = float(-coef[2] / coef[1])

    # ---- batter view (balls faced = not wides)
    nw = 1 - b.p_wd
    e_bat = (b.p_1 + 2 * b.p_2 + 3 * b.p_3 + 4 * b.p_4 + 6 * b.p_6 + NB_BAT_RUNS * b.p_nb) / nw
    faced = b.is_wide == 0
    b["s_out"] = ((b.is_wicket == 1) & (b.dismissed_player_id == b.striker_id)).astype(int)
    b["bat_dot"] = ((b.runs == 0) & (b.legal == 1)).astype(int)
    e_bat *= b.loc[faced, "runs"].sum() / e_bat[faced].sum()
    e_out = b.p_W / nw; e_out *= b.loc[faced, "s_out"].sum() / e_out[faced].sum()
    e_dot = b.p_0 / nw; e_dot *= b.loc[faced, "bat_dot"].sum() / e_dot[faced].sum()
    b["bat_rae"] = b.runs - e_bat
    b["bat_surv"] = e_out - b.s_out
    b["bat_rv"] = b.bat_rae + c_w * b.bat_surv
    b["bat_dotav"] = e_dot - b.bat_dot
    ns_out = (b.is_wicket == 1) & (b.s_out == 0)
    b["bat_wpa"] = np.where(ns_out, 0.0, b.dwp) * 100
    b["faced"] = faced.astype(int)

    # ---- bowler view (all balls)
    b["bw"] = ((b.is_wicket == 1) & ~b.dismissal_type.isin(NON_BOWLER)).astype(int)
    e_conc = b.exp_runs * b.conc.sum() / b.exp_runs.sum()
    e_bw = b.p_W * b.bw.sum() / b.p_W.sum()
    b["bowl_rs"] = e_conc - b.conc
    b["bowl_wae"] = b.bw - e_bw
    b["bowl_rv"] = b.bowl_rs + c_w * b.bowl_wae
    b["bowl_wpa"] = np.where((b.is_wicket == 1) & (b.bw == 0), 0.0, -b.dwp) * 100
    repeat = (b.is_wide == 1) & (b.consec_wides >= 1)
    charge = b.is_nb + np.where(repeat, 1.0, TACTICAL_WIDE) * b.is_wide
    e_x = (b.p_wd + b.p_nb); e_x *= charge.sum() / e_x.sum()
    b["bowl_disc"] = e_x - charge
    b["half"] = b.match_id % 2
    return b, c_w


# ------------------------------------------------------------------ aggregation
def eb(df, idc, val, w):
    """Weighted mean per player + empirical-Bayes shrinkage toward league mean."""
    d = df[[idc, val, w]].dropna()
    d = d.assign(wv=d[w] * d[val], w2=d[w] ** 2)
    g = d.groupby(idc).agg(sw=(w, "sum"), swv=("wv", "sum"), sw2=("w2", "sum"), n=(val, "size"))
    g["mean"] = g.swv / g.sw
    g["neff"] = g.sw ** 2 / g.sw2
    mu = d.wv.sum() / d[w].sum()
    sig2 = np.average((d[val] - mu) ** 2, weights=d[w])
    big = g[g.neff >= 30]
    tau2 = np.average((big["mean"] - mu) ** 2, weights=big.neff) - sig2 * np.average(1 / big.neff, weights=big.neff)
    tau2 = max(tau2, sig2 / 5000)             # floor: K never above 5000 balls
    k = sig2 / tau2
    g["shrunk"] = mu + (g["mean"] - mu) * g.neff / (g.neff + k)
    return g[["n", "neff", "mean", "shrunk"]], k


K_GRID = [50, 100, 200, 400, 800, 1600, 3200]
CORE = {"bat": ["runs", "wkt", "wpa"], "bowl": ["rv"]}        # candidates
MODS = {"bat": ["dots", "collapse"], "bowl": ["disc", "collapse"]}
VENUE_BALANCED = False
MIN_RHO = 0.10          # a component needs this held-out rank correlation to carry weight
W_CORE, W_DELTA, W_CONSIST, W_MODS = 0.55, 0.15, 0.10, 0.20


def spec(b, role):
    """Each component: (value column, 'pressure-side' mask). Prior = same player on the other balls."""
    if role == "bat":
        d = b[b.faced == 1]
        idc, w, nm = "striker_id", "w_bat", "striker_name"
        hp = d.pressure == 1
        sp = {"runs": ("bat_rae", hp), "wkt": ("bat_surv", hp), "wpa": ("bat_wpa", hp), "rv": ("bat_rv", hp),
              "dots": ("bat_dotav", hp & (d.innings == 2) & (d.rrr >= 9)),
              "collapse": ("bat_rv", (d.wk_last18 >= 2) & (d.overid <= 15))}
    else:
        d = b
        idc, w, nm = "bowler_id", "w_bowl", "bowler_name"
        hp = d.pressure == 1
        sp = {"runs": ("bowl_rs", hp), "wkt": ("bowl_wae", hp), "wpa": ("bowl_wpa", hp), "rv": ("bowl_rv", hp),
              "disc": ("bowl_disc", hp),
              "collapse": ("bowl_rv", d.overid.between(7, 10))}
    return d, idc, w, nm, sp


def side_stats(d, idc, col, mask, w):
    """Pressure side: weighted mean + effective n. Other side: EB-shrunk usual level.
    VENUE_BALANCED=True combines per-venue means with sqrt weights (sensitivity check only:
    it lowered batting split-half reliability from 0.28 to 0.19, so it is not in the rating)."""
    p = d[mask]
    p = p.assign(wv=p[w] * p[col], w2=p[w] ** 2)
    g = p.groupby(idc).agg(sw=(w, "sum"), sw2=("w2", "sum"), n=(col, "size"))
    pv = p.groupby([idc, "venue_id"]).agg(sw=(w, "sum"), swv=("wv", "sum"))
    pv["rw"] = np.sqrt(pv.sw); pv["mrw"] = pv.swv / pv.sw * pv.rw
    vb = pv.mrw.groupby(level=0).sum() / pv.rw.groupby(level=0).sum()
    if not VENUE_BALANCED:
        vb = pv.swv.groupby(level=0).sum() / pv.sw.groupby(level=0).sum()     # plain weighted mean
    o = d[~mask].assign(one=1.0)
    u, _ = eb(o, idc, col, "one")
    mu = o[col].mean()
    t = pd.DataFrame({"p": vb, "neff": g.sw ** 2 / g.sw2, "n": g.n}).join(u.shrunk.rename("u"), how="left")
    t["u"] = t.u.fillna(mu)
    return t


def blend(t, K):
    return t.u + (t.p - t.u) * t.neff / (t.neff + K)


def tune(b, role):
    """Pick K per component: best held-out prediction of pressure-side performance (odd vs even matches)."""
    d, idc, w, nm, sp = spec(b, role)
    Ks, rho = {}, {}
    for c, (col, mask) in sp.items():
        h = [side_stats(d[d.half == k], idc, col, mask[d.half == k], w) for k in (0, 1)]
        best = None
        for K in K_GRID:
            mse, rr = [], []
            for a_, b_ in ((0, 1), (1, 0)):
                j = h[a_].join(h[b_], lsuffix="a", rsuffix="b", how="inner")
                j = j[(j.na >= 30) & (j.nb >= 50)]
                if len(j) < 20:
                    continue
                pred = j.ua + (j.pa - j.ua) * j.neffa / (j.neffa + K)
                mse.append(np.average((pred - j.pb) ** 2, weights=j.nb)); rr.append(spearmanr(pred, j.pb)[0])
            if mse and (best is None or np.mean(mse) < best[0]):
                best = (np.mean(mse), K, np.mean(rr))
        Ks[c], rho[c] = (best[1], best[2]) if best else (800, 0.0)
    return Ks, rho


def components(b, role, Ks):
    d, idc, w, nm, sp = spec(b, role)
    out = {}
    for c, (col, mask) in sp.items():
        t = side_stats(d, idc, col, mask, w)
        out[c] = blend(t, Ks[c])
        out[c + "_raw"] = t.p
        out[c + "_usual"] = t.u
        out[c + "_n"] = t.n
        if c == "rv":
            out["delta"] = out[c] - t.u          # shrunk rise/dip vs own usual level
            out["delta_raw"] = t.p - t.u
            out["p_balls"] = t.n
    hp = d[sp["rv"][1]]
    rvcol = sp["rv"][0]
    out["p_matches"] = hp.groupby(idc).match_id.nunique()
    out["n_venues"] = hp.groupby(idc).venue_id.nunique()
    out["o_balls"] = d[~sp["rv"][1]].groupby(idc).size()
    # consistency: share of pressure matches with value above expectation (beta-shrunk, prior 10 matches)
    pm = hp.assign(v=hp[rvcol] * hp[w]).groupby([idc, "match_id"]).v.sum().gt(0)
    c_ = pm.groupby(level=0).agg(["sum", "size"])
    out["consist"] = (c_["sum"] + 10 * pm.mean()) / (c_["size"] + 10)
    out["consist_raw"] = c_["sum"] / c_["size"]
    t = pd.DataFrame(out)
    t.insert(0, "name", d.groupby(idc)[nm].last())
    return t


def make_weights(rho, role):
    """Core 55% and modules 20% are split by held-out rho among components that reach MIN_RHO.
    Change-vs-usual 15% and consistency 10% are fixed because the brief names them."""
    core = {c: rho[c] for c in CORE[role] if rho[c] >= MIN_RHO} or {CORE[role][0]: 1.0}
    mods = {c: rho[c] for c in MODS[role] if rho[c] >= MIN_RHO}
    wc = W_CORE + (0 if mods else W_MODS)
    w = {c: wc * v / sum(core.values()) for c, v in core.items()}
    w.update({c: W_MODS * v / sum(mods.values()) for c, v in mods.items()})
    w["delta"], w["consist"] = W_DELTA, W_CONSIST
    return w


def composite(t, weights, role):
    mb, mm = ELIG[role]
    e = t[(t.p_balls >= mb) & (t.p_matches >= mm)].copy()
    z = pd.DataFrame({c: (e[c] - e[c].mean()) / e[c].std() for c in weights}).fillna(0)
    e["rating"] = sum(z[c] * w for c, w in weights.items())
    for c in weights:
        e["z_" + c] = z[c]
    return e.sort_values("rating", ascending=False)


def run(b, role, cfg=None, verbose=False):
    if cfg is None:
        Ks, rho = tune(b, role)
        cfg = (Ks, make_weights(rho, role))
        if verbose:
            print(f"\n[{role}] component: held-out rho / shrink K / weight")
            for c in ["runs", "wkt", "wpa", "rv"] + MODS[role]:
                print(f"  {c:9s} rho {rho[c]:+.2f}  K {Ks[c]:5d}  weight {cfg[1].get(c, 0):.3f}")
            print("  delta weight 0.150, consist weight 0.100")
    return composite(components(b, role, cfg[0]), cfg[1], role), cfg


# ------------------------------------------------------------------ tests
def boot_ci(x, y, n=1000, rng=None):
    rng = rng or np.random.default_rng(SEED)
    d = [rng.choice(x, len(x)).mean() - rng.choice(y, len(y)).mean() for _ in range(n)]
    return np.mean(x) - np.mean(y), np.percentile(d, 2.5), np.percentile(d, 97.5)


def tests(b):
    print("\n================ MODULE TESTS ================")
    b = b.sort_values(["match_id", "innings", "innings_ball_num"])
    g = b.groupby(["match_id", "innings"])
    for k in (1, 2, 3):
        b[f"n{k}_resid"] = g.bowl_rs.shift(-k) * -1     # runs above expected on kth next ball
        b[f"n{k}_wide"] = g.is_wide.shift(-k)
    b["nx3"] = b[["n1_resid", "n2_resid", "n3_resid"]].sum(axis=1, min_count=3)
    hp = b[b.pressure == 1]
    print("Extras under pressure: runs ABOVE EXPECTED on following balls (context-adjusted)")
    base = hp[(hp.is_wide == 0) & (hp.is_nb == 0)]
    rows = []
    for label, m in [("no-ball", hp.is_nb == 1), ("isolated wide", (hp.is_wide == 1) & (hp.consec_wides == 0)),
                     ("repeat wide", (hp.is_wide == 1) & (hp.consec_wides >= 1))]:
        for col, nm in [("n1_resid", "next ball"), ("nx3", "next 3 balls")]:
            x, y = hp.loc[m, col].dropna().values, base[col].dropna().values
            d, lo, hi = boot_ci(x, y)
            rows.append((label, nm, len(x), round(d, 3), round(lo, 3), round(hi, 3), "YES" if lo > 0 or hi < 0 else "no"))
    print(pd.DataFrame(rows, columns=["extra", "window", "n", "diff_runs", "ci_lo", "ci_hi", "clear"]).to_string(index=False))
    x = hp.loc[hp.is_wide == 1, "n1_wide"].dropna(); y = base.n1_wide.dropna()
    print(f"P(next ball is a wide): after wide {x.mean():.3f} vs after normal ball {y.mean():.3f}; ordinary balls: "
          f"{b.loc[(b.pressure==0)&(b.is_wide==1),'n1_wide'].mean():.3f} vs {b.loc[(b.pressure==0)&(b.is_wide==0),'n1_wide'].mean():.3f}")
    print(f"wide rate: pressure {hp.is_wide.mean():.4f} vs ordinary {b[b.pressure==0].is_wide.mean():.4f}; "
          f"no-ball rate: {hp.is_nb.mean():.4f} vs {b[b.pressure==0].is_nb.mean():.4f}")

    print("\nDot balls in steep chases (innings 2, pressure, RRR>=9): next ball after a dot vs after a scoring ball")
    f = b[(b.faced == 1)].copy()
    gs = f.groupby(["match_id", "innings", "striker_id"])
    f["prev_dot"] = gs.bat_dot.shift(1)
    st = f[(f.pressure == 1) & (f.innings == 2) & (f.rrr >= 9) & f.prev_dot.notna()]
    for col, nm in [("s_out", "dismissal rate"), ("bat_rae", "runs above expected")]:
        x, y = st.loc[st.prev_dot == 1, col].values, st.loc[st.prev_dot == 0, col].values
        d, lo, hi = boot_ci(x, y)
        print(f"  {nm}: after dot {x.mean():.4f} vs after scoring {y.mean():.4f}  diff {d:+.4f} [{lo:+.4f},{hi:+.4f}]  n {len(x)}/{len(y)}")

    print("\nCaptains under pressure (same players, captain vs not, pressure balls)")
    for role, idc, cap, val in [("bat", "striker_id", "striker_cap", "bat_rv"), ("bowl", "bowler_id", "bowler_cap", "bowl_rv")]:
        d = b[(b.pressure == 1) & ((b.faced == 1) if role == "bat" else True)]
        both = d.groupby(idc)[cap].agg(["min", "max", "sum", "size"])
        ids = both[(both["sum"] >= 30) & (both["size"] - both["sum"] >= 30)].index
        dd = d[d[idc].isin(ids)]
        pm = dd.groupby([idc, cap])[val].mean().unstack()
        diff = (pm[1] - pm[0])
        rng = np.random.default_rng(SEED)
        bs = [rng.choice(diff.values, len(diff)).mean() for _ in range(2000)]
        print(f"  {role}: {len(ids)} players; value/ball as captain minus not: {diff.mean():+.4f} "
              f"[{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}]; all captain balls {d[d[cap]==1][val].mean():+.4f} vs others {d[d[cap]==0][val].mean():+.4f}")

    print("\nCollapse states (2+ wickets in last 12 balls, overs 7-15) vs other balls in overs 7-15")
    mid = b[b.overid.between(7, 15)]
    print(mid.groupby("collapse").agg(balls=("tot", "size"), run_rate=("tot", lambda s: s.mean() * 6),
                                      wkts_per_100=("is_wicket", lambda s: s.mean() * 100),
                                      dot_pct=("is_dot", "mean"), mean_wp=("wp", "mean"), mean_LI=("LI", "mean")).round(3).to_string())
    print("\nKnockout vs other matches: pressure share and mean LI")
    print(b.groupby("knock_out").agg(matches=("match_id", "nunique"), press=("pressure", "mean"), LI=("LI", "mean")).round(3).to_string())
    return b


# ------------------------------------------------------------------ robustness
def bootstrap_ranks(b, role, weights, n_boot=150):
    rng = np.random.default_rng(SEED)
    keep = ["match_id", "half", "striker_id", "striker_name", "bowler_id", "bowler_name", "venue_id", "pressure",
            "collapse", "faced", "innings", "rrr", "overid", "wk_last12", "wk_last18", "w_bat", "w_bowl", "bat_rae", "bat_surv",
            "bat_wpa", "bat_rv", "bat_dotav", "bowl_rs", "bowl_wae", "bowl_wpa", "bowl_rv", "bowl_disc"]
    b = b[keep]
    mids = b.match_id.unique()
    grp = {m: g for m, g in b.groupby("match_id")}
    ranks = []
    cfg = weights
    for i in range(n_boot):
        samp = rng.choice(mids, len(mids))
        bb = pd.concat([grp[m] for m in samp], ignore_index=True)
        bb["match_id"] = np.repeat(np.arange(len(samp)), [len(grp[m]) for m in samp])
        e, _ = run(bb, role, cfg)
        ranks.append(e.rating.rank(ascending=False))
    R = pd.concat(ranks, axis=1)
    return pd.DataFrame({"rank_p05": R.quantile(0.05, axis=1), "rank_med": R.median(axis=1),
                         "rank_p95": R.quantile(0.95, axis=1), "p_top10": (R <= 10).mean(axis=1),
                         "p_elig": R.notna().mean(axis=1)})


def sensitivity(raw, role, weights, base_top):
    rows = []
    def add(label, e):
        top = list(e.head(10).index)
        j = e.rating.reindex(base_full.index).dropna()
        rho = spearmanr(j, base_full.rating.reindex(j.index))[0]
        rows.append((label, len(set(top) & set(base_top)), round(rho, 2), len(e)))
    b0, _ = prep(raw)
    global base_full
    base_full, _ = run(b0, role, weights)
    for q in (0.80, 0.67):
        add(f"pressure = top {round((1-q)*100)}% LI", run(prep(raw, q=q)[0], role, weights)[0])
    add("no knockout/away weights", run(prep(raw, use_weights=False)[0], role, weights)[0])
    add("captain balls weighted 1.5x", run(prep(raw, w_cap=1.5)[0], role, weights)[0])
    add("without rain matches", run(prep(raw[raw.revised_target.isna()])[0], role, weights)[0])
    add("without emerging-team events", run(prep(raw[(raw.is_ipl == 1) | (raw.is_t20i == 1)])[0], role, weights)[0])
    for s in sorted(raw.season.astype(str).str[:4].unique()):
        add(f"leave out season {s}", run(prep(raw[raw.season.astype(str).str[:4] != s])[0], role, weights)[0])
    eq = {c: 1 / len(weights[1]) for c in weights[1]}
    add("equal component weights", run(b0, role, (weights[0], eq))[0])
    global VENUE_BALANCED
    VENUE_BALANCED = True
    add("venue-balanced pressure means", run(b0, role, weights)[0])
    VENUE_BALANCED = False
    add("one cutoff for all phases (not phase-relative)", run(prep(raw, phase_rel=False)[0], role, weights)[0])
    cs = sum(v for c, v in weights[1].items() if c in CORE[role])
    lvl = {c: (v / cs if c in CORE[role] else 0.0) for c, v in weights[1].items()}
    add("core components only (no delta/consistency/modules)", run(b0, role, (weights[0], lvl))[0])
    return pd.DataFrame(rows, columns=["scenario", "top10_overlap", "rank_rho_all", "eligible"])


def main():
    raw = pd.read_pickle(OUT + "balls_v2.pkl")
    b, c_w = prep(raw)
    print(f"wicket value = {c_w:.1f} runs (from win-probability on pressure balls)")
    print("pressure balls:", int(b.pressure.sum()), "of", len(b))
    b = tests(b)
    fin, res = [], {}
    for role in ("bat", "bowl"):
        e, w = run(b, role, verbose=True)
        boot = bootstrap_ranks(b, role, w)
        e = e.join(boot)
        e.insert(1, "rank", np.arange(1, len(e) + 1))

        e.round(4).to_csv(OUT + ("batters_pressure_v2.csv" if role == "bat" else "bowlers_pressure_v2.csv"))
        t = e.head(10)[["rank", "name", "p_balls", "p_matches", "rating", "p_top10", "rank_p05", "rank_p95"]].copy()
        t.insert(0, "list", "Batters" if role == "bat" else "Bowlers"); t.insert(2, "player_id", t.index)
        fin.append(t)
        res[role] = (e, w)
        cols = ["rank", "name", "p_balls", "p_matches", "rating", "runs_raw", "runs_usual", "wkt_raw", "wkt_usual", "wpa_raw",
                "rv_raw", "rv_usual", "delta", "consist_raw", "n_venues", "rank_p05", "rank_p95", "p_top10"]
        print(f"\n== TOP 20 {role} (eligible {len(e)}) ==\n", e.head(20)[cols].round(3).to_string())
        s = sensitivity(raw, role, w, list(e.head(10).index))
        s.to_csv(OUT + f"sensitivity_{role}.csv", index=False)
        print("\n", s.to_string(index=False))
        rhos, ns = [], []
        mids = b.match_id.unique()
        for sd in range(20):                      # 20 random match splits
            hm = pd.Series(np.random.default_rng(sd).integers(0, 2, len(mids)), index=mids)
            hh = b.match_id.map(hm)
            h = [run(b[hh == k], role, w)[0].rating for k in (0, 1)]
            j = pd.concat(h, axis=1, join="inner").dropna()
            rhos.append(spearmanr(j.iloc[:, 0], j.iloc[:, 1])[0]); ns.append(len(j))
        print(f"\nFINAL RATING split-half rho ({role}): mean {np.mean(rhos):.2f} over 20 random splits, about {int(np.mean(ns))} players per split")
    pd.concat(fin).round(3).to_csv(OUT + "model_top10.csv", index=False)


if __name__ == "__main__":
    main()
