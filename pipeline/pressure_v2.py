"""
SupeRR Selector S2 - Round 1 - v2 pipeline (features + models 1-3).
1. Win probability (monotonic gradient boosting, out-of-fold by match)
2. Ball outcome model (9 classes)
3. Exact Leverage Index = sum_k P(k) * |WP(after k) - WP(now)|
Run from the repo root: python pipeline/pressure_v2.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

D = os.environ.get("SUPERR_DATA", "data/raw/")          # the six supplied CSVs
OUT = os.environ.get("SUPERR_OUT", "outputs/")           # results folder
SEED = 7
PRESSURE_Q = 0.75
K_VENUE = 5          # venue par shrinkage (matches)
W_KNOCKOUT = 2.0
W_CAPTAIN = 1.5
W_AWAY = 1.25        # T20I only, batting/bowling side playing away

NB_TYPES = ["Overstep", "other"]
ILLEGAL = ["wide"] + NB_TYPES


# ----------------------------------------------------------------- load
def load():
    b = pd.read_csv(D + "ball_by_ball.csv")
    m = pd.read_csv(D + "matches.csv")
    c = pd.read_csv(D + "competitions.csv")
    xi = pd.read_csv(D + "playing_eleven.csv")
    m = m.merge(c[["comp_id", "comp_name", "match_format", "season"]], on="comp_id")
    m["is_ipl"] = m.comp_name.str.startswith("IPL").astype(int)
    m["is_t20i"] = (m.match_format == "T20I").astype(int)
    m["impact_era"] = ((m.is_ipl == 1) & (m.season.astype(str) >= "2023")).astype(int)
    m["decided"] = m.win_team_id.notna() & (m.win_team_id > 0)
    m["knock_out"] = m.knock_out.fillna(0).astype(int)
    # keep every match with ball data; super overs excluded from WPA
    b = b[b.innings <= 2].copy()
    b = b.merge(m[["match_id", "win_team_id", "decided", "is_ipl", "is_t20i", "impact_era",
                   "season", "knock_out", "stage", "venue_id", "home_team_id",
                   "first_inn_revised_overs", "second_inn_revised_overs", "revised_target"]],
                on="match_id")
    cap = xi[xi.is_captain == 1][["match_id", "player_id"]].assign(cap=1)
    b = b.merge(cap.rename(columns={"player_id": "striker_id", "cap": "striker_cap"}),
                on=["match_id", "striker_id"], how="left")
    b = b.merge(cap.rename(columns={"player_id": "bowler_id", "cap": "bowler_cap"}),
                on=["match_id", "bowler_id"], how="left")
    b[["striker_cap", "bowler_cap"]] = b[["striker_cap", "bowler_cap"]].fillna(0).astype(int)
    return b.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)


# ----------------------------------------------------------------- features
def venue_par(b):
    """Leave-one-out, shrunk mean first-innings total per venue (full 20-over innings only)."""
    i1 = b[(b.innings == 1) & b.first_inn_revised_overs.isna()]
    tot = i1.groupby("match_id").agg(tot=("tot", "sum"), lb=("legal", "sum"), wk=("is_wicket", "sum"),
                                     venue_id=("venue_id", "first"), is_ipl=("is_ipl", "first"))
    tot = tot[(tot.lb >= 114) | (tot.wk == 10)]           # full innings or all out
    grp_mean = tot.groupby("is_ipl").tot.mean()
    vs = tot.groupby("venue_id").tot.agg(["sum", "count"])
    tot = tot.join(vs, on="venue_id")
    n = tot["count"] - 1
    loo = (tot["sum"] - tot.tot) / n.replace(0, np.nan)
    g = tot.is_ipl.map(grp_mean)
    tot["par"] = (n * loo.fillna(g) + K_VENUE * g) / (n + K_VENUE)
    par = tot.par
    # matches without a usable own innings: plain shrunk venue mean
    vmean = (vs["sum"] + K_VENUE * grp_mean.mean()) / (vs["count"] + K_VENUE)
    fallback = b.drop_duplicates("match_id").set_index("match_id").venue_id.map(vmean)
    out = b.match_id.map(par)
    out = out.fillna(b.match_id.map(fallback)).fillna(b.is_ipl.map(grp_mean))
    return out


def features(b):
    b["is_wicket"] = b.is_wicket.astype(int)
    b["tot"] = b.runs + b.extras1_runs + b.extras_2_runs
    b["legal"] = (~b.extras_type.isin(ILLEGAL)).astype(int)
    b["is_wide"] = (b.extras_type == "wide").astype(int)
    b["is_nb"] = b.extras_type.isin(NB_TYPES).astype(int)
    b["is_dot"] = ((b.tot == 0) & (b.is_wicket == 0)).astype(int)
    b["is_bdry"] = ((b.is_four == 1) | (b.is_six == 1)).astype(int)

    g = b.groupby(["match_id", "innings"])
    b["cum_before"] = g.tot.cumsum() - b.tot
    b["wk_before"] = g.is_wicket.cumsum() - b.is_wicket
    b["lb_before"] = g.legal.cumsum() - b.legal
    # innings length: revised overs are stored in BALLS
    full = np.where(b.innings == 1, b.first_inn_revised_overs, b.second_inn_revised_overs)
    b["inn_balls"] = pd.Series(full, index=b.index).fillna(120).astype(int)
    b["balls_left"] = (b.inn_balls - b.lb_before).clip(lower=0)
    b["frac_used"] = b.lb_before / b.inn_balls
    b["crr"] = b.cum_before / (b.lb_before.clip(lower=1) / 6)
    b["proj"] = b.cum_before + b.crr * b.balls_left / 6

    tgt = b[b.innings == 1].groupby("match_id").tot.sum() + 1
    tgt = b.match_id.map(tgt)
    tgt = np.where(b.revised_target.notna(), b.revised_target, tgt)
    b["target"] = np.where(b.innings == 2, tgt, np.nan)
    b["runs_left"] = b.target - b.cum_before
    b["rrr"] = b.runs_left / (b.balls_left.clip(lower=1) / 6)

    b["venue_par"] = venue_par(b)
    b["par_rate"] = b.venue_par / 20
    b["rrr_vs_par"] = b.rrr - b.par_rate
    b["proj_vs_par"] = np.where(b.innings == 1, b.proj - b.venue_par, np.nan)

    # rolling / collapse state (before the ball)
    def roll(col, n):
        return g[col].transform(lambda s: s.shift(1).rolling(n, min_periods=1).sum()).fillna(0)
    b["wk_last12"] = roll("is_wicket", 12)
    b["wk_last18"] = roll("is_wicket", 18)
    b["runs_last12"] = roll("tot", 12)
    b["dots_last6"] = roll("is_dot", 6)
    b["bdry_last12"] = roll("is_bdry", 12)
    b["prev_wide"] = g.is_wide.shift(1).fillna(0)
    b["free_hit"] = g.is_nb.shift(1).fillna(0)
    # consecutive wides before this ball
    cw = np.zeros(len(b)); run = 0; last = None
    for i, (k, w) in enumerate(zip(zip(b.match_id, b.innings), b.is_wide)):
        if k != last: run = 0; last = k
        cw[i] = run; run = run + 1 if w else 0
    b["consec_wides"] = cw

    # balls since last wicket / boundary, partnership
    ball_idx = g.cumcount()
    def since(flag):
        last_pos = b[flag].replace(0, np.nan) * ball_idx
        last_pos = last_pos.groupby([b.match_id, b.innings]).ffill().shift(1)
        return (ball_idx - last_pos).fillna(ball_idx + 1)
    b["balls_since_wkt"] = since("is_wicket")
    b["balls_since_bdry"] = since("is_bdry")
    pk = g.is_wicket.cumsum() - b.is_wicket           # partnership id = wk_before
    pg = b.groupby(["match_id", "innings", pk])
    b["pship_runs"] = pg.tot.cumsum() - b.tot
    b["pship_balls"] = pg.cumcount()
    # striker's balls faced so far this innings
    sg = b.groupby(["match_id", "innings", "striker_id"])
    b["bat_balls"] = sg.legal.cumsum() - b.legal
    b["bat_runs"] = sg.runs.cumsum() - b.runs
    b["new_batter"] = (b.bat_balls < 6).astype(int)
    b["set_batter"] = (b.bat_balls >= 20).astype(int)
    # bowler: runs conceded in his previous over this innings
    b["conc"] = b.runs + np.where(b.legal == 0, b.extras1_runs + b.extras_2_runs, 0)
    ov = b.groupby(["match_id", "innings", "bowler_id", "overid"]).conc.sum().rename("ov_conc").reset_index()
    ov["prev_over_conc"] = ov.groupby(["match_id", "innings", "bowler_id"]).ov_conc.shift(1)
    b = b.merge(ov[["match_id", "innings", "bowler_id", "overid", "prev_over_conc"]],
                on=["match_id", "innings", "bowler_id", "overid"], how="left")
    b["bowler_overs_done"] = b.groupby(["match_id", "innings", "bowler_id"]).overid.transform(
        lambda s: s.rank(method="dense") - 1)
    b["mid_window"] = b.overid.between(7, 10).astype(int)
    b["phase"] = pd.cut(b.overid, [0, 6, 10, 15, 20], labels=["PP", "7-10", "11-15", "Death"])
    b["away"] = ((b.is_t20i == 1) & (b.batting_team_id != b.home_team_id)).astype(int)
    b["won"] = np.where(b.decided, (b.batting_team_id == b.win_team_id).astype(float), np.nan)
    return b


# ----------------------------------------------------------------- models
X_WP = ["innings", "balls_left", "wk_before", "cum_before", "target", "runs_left", "rrr", "crr",
        "proj", "frac_used", "venue_par", "rrr_vs_par", "proj_vs_par", "wk_last12", "wk_last18",
        "runs_last12", "dots_last6", "bdry_last12", "balls_since_wkt", "pship_runs", "pship_balls",
        "bat_balls", "new_batter", "free_hit", "is_ipl", "is_t20i", "impact_era", "away"]
MONO = {"wk_before": -1, "cum_before": 1, "runs_left": -1, "rrr": -1, "crr": 1, "proj": 1,
        "rrr_vs_par": -1, "proj_vs_par": 1, "wk_last12": -1, "wk_last18": -1, "runs_last12": 1,
        "pship_runs": 1, "free_hit": 1}
# expected-remaining-runs model (resource proxy)
X_RES = ["balls_left", "wk_before", "venue_par", "is_ipl", "impact_era", "crr", "frac_used"]
OUT_CLASSES = ["0", "1", "2", "3", "4", "6", "W", "wd", "nb"]
X_OUT = ["innings", "balls_left", "wk_before", "runs_left", "rrr", "rrr_vs_par", "proj_vs_par",
         "crr", "bat_balls", "new_batter", "free_hit", "dots_last6", "balls_since_bdry",
         "wk_last12", "pship_balls", "is_ipl", "impact_era", "bowling_code", "bat_code", "overid",
         "venue_par", "frac_used"]


INTER = ["m_per_ball", "wk_x_bl", "rrr_x_bl", "m2", "m1", "wkleft_x_rl"]


def add_inter(X):
    X = X.copy()
    X["m_per_ball"] = X.margin / X.balls_left.clip(lower=1)
    X["wk_x_bl"] = X.wk_before * X.balls_left
    X["rrr_x_bl"] = X.rrr.fillna(0) * X.balls_left
    inn2 = (X.innings == 2).astype(int)
    X["m2"] = X.margin * inn2
    X["m1"] = X.margin * (1 - inn2)
    X["wkleft_x_rl"] = (10 - X.wk_before) * X.runs_left.fillna(0)
    return X


from sklearn.base import BaseEstimator, ClassifierMixin


class WPLogit(BaseEstimator, ClassifierMixin):
    """Logistic win-probability model with interaction terms (chosen: best Brier, transparent)."""
    def __init__(self):
        self.m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               LogisticRegression(max_iter=1000, C=1.0))
    def fit(self, X, y):
        self.m.fit(add_inter(X), y); self.classes_ = self.m.classes_; return self
    def predict_proba(self, X):
        return self.m.predict_proba(add_inter(X))
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def wp_model():
    return WPLogit()


def wp_boost():
    cst = [MONO.get(c, 0) for c in X_WP]
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                          min_samples_leaf=500, l2_regularization=2.0,
                                          monotonic_cst=cst, random_state=SEED)


def fit_wp(b):
    gkf = GroupKFold(n_splits=5)
    tr = b.won.notna()
    X, y, grp = b.loc[tr, X_WP], b.loc[tr, "won"].astype(int), b.loc[tr, "match_id"]
    # resource proxy: expected remaining runs, fit on innings-1 balls
    i1 = b[(b.innings == 1) & (b.inn_balls == 120)]
    rem = i1.groupby("match_id").tot.transform("sum") - i1.cum_before
    reg = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.08, max_iter=150, random_state=SEED)
    reg.fit(i1[X_RES], rem)
    b["exp_rem"] = reg.predict(b[X_RES])
    b["margin"] = np.where(b.innings == 2, b.exp_rem - b.runs_left, b.proj_vs_par)
    if "exp_rem" not in X_WP:
        X_WP.extend(["exp_rem", "margin"]); MONO["margin"] = 1
    X = b.loc[tr, X_WP]
    clf = wp_model()
    oof = cross_val_predict(clf, X, y, groups=grp, cv=gkf, method="predict_proba")[:, 1]
    oof_b = cross_val_predict(wp_boost(), X, y, groups=grp, cv=gkf, method="predict_proba")[:, 1]
    print(f"WP  logistic+interactions (used): Brier {brier_score_loss(y, oof):.4f}  logloss {log_loss(y, oof):.4f}")
    print(f"WP  boosting (comparison):        Brier {brier_score_loss(y, oof_b):.4f}  logloss {log_loss(y, oof_b):.4f}")
    calib = pd.DataFrame({"p": oof, "y": y.values}).assign(bin=lambda d: (d.p * 10).astype(int).clip(0, 9))
    print("calibration (pred bin -> actual win rate):")
    print(calib.groupby("bin").agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size")).round(3).to_string())
    # full model for counterfactual states
    clf.fit(X, y)
    b["wp"] = np.nan
    b.loc[tr, "wp"] = oof
    b.loc[~tr, "wp"] = clf.predict_proba(b.loc[~tr, X_WP])[:, 1]
    return b, clf, reg


def fit_outcome(b):
    b["bowling_code"] = b.bowling_type.astype("category").cat.codes
    b["bat_code"] = (b.batting_type == "LHB").astype(int)
    oc = np.select([b.is_wicket == 1, b.is_wide == 1, b.is_nb == 1],
                   ["W", "wd", "nb"], default=None)
    r = b.tot.clip(upper=6).replace({5: 4}).astype(int).astype(str)
    b["outcome"] = np.where(pd.isna(oc), r, oc)
    gkf = GroupKFold(n_splits=5)
    clf = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.06, max_iter=200,
                                         categorical_features=[X_OUT.index("bowling_code")],
                                         random_state=SEED)
    P = cross_val_predict(clf, b[X_OUT], b.outcome, groups=b.match_id, cv=gkf, method="predict_proba")
    classes = sorted(b.outcome.unique())
    P = pd.DataFrame(P, columns=classes, index=b.index)[OUT_CLASSES]
    print(f"\nOUTCOME model logloss {log_loss(b.outcome, P[classes].values if False else P[OUT_CLASSES].reindex(columns=classes).values, labels=classes):.4f}")
    for k in OUT_CLASSES:
        b["p_" + k] = P[k].values
    # expected runs & wicket prob per ball (baselines for ratings later)
    rv = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "6": 6, "W": 0, "wd": 1, "nb": 1}
    b["exp_runs"] = sum(P[k] * rv[k] for k in OUT_CLASSES)
    b["exp_wkt"] = P["W"]
    return b


def next_state(b, k):
    """Approximate state after outcome k (core state only)."""
    s = b[X_WP].copy()
    dr = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "6": 6, "W": 0, "wd": 1, "nb": 1}[k]
    legal = k not in ("wd", "nb")
    wk = 1 if k == "W" else 0
    s["cum_before"] += dr
    s["runs_left"] -= dr
    s["wk_before"] += wk
    if legal:
        s["balls_left"] = (s.balls_left - 1).clip(lower=0)
    lb = b.inn_balls - s.balls_left
    s["frac_used"] = lb / b.inn_balls
    s["crr"] = s.cum_before / (lb.clip(lower=1) / 6)
    s["proj"] = s.cum_before + s.crr * s.balls_left / 6
    s["rrr"] = s.runs_left / (s.balls_left.clip(lower=1) / 6)
    s["rrr_vs_par"] = s.rrr - b.par_rate
    s["proj_vs_par"] = np.where(b.innings == 1, s.proj - b.venue_par, np.nan)
    s["wk_last12"] += wk; s["wk_last18"] += wk
    s["runs_last12"] += dr
    s["dots_last6"] = (s.dots_last6 + (1 if dr == 0 and not wk else 0)).clip(upper=6)
    s["bdry_last12"] += 1 if dr >= 4 else 0
    s["balls_since_wkt"] = 0 if wk else s.balls_since_wkt + 1
    s["pship_runs"] = 0 if wk else s.pship_runs + dr
    s["pship_balls"] = 0 if wk else s.pship_balls + 1
    s["bat_balls"] = 0 if wk else s.bat_balls + (1 if legal else 0)
    s["new_batter"] = (s.bat_balls < 6).astype(int)
    s["free_hit"] = 1 if k == "nb" else 0
    s["exp_rem"] = b.exp_rem   # refreshed below
    s["margin"] = np.where(b.innings == 2, s.exp_rem - s.runs_left, s.proj_vs_par)
    return s


def terminal_wp(b, s, k, clf, reg, wp_inn2_start):
    """WP after outcome k with innings-end handling."""
    s = s.copy()
    s["exp_rem"] = reg.predict(s[X_RES])
    s["margin"] = np.where(b.innings == 2, s.exp_rem - s.runs_left, s.proj_vs_par)
    wp = clf.predict_proba(s)[:, 1]
    end = (s.balls_left <= 0) | (s.wk_before >= 10)
    i1 = (b.innings == 1).values
    # innings-1 end -> 1 - WP(chase start with this total)
    e1 = end & i1
    if e1.any():
        wp[e1] = 1 - wp_inn2_start(s.loc[e1, "cum_before"] + 1, b.loc[e1])
    # innings-2: chase done / failed
    i2 = ~i1
    done = i2 & (s.runs_left <= 0).values
    wp[done] = 1.0
    failed = i2 & end & (s.runs_left > 0).values
    wp[failed] = np.where(s.loc[failed, "runs_left"] == 1, 0.5, 0.0)
    return wp


def leverage(b, clf, reg):
    def wp_inn2_start(total, rows):
        s = pd.DataFrame(index=rows.index, columns=X_WP, dtype=float)
        s[:] = np.nan
        s["innings"] = 2; s["balls_left"] = 120; s["wk_before"] = 0; s["cum_before"] = 0
        s["target"] = total.values; s["runs_left"] = total.values; s["rrr"] = total.values / 20
        s["crr"] = 0; s["proj"] = 0; s["frac_used"] = 0; s["venue_par"] = rows.venue_par.values
        s["rrr_vs_par"] = s.rrr - rows.par_rate.values; s["proj_vs_par"] = np.nan
        for c in ["wk_last12", "wk_last18", "runs_last12", "dots_last6", "bdry_last12", "pship_runs",
                  "pship_balls", "bat_balls", "free_hit"]:
            s[c] = 0
        s["balls_since_wkt"] = 1; s["new_batter"] = 1
        for c in ["is_ipl", "is_t20i", "impact_era"]:
            s[c] = rows[c].values
        s["away"] = 1 - rows.away.values
        s["exp_rem"] = reg.predict(s[X_RES]); s["margin"] = s.exp_rem - s.runs_left
        return clf.predict_proba(s[X_WP])[:, 1]

    # actual next-state wp (same machinery -> consistent dwp)
    b["wp_next"] = np.nan
    LI = np.zeros(len(b))
    for k in OUT_CLASSES:
        s = next_state(b, k)
        w = terminal_wp(b, s, k, clf, reg, wp_inn2_start)
        LI += b["p_" + k].values * np.abs(w - b.wp.values)
        m = (b.outcome == k).values
        b.loc[m, "wp_next"] = w[m]
    b["dwp"] = b.wp_next - b.wp
    b["LI"] = LI / LI.mean()
    b["pressure"] = (b.LI >= b.groupby("phase", observed=True).LI.transform(lambda s: s.quantile(PRESSURE_Q))).astype(int)
    return b


def report(b):
    print("\n== balls", len(b), "matches", b.match_id.nunique(), "decided", b[b.decided].match_id.nunique())
    t = b.groupby("pressure").agg(balls=("tot", "size"), mean_LI=("LI", "mean"),
                                  run_rate=("tot", lambda s: s.mean() * 6),
                                  wkts_per_100=("is_wicket", lambda s: s.mean() * 100),
                                  dot_pct=("is_dot", "mean"), wide_pct=("is_wide", "mean"),
                                  mean_wp=("wp", "mean"))
    print("\n== pressure vs ordinary ==\n", t.round(3).to_string())
    print("\n== share of balls flagged, innings x phase ==")
    print(b.pivot_table(index="innings", columns="phase", values="pressure", aggfunc="mean", observed=True).round(3).to_string())
    print("\n== mean LI, innings x phase ==")
    print(b.pivot_table(index="innings", columns="phase", values="LI", aggfunc="mean", observed=True).round(2).to_string())
    print("\n== share flagged by competition group ==")
    print(b.assign(grp=np.select([b.is_ipl == 1, b.is_t20i == 1], ["IPL", "T20I"], "Emerging"))
           .groupby("grp").pressure.mean().round(3).to_string())
    print("\n== share flagged, rain vs normal ==")
    print(b.groupby(b.revised_target.notna()).pressure.mean().round(3).to_string())
    # collapse check: pressure share by wickets in last 12 balls, innings 1
    print("\n== innings 1: pressure share by wickets in last 12 balls ==")
    print(b[b.innings == 1].groupby("wk_last12").pressure.agg(["mean", "size"]).round(3).to_string())
    print("\n== LI top deciles: mean |dwp| actual (should rise with LI) ==")
    b["LI_dec"] = pd.qcut(b.LI, 10, labels=False)
    print(b.groupby("LI_dec").agg(LI=("LI", "mean"), abs_dwp=("dwp", lambda s: s.abs().mean())).round(3).to_string())


def main():
    b = features(load())
    print("extras 'other' confirmed as no-ball variant (free hit next ball).")
    b, clf, reg = fit_wp(b)
    b = fit_outcome(b)
    b = leverage(b, clf, reg)
    report(b)
    keep = ["match_id", "innings", "overid", "innings_ball_num", "batting_team_id", "bowling_team_id",
            "striker_id", "striker_name", "non_striker_id", "bowler_id", "bowler_name", "batting_type",
            "bowling_type", "runs", "tot", "conc", "legal", "is_wide", "is_nb", "is_dot", "is_bdry", "is_wicket",
            "dismissal_type", "dismissed_player_id", "extras_type", "delivery_style", "stroke", "on_or_off",
            "balls_left", "wk_before", "cum_before", "runs_left", "rrr", "phase", "mid_window", "wk_last12",
            "dots_last6", "consec_wides", "free_hit", "new_batter", "bat_balls", "prev_over_conc",
            "pace_or_spin", "wk_last18", "pship_runs", "inn_balls", "over_or_around", "venue_id", "venue_par", "is_ipl", "is_t20i", "impact_era", "season", "knock_out", "stage",
            "away", "striker_cap", "bowler_cap", "decided", "won", "revised_target",
            "wp", "wp_next", "dwp", "LI", "pressure", "outcome", "exp_runs", "exp_wkt"] + ["p_" + k for k in OUT_CLASSES]
    b[keep].to_pickle(OUT + "balls_v2.pkl")
    print("\nsaved", OUT + "balls_v2.pkl")


if __name__ == "__main__":
    main()
