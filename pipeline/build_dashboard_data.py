"""
SupeRR Selector S2 - Round 1 - dashboard data (step 3 of 3).
Reads the ball table and player tables written by pressure_v2.py and rating_v2.py,
applies the selector's documented overrides, and writes site/data/dashboard.json
plus outputs/submitted_top10.csv.
Run from the repo root: python pipeline/build_dashboard_data.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import rating_v2 as r  # noqa: E402

D = os.environ.get("SUPERR_DATA", "data/raw/")
OUT = os.environ.get("SUPERR_OUT", "outputs/")
SITE = os.environ.get("SUPERR_SITE", "site/data/")

# Selector's calls on top of the model. Each is shown on the dashboard and in the report.
# Buttler (model rank 11, rating 0.80) replaces Priyansh Arya (model rank 9, 0.85):
# the gap is inside the uncertainty of batting ranks 6-10, and Buttler's sample is
# about four times larger (671 vs 174 pressure balls).
OVERRIDES = {
    "bat": [{"out": 25669, "in": 813, "slot": 9,
             "why": "Selector's call. The model has him 11th, 0.05 behind 9th - inside the "
                    "uncertainty for ranks 6-10 - on a sample four times larger than the player he replaces."}],
    "bowl": [],
}

PHASES = ["PP", "7-10", "11-15", "Death"]
PHASE_LABEL = {"PP": "Powerplay (1-6)", "7-10": "Overs 7-10", "11-15": "Overs 11-15", "Death": "Death (16-20)"}
COMP_LABEL = {"runs": "Runs above expected", "wkt": "Wicket preservation", "dots": "Dot-ball avoidance",
              "collapse": "Rebuilding after a collapse", "rv": "Run value (runs + wickets)",
              "disc": "Discipline (wides, no-balls)", "delta": "Rise vs own usual level",
              "consist": "Consistency across matches"}
BOWL_COMP_LABEL = dict(COMP_LABEL, collapse="Overs 7-10 control")


def f(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), nd)


def raw_stats(b, role):
    if role == "bat":
        d = b[b.faced == 1]
        g = d.groupby(["striker_id", "pressure"]).agg(balls=("legal", "sum"), runs=("runs", "sum"),
                                                     outs=("s_out", "sum"), dots=("bat_dot", "sum"),
                                                     bdry=("is_bdry", "sum"), n=("runs", "size"))
        g["sr"] = g.runs / g.balls * 100
        g["avg"] = g.runs / g.outs.replace(0, np.nan)
        g["dot_pct"] = g.dots / g.n * 100
        g["bdry_pct"] = g.bdry / g.n * 100
        keep = ["balls", "runs", "outs", "sr", "avg", "dot_pct", "bdry_pct"]
    else:
        d = b
        g = d.groupby(["bowler_id", "pressure"]).agg(balls=("legal", "sum"), conc=("conc", "sum"),
                                                    wkts=("bw", "sum"), dots=("is_dot", "sum"),
                                                    ext=("is_wide", "sum"), nb=("is_nb", "sum"), n=("conc", "size"))
        g["econ"] = g.conc / g.balls * 6
        g["bpw"] = g.balls / g.wkts.replace(0, np.nan)
        g["dot_pct"] = g.dots / g.n * 100
        g["extras_pct"] = (g.ext + g.nb) / g.n * 100
        keep = ["balls", "conc", "wkts", "econ", "bpw", "dot_pct", "extras_pct"]
    out = {}
    for (pid, p), row in g[keep].iterrows():
        out.setdefault(int(pid), {})["pressure" if p == 1 else "ordinary"] = {k: f(row[k], 2) for k in keep}
    return out


def players(role, b):
    e = pd.read_csv(OUT + ("batters_pressure_v2.csv" if role == "bat" else "bowlers_pressure_v2.csv"))
    idc = "striker_id" if role == "bat" else "bowler_id"
    e = e.set_index(idc)
    zc = [c for c in e.columns if c.startswith("z_")]
    rs = raw_stats(b, role)
    lab = COMP_LABEL if role == "bat" else BOWL_COMP_LABEL
    rows = []
    for pid, x in e.iterrows():
        rows.append({
            "id": int(pid), "name": x["name"], "model_rank": int(x["rank"]), "rating": f(x.rating),
            "p_balls": int(x.p_balls), "p_matches": int(x.p_matches), "n_venues": int(x.n_venues),
            "usual": f(x.rv_usual, 4), "pressure_level": f(x.rv, 4), "pressure_raw": f(x.rv_raw, 4),
            "delta": f(x.delta, 4), "consistency": f(x.consist_raw, 3),
            "p_top10": f(x.p_top10, 3), "rank_lo": f(x.rank_p05, 1), "rank_hi": f(x.rank_p95, 1),
            "z": {c[2:]: f(x[c], 2) for c in zc},
            "stats": rs.get(int(pid), {}),
        })
    comps = [{"key": c[2:], "label": lab.get(c[2:], c[2:])} for c in zc]
    return rows, comps


def submitted(rows, role):
    order = sorted(rows, key=lambda p: p["model_rank"])
    top = [p["id"] for p in order[:10]]
    notes = {}
    for o in OVERRIDES[role]:
        top.remove(o["out"])
        top.insert(o["slot"] - 1, o["in"])
        notes[o["in"]] = o["why"]
    return [{"rank": i + 1, "id": pid, "note": notes.get(pid)} for i, pid in enumerate(top)]


def hero_match(b):
    """Tightest knockout chase: highest mean leverage over the last 30 balls of innings 2."""
    m = pd.read_csv(D + "matches.csv")
    c = pd.read_csv(D + "competitions.csv")
    t = pd.read_csv(D + "teams.csv")
    ko = b[(b.knock_out == 1) & b.decided]
    i2 = ko[ko.innings == 2]
    score = i2.groupby("match_id").apply(lambda g: g.LI.tail(30).mean())
    mid = int(score.idxmax())
    g = b[b.match_id == mid].sort_values(["innings", "innings_ball_num"])
    mm = m[m.match_id == mid].iloc[0]
    comp = c[c.comp_id == mm.comp_id].iloc[0]
    names = t[t.comp_id == mm.comp_id].set_index("team_id").team_name
    chase_team = int(g[g.innings == 2].batting_team_id.iloc[0])
    first_team = int(g[g.innings == 1].batting_team_id.iloc[0])
    wp_chase = np.where(g.innings == 2, g.wp, 1 - g.wp)
    pts = [{"i": k, "inn": int(r_.innings), "over": int(r_.overid), "wp": f(w, 3), "p": int(r_.pressure),
            "w": int(r_.is_wicket), "r": int(r_.tot)} for k, (w, (_, r_)) in enumerate(zip(wp_chase, g.iterrows()))]
    t1 = int(g[g.innings == 1].tot.sum()); w1 = int(g[g.innings == 1].is_wicket.sum())
    t2 = int(g[g.innings == 2].tot.sum()); w2 = int(g[g.innings == 2].is_wicket.sum())
    return {"match_id": mid, "competition": comp.comp_name, "stage": mm.stage, "date": mm.match_date,
            "first": names.get(first_team, str(first_team)), "chase": names.get(chase_team, str(chase_team)),
            "first_score": f"{t1}/{w1}", "chase_score": f"{t2}/{w2}", "result": mm.result,
            "chase_won": int(mm.win_team_id == chase_team), "points": pts}


def main():
    raw = pd.read_pickle(OUT + "balls_v2.pkl")
    b, c_w = r.prep(raw)
    bat, bat_c = players("bat", b)
    bowl, bowl_c = players("bowl", b)

    grid = (b.pivot_table(index="innings", columns="phase", values="pressure", aggfunc="mean", observed=True)
              .reindex(columns=PHASES))
    b["LI_dec"] = pd.qcut(b.LI, 10, labels=False)
    dec = b.groupby("LI_dec").agg(LI=("LI", "mean"), swing=("dwp", lambda s: s.abs().mean()))
    grp = b.groupby("pressure").agg(rr=("tot", lambda s: s.mean() * 6), wk=("is_wicket", lambda s: s.mean() * 100),
                                    dotp=("is_dot", "mean"), LI=("LI", "mean"), n=("tot", "size"))
    weights = {}
    for role in ("bat", "bowl"):
        Ks, rho = r.tune(b, role)
        weights[role] = {k: f(v, 3) for k, v in r.make_weights(rho, role).items()}
    sens = {k: pd.read_csv(OUT + f"sensitivity_{k}.csv").to_dict("records") for k in ("bat", "bowl")}

    data = {
        "meta": {"matches": int(b.match_id.nunique()), "balls": int(len(b)),
                 "pressure_balls": int(b.pressure.sum()), "eligible_bat": len(bat), "eligible_bowl": len(bowl),
                 "wicket_runs": f(c_w, 1), "seasons": "2022-2026",
                 "split_half": {"bat": 0.25, "bowl": 0.45},
                 "eligibility": {"bat": list(r.ELIG["bat"]), "bowl": list(r.ELIG["bowl"])}},
        "pressure_vs_ordinary": {("pressure" if k else "ordinary"): {"run_rate": f(v.rr, 2), "wkts_per_100": f(v.wk, 2),
                                                                     "dot_pct": f(v.dotp * 100, 1), "mean_li": f(v.LI, 2),
                                                                     "balls": int(v.n)} for k, v in grp.iterrows()},
        "grid": {"phases": [PHASE_LABEL[p] for p in PHASES],
                 "innings": [{"innings": int(i), "share": [f(v * 100, 1) for v in row]} for i, row in grid.iterrows()]},
        "deciles": [{"decile": int(k) + 1, "li": f(v.LI, 2), "swing": f(v.swing, 4)} for k, v in dec.iterrows()],
        "hero": hero_match(b),
        "bat": {"players": bat, "components": bat_c, "submitted": submitted(bat, "bat")},
        "bowl": {"players": bowl, "components": bowl_c, "submitted": submitted(bowl, "bowl")},
        "sensitivity": sens,
        "weights": weights,
    }
    os.makedirs(SITE, exist_ok=True)
    with open(SITE + "dashboard.json", "w") as fh:
        json.dump(data, fh, separators=(",", ":"))
    rows = []
    for role in ("bat", "bowl"):
        byid = {p["id"]: p for p in data[role]["players"]}
        for s in data[role]["submitted"]:
            p = byid[s["id"]]
            rows.append({"list": "Batters" if role == "bat" else "Bowlers", "rank": s["rank"], "player_id": s["id"],
                         "name": p["name"], "model_rank": p["model_rank"], "rating": p["rating"],
                         "selector_note": s["note"] or ""})
    pd.DataFrame(rows).to_csv(OUT + "submitted_top10.csv", index=False)
    print("wrote", SITE + "dashboard.json", "and", OUT + "submitted_top10.csv")
    print("hero:", data["hero"]["competition"], data["hero"]["stage"], data["hero"]["first"], data["hero"]["first_score"],
          "v", data["hero"]["chase"], data["hero"]["chase_score"], "|", data["hero"]["result"])


if __name__ == "__main__":
    main()
