r"""Summarise scripts/reliability_study.py outputs (one JSON per recording) into condition-effect tables.

Reference noise = between-recording repeatability of the UNDEGRADED fits of the same subject (current defaults):
Sw = SD of girth/height across recordings, RC = 2.77 x Sw (95% of repeat differences of an unchanged body).
A condition whose girth change exceeds RC is a real, measurable degradation; below Sw it is indistinguishable
from re-recording the person.

Usage: .venv\\Scripts\\python.exe scripts/summarize_reliability.py outputs/accuracy_study/reliability/*.json \
           --markdown outputs/accuracy_study/reliability/summary.md
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

SITES = ["chest", "waist", "abdomen", "hip", "upper_arm_left", "upper_arm_right", "forearm_left", "forearm_right",
         "thigh_left", "thigh_right", "calf_left", "calf_right"]
GROUPS = {"chest": ["chest"], "waist": ["waist"], "abdomen": ["abdomen"], "hip": ["hip"],
          "upper arm": ["upper_arm_left", "upper_arm_right"], "forearm": ["forearm_left", "forearm_right"],
          "thigh": ["thigh_left", "thigh_right"], "calf": ["calf_left", "calf_right"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--height-cm", type=float, default=176.0)
    ap.add_argument("--markdown", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    runs = {}
    for f in a.files:
        runs.update(json.loads(f.read_text(encoding="utf-8"))["runs"])
    H = a.height_cm
    base = {r: d["conditions"]["baseline#0"]["girths_x_height"] for r, d in runs.items() if "baseline#0" in d["conditions"]}
    # Between-recording repeatability under current defaults.
    rep = {}
    for g, sites in GROUPS.items():
        vals = np.array([[base[r][s] for s in sites] for r in base])  # recordings x sides
        mean = vals.mean()
        sw = float(np.mean(vals.std(axis=0, ddof=1))) if len(base) > 1 else float("nan")
        rep[g] = {"mean_x_height": float(mean), "mean_cm": float(mean * H), "sw_pct": 100 * sw / mean,
                  "rc_pct": 277 * sw / mean, "rc_cm": 2.77 * sw * H, "recordings": len(base)}
    # Condition effects: % change from the same recording's baseline, pooled over recordings and seeds.
    conds = sorted({k.split("#")[0] for d in runs.values() for k in d["conditions"]} - {"baseline"},
                   key=lambda c: list(next(iter(runs.values()))["conditions"]).index(next(
                       k for k in next(iter(runs.values()))["conditions"] if k.startswith(c + "#"))))
    effects = {}
    for c in conds:
        e = {"groups": {}, "protrusion": [], "excluded": [], "views": [], "max_beta": []}
        for r, d in runs.items():
            for k, row in d["conditions"].items():
                if k.split("#")[0] != c or r not in base:
                    continue
                e["protrusion"].append(row["silhouette_vs_degraded_masks"]["protrusion_frac"])
                e["excluded"].append(row["views_excluded_after_fit"])
                e["views"].append(row["views_used"])
                e["max_beta"].append(row["max_abs_beta_sd"])
                for g, sites in GROUPS.items():
                    for s in sites:
                        if row["girths_x_height"].get(s) and base[r].get(s):
                            e["groups"].setdefault(g, []).append(
                                100 * (row["girths_x_height"][s] - base[r][s]) / base[r][s])
        out = {"n_fits": len(e["protrusion"]), "views_used": int(np.median(e["views"])),
               "views_excluded_mean": float(np.mean(e["excluded"])), "protrusion_mean": float(np.mean(e["protrusion"])),
               "max_beta_sd": float(np.max(e["max_beta"])), "groups": {}}
        for g, d in e["groups"].items():
            d = np.array(d)
            out["groups"][g] = {"mean_pct": float(d.mean()), "mean_abs_pct": float(np.abs(d).mean()),
                                "max_abs_pct": float(np.abs(d).max()), "max_abs_cm": float(np.abs(d).max() / 100 *
                                                                                          rep[g]["mean_cm"])}
            out["groups"][g]["vs_rc"] = ("exceeds RC" if out["groups"][g]["max_abs_pct"] > rep[g]["rc_pct"] else
                                         "within RC" if out["groups"][g]["max_abs_pct"] > rep[g]["sw_pct"] else
                                         "below Sw")
        effects[c] = out
    # Jackknife (3 of 4 heading folds): spread of each girth across folds vs between-recording Sw.
    jk = {}
    for g, sites in GROUPS.items():
        sds = []
        for r, d in runs.items():
            if d.get("jackknife"):
                for s in sites:
                    v = np.array([f[s] for f in d["jackknife"] if f.get(s)])
                    if len(v) > 1:
                        sds.append(100 * v.std(ddof=1) / v.mean())
        if sds:
            jk[g] = {"fold_sd_pct_mean": float(np.mean(sds)), "fold_sd_pct_max": float(np.max(sds)),
                     "ratio_to_between_recording_sw": float(np.mean(sds) / rep[g]["sw_pct"]) if rep[g]["sw_pct"] else None}
    res = {"height_cm": H, "repeatability_current_defaults": rep, "effects": effects, "jackknife": jk,
           "baselines": {r: {g: base[r][GROUPS[g][0]] for g in GROUPS} for r in base}}
    md = ["# Reliability study summary", "",
          f"Recordings: {', '.join(base)} · girths in cm at {H:g} cm · changes are % of the same recording's "
          "undegraded fit, pooled over recordings and random seeds.", "",
          "## Between-recording repeatability (current defaults, undegraded)", "",
          "| Site | Mean cm | Sw % | RC % | RC cm |", "|---|---|---|---|---|"]
    for g, v in rep.items():
        md.append(f"| {g} | {v['mean_cm']:.1f} | {v['sw_pct']:.2f} | {v['rc_pct']:.2f} | {v['rc_cm']:.1f} |")
    md += ["", "## Condition effects: max |change| % (mean signed change %)", "",
           "| Condition | views | excl. | protr. | " + " | ".join(GROUPS) + " |",
           "|---|---|---|---|" + "---|" * len(GROUPS)]
    for c, o in effects.items():
        cells = []
        for g in GROUPS:
            x = o["groups"].get(g)
            if not x:
                cells.append("—")
                continue
            mark = {"exceeds RC": " **▲**", "within RC": "", "below Sw": ""}[x["vs_rc"]]
            cells.append(f"{x['max_abs_pct']:.1f} ({x['mean_pct']:+.1f}){mark}")
        md.append(f"| {c} | {o['views_used']} | {o['views_excluded_mean']:.1f} | {o['protrusion_mean']:.3f} | "
                  + " | ".join(cells) + " |")
    md += ["", "**▲** = the worst change exceeds the between-recording repeatability coefficient (RC).", "",
           "## Jackknife over heading folds (3 of 4 folds each)", "",
           "| Site | fold SD % (mean) | fold SD % (max) | ÷ between-recording Sw |", "|---|---|---|---|"]
    for g, v in jk.items():
        md.append(f"| {g} | {v['fold_sd_pct_mean']:.2f} | {v['fold_sd_pct_max']:.2f} | "
                  f"{v['ratio_to_between_recording_sw']:.2f} |")
    text = "\n".join(md) + "\n"
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)
    if a.markdown:
        a.markdown.write_text(text, encoding="utf-8")
    if a.json:
        a.json.write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
