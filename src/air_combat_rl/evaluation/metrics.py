"""Unified episode metrics for blue-escape batch evaluation."""
from __future__ import annotations
from collections import Counter, defaultdict
import math
import numpy as np

OUTCOMES=("hit","crash","success","exhausted","timeout","running")
GROUP_KEYS=("scenario","platform","missile_count","seed","algorithm")

def _mean_std(xs):
    xs=[float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not xs: return {"mean":None,"std":None}
    return {"mean":float(np.mean(xs)),"std":float(np.std(xs, ddof=0))}

def _rates(rows):
    n=len(rows); c=Counter(str(r.get("outcome","running")) for r in rows)
    out={"episode_count":n,"outcome_counts":{k:int(c.get(k,0)) for k in OUTCOMES}}
    def rate(num): return 0.0 if n==0 else float(num)/n
    out.update({
        "hit_rate":rate(c["hit"]),"crash_rate":rate(c["crash"]),"success_rate":rate(c["success"]),
        "exhausted_rate":rate(c["exhausted"]),"timeout_rate":rate(c["timeout"]),
        "survival_rate":rate(n-c["hit"]-c["crash"]),
        "escape_completion_rate":rate(c["success"]+c["exhausted"]),
    })
    return out

def summarize_episodes(rows):
    """Return metrics for all rows and grouped by scenario/platform/missile_count/seed/algorithm."""
    rows=list(rows)
    def block(rs):
        b=_rates(rs)
        b.update({
            "reward":_mean_std([r.get("total_reward") for r in rs]),
            "duration_s":_mean_std([r.get("duration_s") for r in rs]),
            "policy_steps":_mean_std([r.get("policy_steps") for r in rs]),
            "min_sampled_distance_m":_mean_std([r.get("min_sampled_distance_m") for r in rs]),
            "min_altitude_y_m":_mean_std([r.get("min_altitude_y_m") for r in rs]),
            "lowest_altitude_y_m": min([float(r.get("min_altitude_y_m")) for r in rs if r.get("min_altitude_y_m") is not None], default=None),
            "ground_collision_count": int(sum(1 for r in rs if r.get("ground_collision_count",0))),
            "missile_exhaustion_count": int(sum(1 for r in rs if r.get("missile_exhaustion_count",0))),
        })
        proj=[r.get("projected_ppo",{}) or {} for r in rs]
        distances=[d for p in proj for d in p.get("projection_distances",[])]
        exact=sum(1 for d in distances if abs(float(d)) <= 1e-9)
        cont=[v for p in proj for v in p.get("continuous_actions",[])]
        cont_arr=np.asarray(cont,float) if cont else np.empty((0,3))
        dist_counts=Counter(); map_counts=Counter(); legal=[]; sat=[]
        for p in proj:
            dist_counts.update({str(k):int(v) for k,v in (p.get("projected_action_distribution",{}) or {}).items()})
            map_counts.update({str(k):int(v) for k,v in (p.get("continuous_to_discrete_mapping_frequency",{}) or {}).items()})
            legal += p.get("valid_action_counts",[]) or []
            sat += p.get("saturation_flags",[]) or []
        b["projected_ppo"]={
            "projection_distance": {**_mean_std(distances), "max": (max(distances) if distances else None)},
            "exact_projection_rate": (0.0 if not distances else exact/len(distances)),
            "projected_action_distribution": dict(dist_counts),
            "continuous_action_mean": (cont_arr.mean(axis=0).tolist() if len(cont_arr) else None),
            "continuous_action_std": (cont_arr.std(axis=0).tolist() if len(cont_arr) else None),
            "saturation_rate": (0.0 if not sat else sum(1 for x in sat if x)/len(sat)),
            "valid_action_count": _mean_std(legal),
            "continuous_to_discrete_mapping_frequency": dict(map_counts),
        }
        return b
    grouped=defaultdict(list)
    for r in rows:
        grouped[tuple(r.get(k) for k in GROUP_KEYS)].append(r)
    return {"overall":block(rows),"groups":[{"keys":dict(zip(GROUP_KEYS,k)),"metrics":block(v)} for k,v in sorted(grouped.items(), key=lambda kv: str(kv[0]))]}
