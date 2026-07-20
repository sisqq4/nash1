"""Train one blue escape algorithm runtime."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
from air_combat_rl.runtime import build_blue_escape_env, build_algorithm_runtime
from air_combat_rl.training.runner import run_training

def _load_algorithm_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "algorithm" not in config and "name" in config:
        config = {"algorithm": {"name": config["name"]}, **{k: v for k, v in config.items() if k != "name"}}
    return config

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scenario',required=True); ap.add_argument('--actions',required=True); ap.add_argument('--algorithm',required=True); ap.add_argument('--platform',choices=['zdj','yjj'],default='zdj'); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--output-dir',required=True); ap.add_argument('--total-steps',type=int,default=64); ap.add_argument('--checkpoint-interval',type=int,default=64); ap.add_argument('--max-policy-steps',type=int,default=None)
    args=ap.parse_args()
    cfg=_load_algorithm_config(Path(args.algorithm)); cfg['seed']=args.seed
    env,rt_cfg=build_blue_escape_env(args.scenario,args.actions,args.platform,args.seed,args.max_policy_steps)
    runtime=build_algorithm_runtime(cfg,env)
    result=run_training(runtime,output_dir=args.output_dir,algorithm_config=cfg,seed=args.seed,total_steps=args.total_steps,checkpoint_interval=args.checkpoint_interval)
    print(json.dumps(result,sort_keys=True))
if __name__=='__main__': main()
