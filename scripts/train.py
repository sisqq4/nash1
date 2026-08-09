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
    source=ap.add_mutually_exclusive_group(required=True); source.add_argument('--scenario'); source.add_argument('--curriculum')
    ap.add_argument('--actions',required=True); ap.add_argument('--algorithm',required=True); ap.add_argument('--platform',choices=['zdj','yjj'],default='zdj'); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--output-dir',required=True); ap.add_argument('--total-steps',type=int,default=64); ap.add_argument('--checkpoint-interval',type=int,default=64); ap.add_argument('--max-policy-steps',type=int,default=None); ap.add_argument('--resume',help='resume from a type-compatible unified checkpoint')
    ap.add_argument('--device',default='auto'); ap.add_argument('--num-envs',type=int); ap.add_argument('--env-backend',choices=['serial','subprocess']); ap.add_argument('--worker-start-method',choices=['spawn','forkserver'],default='spawn'); ap.add_argument('--amp',action=argparse.BooleanOptionalAction,default=None)
    args=ap.parse_args()
    cfg=_load_algorithm_config(Path(args.algorithm)); cfg['seed']=args.seed
    if cfg.get('algorithm',{}).get('backend') == 'torch':
        from air_combat_rl.training.torch_runner import run_torch_training
        result=run_torch_training(algorithm_config=cfg,actions=args.actions,platform=args.platform,seed=args.seed,output_dir=args.output_dir,total_steps=args.total_steps,checkpoint_interval=args.checkpoint_interval,device=args.device,num_envs=args.num_envs,env_backend=args.env_backend,start_method=args.worker_start_method,scenario=args.scenario,curriculum=args.curriculum,max_policy_steps=args.max_policy_steps,resume=args.resume,amp=args.amp)
        print(json.dumps(result,sort_keys=True)); return
    if args.curriculum: raise SystemExit('--curriculum requires an algorithm config with algorithm.backend: torch')
    env,rt_cfg=build_blue_escape_env(args.scenario,args.actions,args.platform,args.seed,args.max_policy_steps)
    runtime=build_algorithm_runtime(cfg,env)
    result=run_training(runtime,output_dir=args.output_dir,algorithm_config=cfg,seed=args.seed,total_steps=args.total_steps,checkpoint_interval=args.checkpoint_interval,resume=args.resume)
    print(json.dumps(result,sort_keys=True))
if __name__=='__main__': main()
