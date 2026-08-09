"""Evaluate blue-escape algorithms and baselines across scenarios/seeds."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import yaml
from air_combat_rl.evaluation.evaluator import EvaluationError,run_evaluation

def main(argv=None):
    p=argparse.ArgumentParser();source=p.add_mutually_exclusive_group(required=True);source.add_argument('--scenarios',nargs='+');source.add_argument('--evaluation-suite')
    p.add_argument('--actions',required=True);p.add_argument('--algorithm',required=True);p.add_argument('--checkpoint');p.add_argument('--platform',choices=['zdj','yjj'],default='zdj');p.add_argument('--episodes',type=int);p.add_argument('--seeds',nargs='+',type=int);p.add_argument('--deterministic',action='store_true');p.add_argument('--output-dir',required=True);p.add_argument('--constant-action-id',type=int,default=0);p.add_argument('--max-policy-steps',type=int);p.add_argument('--device',default='auto');p.add_argument('--num-envs',type=int,default=8);p.add_argument('--env-backend',choices=['serial','subprocess'],default='subprocess');p.add_argument('--worker-start-method',choices=['spawn','forkserver'],default='spawn')
    a=p.parse_args(argv)
    if a.evaluation_suite:
        suite=yaml.safe_load(Path(a.evaluation_suite).read_text(encoding='utf-8')) or {};scenarios=suite.get('scenarios',[]);episodes=a.episodes or int(suite.get('episodes_per_seed',1));seeds=a.seeds or [int(x) for x in suite.get('seeds',[])];deterministic=a.deterministic or bool(suite.get('deterministic',False))
    else:scenarios=a.scenarios;episodes=a.episodes;seeds=a.seeds;deterministic=a.deterministic
    if episodes is None or not seeds:p.error('--episodes and --seeds are required unless supplied by --evaluation-suite')
    try:res=run_evaluation(scenarios=scenarios,actions=a.actions,algorithm_config_path=a.algorithm,checkpoint=a.checkpoint,platform=a.platform,episodes=episodes,seeds=seeds,deterministic=deterministic,output_dir=a.output_dir,constant_action_id=a.constant_action_id,max_policy_steps=a.max_policy_steps,device=a.device,num_envs=a.num_envs,env_backend=a.env_backend,start_method=a.worker_start_method)
    except EvaluationError as e:raise SystemExit(str(e))
    print(json.dumps({'output_dir':res['output_dir'],'episodes':res['episodes']},sort_keys=True))
if __name__=='__main__':main()
