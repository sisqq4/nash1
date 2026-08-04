"""Evaluate blue-escape algorithms and baselines across scenarios/seeds."""
from __future__ import annotations
import argparse, json
from air_combat_rl.evaluation.evaluator import EvaluationError, run_evaluation

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--scenarios',nargs='+',required=True); p.add_argument('--actions',required=True); p.add_argument('--algorithm',required=True); p.add_argument('--checkpoint'); p.add_argument('--platform',choices=['zdj','yjj'],default='zdj'); p.add_argument('--episodes',type=int,required=True); p.add_argument('--seeds',nargs='+',type=int,required=True); p.add_argument('--deterministic',action='store_true'); p.add_argument('--output-dir',required=True); p.add_argument('--constant-action-id',type=int,default=0); p.add_argument('--max-policy-steps',type=int)
    a=p.parse_args(argv)
    try:
        res=run_evaluation(scenarios=a.scenarios,actions=a.actions,algorithm_config_path=a.algorithm,checkpoint=a.checkpoint,platform=a.platform,episodes=a.episodes,seeds=a.seeds,deterministic=a.deterministic,output_dir=a.output_dir,constant_action_id=a.constant_action_id,max_policy_steps=a.max_policy_steps)
    except EvaluationError as e:
        raise SystemExit(str(e))
    print(json.dumps({"output_dir":res['output_dir'],"episodes":res['episodes']},sort_keys=True))
if __name__=='__main__': main()
