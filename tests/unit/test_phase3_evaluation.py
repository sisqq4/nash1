import csv, json, subprocess, sys
from pathlib import Path
import pytest

from air_combat_rl.evaluation.metrics import summarize_episodes
from air_combat_rl.evaluation.evaluator import EvaluationError, run_evaluation
from air_combat_rl.runtime import build_blue_escape_env, build_algorithm_runtime
from air_combat_rl.training.runner import save_checkpoint


def rows():
    return [
        {"scenario":"a","platform":"zdj","missile_count":1,"seed":0,"algorithm":"x","outcome":"hit","total_reward":-1,"duration_s":1,"policy_steps":1,"min_sampled_distance_m":10,"min_altitude_y_m":100,"ground_collision_count":0,"missile_exhaustion_count":0},
        {"scenario":"a","platform":"zdj","missile_count":1,"seed":0,"algorithm":"x","outcome":"crash","total_reward":-2,"duration_s":2,"policy_steps":2,"min_sampled_distance_m":20,"min_altitude_y_m":0,"ground_collision_count":1,"missile_exhaustion_count":0},
        {"scenario":"b","platform":"zdj","missile_count":1,"seed":1,"algorithm":"x","outcome":"success","total_reward":3,"duration_s":3,"policy_steps":3,"min_sampled_distance_m":30,"min_altitude_y_m":200,"ground_collision_count":0,"missile_exhaustion_count":0},
        {"scenario":"b","platform":"zdj","missile_count":1,"seed":1,"algorithm":"x","outcome":"exhausted","total_reward":4,"duration_s":4,"policy_steps":4,"min_sampled_distance_m":40,"min_altitude_y_m":300,"ground_collision_count":0,"missile_exhaustion_count":1},
        {"scenario":"b","platform":"zdj","missile_count":1,"seed":2,"algorithm":"x","outcome":"timeout","total_reward":0,"duration_s":5,"policy_steps":5,"min_sampled_distance_m":50,"min_altitude_y_m":400,"ground_collision_count":0,"missile_exhaustion_count":0},
    ]

def test_outcome_survival_escape_timeout_and_grouped_metrics():
    m=summarize_episodes(rows())
    o=m['overall']
    assert o['episode_count']==5
    assert o['hit_rate']==pytest.approx(.2) and o['crash_rate']==pytest.approx(.2)
    assert o['survival_rate']==pytest.approx(.6)
    assert o['escape_completion_rate']==pytest.approx(.4)
    assert o['timeout_rate']==pytest.approx(.2)
    assert len(m['groups'])==3

def test_empty_data_behavior():
    m=summarize_episodes([])['overall']
    assert m['episode_count']==0 and m['survival_rate']==0.0
    assert m['reward']['mean'] is None

def test_csv_json_markdown_outputs_and_seed_repro(tmp_path):
    args=dict(scenarios=['configs/scenario/fixed_1v1.yaml'],actions='configs/actions/blue_29.yaml',algorithm_config_path='configs/algorithm/random_valid.yaml',checkpoint=None,platform='yjj',episodes=2,seeds=[3],deterministic=False,max_policy_steps=1)
    a=run_evaluation(**args, output_dir=tmp_path/'a')
    b=run_evaluation(**args, output_dir=tmp_path/'b')
    assert a['episodes']==2
    for name in ['manifest.json','episodes.csv','metrics.json','report.md','evaluation_steps.jsonl']:
        assert (tmp_path/'a'/name).exists()
    assert (tmp_path/'a'/'trajectories').is_dir()
    ea=list(csv.DictReader((tmp_path/'a'/'episodes.csv').open()))
    eb=list(csv.DictReader((tmp_path/'b'/'episodes.csv').open()))
    assert [r['outcome'] for r in ea] == [r['outcome'] for r in eb]
    assert 'timeout_rate' in (tmp_path/'a'/'metrics.json').read_text()
    assert 'Projected PPO投影诊断' in (tmp_path/'a'/'report.md').read_text()

def _checkpoint(tmp_path, alg):
    env,_=build_blue_escape_env('configs/scenario/fixed_1v1.yaml','configs/actions/blue_29.yaml','zdj',0,max_policy_steps=1)
    cfg={'algorithm':{'name':alg}, 'seed':0}
    rt=build_algorithm_runtime(cfg,env)
    p=tmp_path/f'{alg}.pt'
    save_checkpoint(p,rt,algorithm_config=cfg,global_step=0,episode=0,seed=0)
    return p

@pytest.mark.parametrize('alg,cfg', [('ppo_projected','configs/algorithm/ppo_projected.yaml'),('ppo_discrete','configs/algorithm/ppo_discrete.yaml'),('rainbow_dqn','configs/algorithm/rainbow_dqn.yaml')])
def test_three_algorithm_short_eval_and_checkpoint_compatibility(tmp_path, alg, cfg):
    ck=_checkpoint(tmp_path,alg)
    res=run_evaluation(scenarios=['configs/scenario/fixed_1v1.yaml'],actions='configs/actions/blue_29.yaml',algorithm_config_path=cfg,checkpoint=ck,platform='zdj',episodes=1,seeds=[0],deterministic=True,output_dir=tmp_path/f'eval_{alg}',max_policy_steps=1)
    assert res['episodes']==1
    manifest=json.loads((tmp_path/f'eval_{alg}'/'manifest.json').read_text())
    assert manifest['fair_comparison']['same_scenarios']

def test_checkpoint_algorithm_mismatch_errors(tmp_path):
    ck=_checkpoint(tmp_path,'rainbow_dqn')
    with pytest.raises(EvaluationError, match='checkpoint algorithm'):
        run_evaluation(scenarios=['configs/scenario/fixed_1v1.yaml'],actions='configs/actions/blue_29.yaml',algorithm_config_path='configs/algorithm/ppo_discrete.yaml',checkpoint=ck,platform='zdj',episodes=1,seeds=[0],deterministic=True,output_dir=tmp_path/'bad',max_policy_steps=1)

def test_episode_state_isolation_and_projected_deterministic_projection(tmp_path):
    ck=_checkpoint(tmp_path,'ppo_projected')
    run_evaluation(scenarios=['configs/scenario/fixed_1v1.yaml'],actions='configs/actions/blue_29.yaml',algorithm_config_path='configs/algorithm/ppo_projected.yaml',checkpoint=ck,platform='zdj',episodes=2,seeds=[0],deterministic=True,output_dir=tmp_path/'proj',max_policy_steps=1)
    steps=[json.loads(x) for x in (tmp_path/'proj'/'evaluation_steps.jsonl').read_text().splitlines()]
    assert len(steps)==2 and steps[0]['time_s']==steps[1]['time_s']
    assert steps[0]['action_id']==steps[1]['action_id']
    metrics=json.loads((tmp_path/'proj'/'metrics.json').read_text())
    diag=metrics['overall']['projected_ppo']
    assert diag['valid_action_count']['mean'] == 29.0


def test_projected_ppo_uses_training_wrapper_squashed_action_and_reloads_checkpoint_each_episode(tmp_path, monkeypatch):
    from air_combat_rl.evaluation import evaluator

    ck=_checkpoint(tmp_path,'ppo_projected')
    seen=[]
    original=evaluator._checkpoint_load

    def wrapped(runtime, path, alg, env):
        payload=original(runtime,path,alg,env)
        if alg == 'ppo_projected':
            runtime.trainer.actor_critic.mean_b[...] = [9.0, -9.0, 9.0]
            seen.append(runtime.trainer.actor_critic.mean_b.tolist())
        return payload

    monkeypatch.setattr(evaluator, '_checkpoint_load', wrapped)
    evaluator.run_evaluation(scenarios=['configs/scenario/fixed_1v1.yaml'],actions='configs/actions/blue_29.yaml',algorithm_config_path='configs/algorithm/ppo_projected.yaml',checkpoint=ck,platform='zdj',episodes=2,seeds=[0],deterministic=True,output_dir=tmp_path/'squashed',max_policy_steps=1)
    metrics=json.loads((tmp_path/'squashed'/'metrics.json').read_text())
    assert len(seen) == 2
    assert metrics['overall']['projected_ppo']['saturation_rate'] == 1.0

def test_error_conditions(tmp_path):
    with pytest.raises(EvaluationError, match='episodes'):
        run_evaluation(scenarios=['configs/scenario/fixed_1v1.yaml'],actions='configs/actions/blue_29.yaml',algorithm_config_path='configs/algorithm/random_valid.yaml',checkpoint=None,platform='zdj',episodes=0,seeds=[0],deterministic=True,output_dir=tmp_path/'x')
    with pytest.raises(EvaluationError, match='scenario list'):
        run_evaluation(scenarios=[],actions='configs/actions/blue_29.yaml',algorithm_config_path='configs/algorithm/random_valid.yaml',checkpoint=None,platform='zdj',episodes=1,seeds=[0],deterministic=True,output_dir=tmp_path/'x')
    out=tmp_path/'exists'; out.mkdir(); (out/'keep').write_text('x')
    with pytest.raises(EvaluationError, match='output directory conflict'):
        run_evaluation(scenarios=['configs/scenario/fixed_1v1.yaml'],actions='configs/actions/blue_29.yaml',algorithm_config_path='configs/algorithm/random_valid.yaml',checkpoint=None,platform='zdj',episodes=1,seeds=[0],deterministic=True,output_dir=out)
    with pytest.raises(EvaluationError, match='illegal action'):
        run_evaluation(scenarios=['configs/scenario/fixed_1v1.yaml'],actions='configs/actions/blue_29.yaml',algorithm_config_path='configs/algorithm/constant.yaml',checkpoint=None,platform='yjj',episodes=1,seeds=[0],deterministic=True,output_dir=tmp_path/'illegal',constant_action_id=2,max_policy_steps=1)

def test_cli_help():
    p=subprocess.run([sys.executable,'scripts/evaluate.py','--help'],env={'PYTHONPATH':'src'},capture_output=True,text=True)
    assert p.returncode==0 and '--scenarios' in p.stdout
