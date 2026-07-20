"""Unified training loop, JSONL logging, and atomic checkpoints."""
from __future__ import annotations
from dataclasses import asdict, is_dataclass
from pathlib import Path
import json, os, tempfile
import numpy as np

class CheckpointTypeError(ValueError): pass

def _jsonable(x):
    if is_dataclass(x): return _jsonable(asdict(x))
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, dict): return {str(k): _jsonable(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [_jsonable(v) for v in x]
    if hasattr(x, '__dict__'): return _jsonable({k:v for k,v in x.__dict__.items() if k!='rng'})
    return x

def save_checkpoint(path, runtime, *, algorithm_config, global_step, episode, seed):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    trainer=runtime.trainer; model=getattr(trainer,'actor_critic',getattr(trainer,'q_network',None))
    payload={"algorithm_name":runtime.name,"model_state": model.state_dict() if model is not None else {},"optimizer_state":{},"global_step":global_step,"episode":episode,"algorithm_config":algorithm_config,"observation_dim":getattr(model,'obs_dim',None),"action_dim":getattr(model,'action_dim',None),"action_catalog_version":getattr(getattr(runtime.env,'actions',getattr(runtime.env,'base_env',None).actions if hasattr(runtime.env,'base_env') else None),'version',None),"seed":seed,"trainer_state":{"replay_size":len(getattr(trainer,'replay_buffer',[])) if hasattr(trainer,'replay_buffer') else None}}
    if runtime.name=='ppo_projected':
        payload.update({"continuous_action_dim":3,"action_bounds":{"low":[-1,-1,-1],"high":[1,1,1]},"projection_metric":"weighted_normalized_command_distance","projection_weights":_jsonable(getattr(runtime.env.mapper,'config',{})),"log_std":_jsonable(getattr(model,'log_std',None))})
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=path.name+'.tmp.'); os.close(fd)
    with open(tmp,'w',encoding='utf-8') as f: json.dump(_jsonable(payload),f,indent=2,sort_keys=True)
    os.replace(tmp,path)

def assert_checkpoint_algorithm(path, expected):
    data=json.loads(Path(path).read_text())
    if data.get('algorithm_name')!=expected: raise CheckpointTypeError(f"checkpoint algorithm {data.get('algorithm_name')!r} != {expected!r}")
    return data

def run_training(runtime, *, output_dir, algorithm_config, seed, total_steps=64, checkpoint_interval=64):
    out=Path(output_dir); ck=out/'checkpoints'; ck.mkdir(parents=True,exist_ok=True)
    (out/'manifest.json').write_text(json.dumps({"algorithm_name":runtime.name,"seed":seed,"algorithm_config":algorithm_config},indent=2),encoding='utf-8')
    episode=0; obs_info=None
    with (out/'train_metrics.jsonl').open('a',encoding='utf-8') as mf, (out/'episodes.jsonl').open('a',encoding='utf-8') as ef:
        while getattr(runtime.trainer,'global_step',0) < total_steps:
            if runtime.name.startswith('ppo'):
                runtime.trainer.collect_rollout(seed if getattr(runtime.trainer,'global_step',0)==0 else None); stats=runtime.trainer.train_one_update(); metrics=stats if isinstance(stats,dict) else asdict(stats)
            else:
                if obs_info is None: obs_info=runtime.env.reset(seed); episode+=1; ep_reward=0.0; ep_len=0
                obs,info=obs_info
                for _ in range(min(32,total_steps-runtime.trainer.global_step)):
                    res=runtime.trainer.collect_step(obs); ep_reward+=res.reward; ep_len+=1; obs=res.observation
                    if res.terminated or res.truncated:
                        ef.write(json.dumps({"episode":episode,"reward":ep_reward,"length":ep_len,"outcome":res.info.get('outcome')})+'\n'); obs,info=runtime.env.reset(); episode+=1; ep_reward=0.0; ep_len=0
                obs_info=(obs,info); stats=runtime.trainer.train_one_update(); metrics=asdict(stats)
            metrics["algorithm_name"]=runtime.name; mf.write(json.dumps(_jsonable(metrics),sort_keys=True)+'\n'); mf.flush()
            step=int(metrics.get('global_step',getattr(runtime.trainer,'global_step',0)))
            if step and step % checkpoint_interval == 0: save_checkpoint(ck/f'step_{step}.pt',runtime,algorithm_config=algorithm_config,global_step=step,episode=episode,seed=seed)
    step=getattr(runtime.trainer,'global_step',0); save_checkpoint(ck/'latest.pt',runtime,algorithm_config=algorithm_config,global_step=step,episode=episode,seed=seed)
    return {"global_step":step,"output_dir":str(out)}


def build_algorithm_runtime(config: dict, env):
    """Backward-compatible import location for algorithm runtime construction."""
    from air_combat_rl.runtime import build_algorithm_runtime as _build_algorithm_runtime
    return _build_algorithm_runtime(config, env)
