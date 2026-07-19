from __future__ import annotations
import pickle, random
METADATA={"algorithm_name":"ppo_projected","action_interface":"projected_continuous_to_discrete","action_dimension":3,"coordinate_convention":"XZY","environment_version":"blue_escape/M5"}
def save_ppo_checkpoint(path, actor_critic, optimizer_state=None, obs_normalization=None, global_step=0, curriculum_stage=None, config=None):
    payload={"metadata":METADATA.copy(),"actor_critic":actor_critic.state_dict() if hasattr(actor_critic,"state_dict") else actor_critic,"optimizer":optimizer_state,"observation_normalization":obs_normalization,"global_step":global_step,"current_curriculum_stage":curriculum_stage,"random_state":random.getstate(),"config":config}
    with open(path,"wb") as f: pickle.dump(payload,f)
def load_ppo_checkpoint(path):
    with open(path,"rb") as f: payload=pickle.load(f)
    meta=payload.get("metadata",{})
    if meta.get("algorithm_name") != "ppo_projected": raise ValueError("checkpoint is not a ppo_projected checkpoint")
    if meta.get("action_dimension") != 3: raise ValueError("PPO checkpoint action_dimension must be 3")
    return payload
