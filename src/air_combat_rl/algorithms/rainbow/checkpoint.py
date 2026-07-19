from __future__ import annotations
import pickle
from air_combat_rl.algorithms.rainbow.network import RainbowQNetwork
class RainbowCheckpointError(ValueError): pass
def save_rainbow_checkpoint(path, trainer, config=None):
    payload={"metadata":{"algorithm_name":"rainbow_dqn","action_interface":"discrete_29"},"q_network":trainer.q_network.state_dict(),"target_network":trainer.target_network.state_dict(),"global_step":trainer.global_step,"config":config}
    with open(path,"wb") as f: pickle.dump(payload,f)
def load_rainbow_checkpoint(path):
    with open(path,"rb") as f: payload=pickle.load(f)
    meta=payload.get("metadata",{}) if isinstance(payload,dict) else {}; algo=meta.get("algorithm_name", payload.get("algorithm_name") if isinstance(payload,dict) else None)
    if algo not in (None,"rainbow_dqn","RainbowDQN"): raise RainbowCheckpointError(f"not a Rainbow DQN checkpoint: {algo}")
    return payload
def load_rainbow_networks(path):
    payload=load_rainbow_checkpoint(path)
    if "q_network" not in payload: return payload, None, None
    q=RainbowQNetwork.from_state_dict(payload["q_network"]); target=RainbowQNetwork.from_state_dict(payload.get("target_network", payload["q_network"]))
    return payload, q, target
