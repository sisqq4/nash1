"""Small discrete Q network used by the Rainbow compatibility trainer."""
from __future__ import annotations
import numpy as np
class RainbowQNetwork:
    def __init__(self, obs_dim: int, action_dim: int = 29, seed: int | None = None) -> None:
        self.obs_dim = obs_dim; self.action_dim = action_dim; self.rng = np.random.default_rng(seed)
        self.weights = self.rng.normal(0, 0.01, (obs_dim, action_dim)); self.bias = np.zeros(action_dim)
    def predict_q(self, observation):
        obs = np.asarray(observation, float)
        return obs @ self.weights + self.bias
    def copy_from(self, other: "RainbowQNetwork") -> None:
        self.weights = other.weights.copy(); self.bias = other.bias.copy()
    def state_dict(self): return {"weights": self.weights.copy(), "bias": self.bias.copy(), "obs_dim": self.obs_dim, "action_dim": self.action_dim}
    @classmethod
    def from_state_dict(cls, state):
        net = cls(int(state.get("obs_dim", np.asarray(state["weights"]).shape[0])), int(state.get("action_dim", np.asarray(state["weights"]).shape[1])))
        net.weights = np.asarray(state["weights"], float); net.bias = np.asarray(state["bias"], float); return net
