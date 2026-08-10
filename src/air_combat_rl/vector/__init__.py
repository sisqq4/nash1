"""Vectorized CPU simulation backends."""

from src.air_combat_rl.vector.types import EnvSpec, VectorStepResult
from src.air_combat_rl.vector.serial_vector_env import SerialVectorEnv
from src.air_combat_rl.vector.subprocess_vector_env import SubprocessVectorEnv

__all__ = ["EnvSpec", "VectorStepResult", "SerialVectorEnv", "SubprocessVectorEnv"]
