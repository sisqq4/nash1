"""Spawn-based subprocess vector environment; CUDA remains in the parent."""

from __future__ import annotations

import multiprocessing as mp
import traceback

import numpy as np

from air_combat_rl.vector.env_factory import make_env
from air_combat_rl.vector.types import EnvSpec, VectorStepResult


def _worker(remote, spec: EnvSpec) -> None:
    try:
        env = make_env(spec)
        while True:
            command, payload = remote.recv()
            if command == "reset":
                remote.send((True, env.reset(payload)))
            elif command == "step":
                result = env.step(payload)
                remote.send(
                    (
                        True,
                        (
                            result.observation,
                            result.reward,
                            result.terminated,
                            result.truncated,
                            result.info,
                        ),
                    )
                )
            elif command == "reconfigure":
                spec = payload
                env = make_env(spec)
                remote.send((True, env.reset(spec.seed)))
            elif command == "close":
                remote.send((True, None))
                break
            else:
                raise ValueError(f"unknown worker command {command!r}")
    except (EOFError, KeyboardInterrupt):
        pass
    except Exception as exc:
        try:
            remote.send((False, (type(exc).__name__, str(exc), traceback.format_exc())))
        except Exception:
            pass
    finally:
        remote.close()


class SubprocessVectorEnv:
    """Run each CPU simulator in a spawned child process."""

    def __init__(self, specs: list[EnvSpec], start_method: str = "spawn") -> None:
        if not specs:
            raise ValueError("at least one environment is required")
        self.specs = list(specs)
        context = mp.get_context(start_method)
        self.remotes = []
        self.processes = []
        self.closed = False
        self._observations: list[np.ndarray | None] = [None] * len(specs)
        self._infos: list[dict] = [{} for _ in specs]
        for spec in specs:
            parent, child = context.Pipe()
            process = context.Process(target=_worker, args=(child, spec), daemon=True)
            process.start()
            child.close()
            self.remotes.append(parent)
            self.processes.append(process)

    @property
    def num_envs(self) -> int:
        return len(self.remotes)

    @staticmethod
    def _receive(remote):
        ok, payload = remote.recv()
        if not ok:
            kind, message, worker_traceback = payload
            raise RuntimeError(
                f"environment worker failed ({kind}): {message}\n{worker_traceback}"
            )
        return payload

    def reset(self) -> tuple[np.ndarray, list[dict]]:
        for remote, spec in zip(self.remotes, self.specs):
            remote.send(("reset", spec.seed))
        pairs = [self._receive(remote) for remote in self.remotes]
        self._observations = [pair[0] for pair in pairs]
        self._infos = [pair[1] for pair in pairs]
        return np.stack(self._observations).astype(np.float32), list(self._infos)

    def reset_at(self, index: int, spec: EnvSpec) -> tuple[np.ndarray, dict]:
        self.specs[index] = spec
        self.remotes[index].send(("reconfigure", spec))
        observation, info = self._receive(self.remotes[index])
        self._observations[index] = observation
        self._infos[index] = info
        return observation, info

    def step(self, actions, active_mask=None) -> VectorStepResult:
        if len(actions) != self.num_envs:
            raise ValueError("action batch size does not match num_envs")
        active = (
            np.ones(self.num_envs, dtype=bool)
            if active_mask is None
            else np.asarray(active_mask, dtype=bool)
        )
        if active.shape != (self.num_envs,):
            raise ValueError("active_mask must have shape [num_envs]")
        if any(observation is None for observation in self._observations):
            raise RuntimeError("vector environment must be reset before step")

        for index, (remote, action) in enumerate(zip(self.remotes, actions)):
            if active[index]:
                remote.send(("step", action))

        observations = list(self._observations)
        infos = list(self._infos)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        for index, remote in enumerate(self.remotes):
            if not active[index]:
                continue
            observation, reward, was_terminated, was_truncated, info = self._receive(
                remote
            )
            observations[index] = observation
            infos[index] = info
            rewards[index] = reward
            terminated[index] = was_terminated
            truncated[index] = was_truncated

        self._observations = observations
        self._infos = infos
        return VectorStepResult(
            np.stack(observations).astype(np.float32),
            rewards,
            terminated,
            truncated,
            infos,
        )

    def close(self) -> None:
        if self.closed:
            return
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for remote in self.remotes:
            try:
                self._receive(remote)
            except Exception:
                pass
            remote.close()
        for process in self.processes:
            process.join(timeout=5)
        for process in self.processes:
            if process.is_alive():
                process.terminate()
                process.join()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
