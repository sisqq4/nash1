"""Rule-machine blue escape policy aligned with the M4 reward design."""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from air_combat_rl.core.coordinates import normalize_angle_rad
from air_combat_rl.core.math3d import flight_velocity
from air_combat_rl.simulation.snapshot import WorldSnapshot
from collections.abc import Sequence


A_FORWARD = 0
A_ACCEL = 1
A_DIVE = 14
A_LEFT_TURN = 5
A_RIGHT_TURN = 8
A_CLIMB = 11
A_LEFT_CLIMB = 17
A_RIGHT_CLIMB = 20
A_LEFT_DIVE = 23
A_RIGHT_DIVE = 26


@dataclass(frozen=True, slots=True)
class BlueBehaviorTreeConfig:
    low_altitude_m: float = 1_500.0
    min_altitude_m: float = 1_000.0
    predicted_altitude_m: float = 1_200.0
    altitude_prediction_s: float = 2.0
    speed_min_mps: float = 250.0
    speed_recovery_altitude_m: float = 3_000.0
    effective_threat_range_m: float = 80_000.0
    medium_threat_range_m: float = 20_000.0
    high_threat_range_m: float = 8_000.0
    cruise_speed_mps: float = 320.0
    attitude_tolerance_rad: float = math.radians(5.0)
    heading_tolerance_rad: float = math.radians(8.0)
    beam_dwell_steps: int = 4
    break_dwell_steps: int = 6
    seed: int = 0


@dataclass(slots=True)
class BlueBehaviorTreePolicy:
    """Finite-state baseline with safety, speed, cruise, beam, and break rules."""

    config: BlueBehaviorTreeConfig = BlueBehaviorTreeConfig()
    state: str = "cruise"
    dwell_steps_remaining: int = 0
    break_sign: int = 1

    def __post_init__(self) -> None:
        self._rng = random.Random(self.config.seed)
        self._cruise_heading: float | None = None
        self._cruise_gamma: float = 0.0

    def reset(self, seed: int | WorldSnapshot | None = None, snapshot: WorldSnapshot | None = None) -> None:
        if isinstance(seed, WorldSnapshot):
            snapshot = seed
            seed = None
        self.state = "cruise"
        self.dwell_steps_remaining = 0
        if seed is not None:
            self._rng.seed(seed)
        self.break_sign = 1
        self._cruise_heading = snapshot.blue.kinematics.angles.psi if snapshot is not None else None
        self._cruise_gamma = snapshot.blue.kinematics.angles.gamma if snapshot is not None else 0.0


    def act(self, observation: Sequence[float], action_mask: Sequence[bool] | None = None) -> int:
        """Select an action from the same observation/mask shape used by RL policies."""
        obs = _ParsedObservation.from_observation(observation)
        if self._cruise_heading is None:
            self._cruise_heading = obs.psi
            self._cruise_gamma = obs.gamma
        if _observation_low_altitude_risk(obs, self.config):
            self.state = "safe"
            self.dwell_steps_remaining = 0
            return _masked(A_CLIMB, action_mask)
        if obs.speed_mps < self.config.speed_min_mps:
            self.state = "speed"
            self.dwell_steps_remaining = 0
            action = A_DIVE if obs.altitude_m > self.config.speed_recovery_altitude_m else A_ACCEL
            return _masked(action, action_mask)
        primary = obs.primary_threat(self.config.effective_threat_range_m)
        level = _threat_level(primary.distance_m, self.config) if primary is not None else 0
        if self.dwell_steps_remaining > 0 and self.state in {"beam", "break"}:
            self.dwell_steps_remaining -= 1
        else:
            if level == 3:
                if self.state != "break":
                    self.break_sign = self._rng.choice([-1, 1])
                self.state = "break"
                self.dwell_steps_remaining = self.config.break_dwell_steps
            elif level == 2:
                self.state = "beam"
                self.dwell_steps_remaining = self.config.beam_dwell_steps
            else:
                self.state = "cruise"
                self.dwell_steps_remaining = 0
        if self.state == "break" and primary is not None:
            action = self._break_action_from_observation(obs)
        elif self.state == "beam" and primary is not None:
            action = self._beam_action_from_observation(primary)
        else:
            action = self._cruise_action_from_observation(obs)
        return _masked(action, action_mask)

    def select_action(self, snapshot: WorldSnapshot) -> int:
        blue = snapshot.blue.kinematics
        if self._cruise_heading is None:
            self._cruise_heading = blue.angles.psi
            self._cruise_gamma = blue.angles.gamma
        if _low_altitude_risk(snapshot, self.config):
            self.state = "safe"
            self.dwell_steps_remaining = 0
            return A_CLIMB
        if blue.speed < self.config.speed_min_mps:
            self.state = "speed"
            self.dwell_steps_remaining = 0
            return A_DIVE if blue.position.y > self.config.speed_recovery_altitude_m else A_ACCEL
        primary, distance = _primary_threat(snapshot, self.config)
        level = _threat_level(distance, self.config) if primary is not None else 0
        if self.dwell_steps_remaining > 0 and self.state in {"beam", "break"}:
            self.dwell_steps_remaining -= 1
        else:
            if level == 3:
                if self.state != "break":
                    self.break_sign = self._rng.choice([-1, 1])
                self.state = "break"
                self.dwell_steps_remaining = self.config.break_dwell_steps
            elif level == 2:
                self.state = "beam"
                self.dwell_steps_remaining = self.config.beam_dwell_steps
            else:
                self.state = "cruise"
                self.dwell_steps_remaining = 0
        if self.state == "break" and primary is not None:
            return self._break_action(snapshot)
        if self.state == "beam" and primary is not None:
            return self._beam_action(snapshot, primary)
        return self._cruise_action(snapshot)

    def _cruise_action(self, snapshot: WorldSnapshot) -> int:
        blue = snapshot.blue.kinematics
        if blue.speed < self.config.cruise_speed_mps:
            return A_ACCEL
        gamma_error = self._cruise_gamma - blue.angles.gamma
        if gamma_error > self.config.attitude_tolerance_rad:
            return A_CLIMB
        if gamma_error < -self.config.attitude_tolerance_rad:
            return A_DIVE
        heading_error = normalize_angle_rad((self._cruise_heading or 0.0) - blue.angles.psi)
        if heading_error > self.config.heading_tolerance_rad:
            return A_RIGHT_TURN
        if heading_error < -self.config.heading_tolerance_rad:
            return A_LEFT_TURN
        return A_FORWARD

    def _beam_action(self, snapshot: WorldSnapshot, primary) -> int:
        blue = snapshot.blue.kinematics
        rel_azimuth = math.atan2(primary.kinematics.position.z - blue.position.z, primary.kinematics.position.x - blue.position.x)
        candidates = (normalize_angle_rad(rel_azimuth + math.pi / 2.0), normalize_angle_rad(rel_azimuth - math.pi / 2.0))
        target = min(candidates, key=lambda psi: abs(normalize_angle_rad(psi - blue.angles.psi)))
        error = normalize_angle_rad(target - blue.angles.psi)
        if abs(error) <= self.config.heading_tolerance_rad:
            return A_ACCEL
        return A_RIGHT_TURN if error > 0.0 else A_LEFT_TURN

    def _break_action(self, snapshot: WorldSnapshot) -> int:
        altitude = snapshot.blue.kinematics.position.y
        if altitude > self.config.speed_recovery_altitude_m:
            return A_RIGHT_DIVE if self.break_sign > 0 else A_LEFT_DIVE
        return self._rng.choice([A_RIGHT_TURN, A_RIGHT_CLIMB]) if self.break_sign > 0 else self._rng.choice([A_LEFT_TURN, A_LEFT_CLIMB])

    def _cruise_action_from_observation(self, obs: "_ParsedObservation") -> int:
        if obs.speed_mps < self.config.cruise_speed_mps:
            return A_ACCEL
        gamma_error = self._cruise_gamma - obs.gamma
        if gamma_error > self.config.attitude_tolerance_rad:
            return A_CLIMB
        if gamma_error < -self.config.attitude_tolerance_rad:
            return A_DIVE
        heading_error = normalize_angle_rad((self._cruise_heading or 0.0) - obs.psi)
        if heading_error > self.config.heading_tolerance_rad:
            return A_RIGHT_TURN
        if heading_error < -self.config.heading_tolerance_rad:
            return A_LEFT_TURN
        return A_FORWARD

    def _beam_action_from_observation(self, primary: "_ParsedThreat") -> int:
        candidates = (normalize_angle_rad(primary.bearing_rad + math.pi / 2.0), normalize_angle_rad(primary.bearing_rad - math.pi / 2.0))
        target = min(candidates, key=lambda angle: abs(normalize_angle_rad(angle)))
        if abs(target) <= self.config.heading_tolerance_rad:
            return A_ACCEL
        return A_RIGHT_TURN if target > 0.0 else A_LEFT_TURN

    def _break_action_from_observation(self, obs: "_ParsedObservation") -> int:
        if obs.altitude_m > self.config.speed_recovery_altitude_m:
            return A_RIGHT_DIVE if self.break_sign > 0 else A_LEFT_DIVE
        return self._rng.choice([A_RIGHT_TURN, A_RIGHT_CLIMB]) if self.break_sign > 0 else self._rng.choice([A_LEFT_TURN, A_LEFT_CLIMB])


def _low_altitude_risk(snapshot: WorldSnapshot, config: BlueBehaviorTreeConfig) -> bool:
    blue = snapshot.blue.kinematics
    vz = flight_velocity(blue).y
    predicted_altitude = blue.position.y + vz * config.altitude_prediction_s
    return (blue.position.y < config.low_altitude_m and vz < 0.0) or predicted_altitude < config.predicted_altitude_m or blue.position.y < config.min_altitude_m


def _primary_threat(snapshot: WorldSnapshot, config: BlueBehaviorTreeConfig):
    threats = []
    for missile in snapshot.missiles:
        if missile.alive and missile.locked:
            distance = math.sqrt(
                (missile.kinematics.position.x - snapshot.blue.kinematics.position.x) ** 2
                + (missile.kinematics.position.z - snapshot.blue.kinematics.position.z) ** 2
                + (missile.kinematics.position.y - snapshot.blue.kinematics.position.y) ** 2
            )
            if distance <= config.effective_threat_range_m:
                threats.append((distance, missile))
    if not threats:
        return None, math.inf
    distance, missile = min(threats, key=lambda item: item[0])
    return missile, distance


def _threat_level(distance: float, config: BlueBehaviorTreeConfig) -> int:
    if math.isinf(distance):
        return 0
    if distance < config.high_threat_range_m:
        return 3
    if distance < config.medium_threat_range_m:
        return 2
    return 1


@dataclass(frozen=True, slots=True)
class _ParsedThreat:
    distance_m: float
    closing_mps: float
    bearing_rad: float
    threat_score: float


@dataclass(frozen=True, slots=True)
class _ParsedObservation:
    altitude_m: float
    speed_mps: float
    gamma: float
    psi: float
    threats: tuple[_ParsedThreat, ...]

    @classmethod
    def from_observation(cls, observation: Sequence[float]) -> "_ParsedObservation":
        if len(observation) < 8:
            raise ValueError("blue escape observation must include the 8 ownship features")
        speed_scale = 2_000.0
        distance_scale = 100_000.0
        threats = []
        for offset in range(8, len(observation), 11):
            slot = observation[offset:offset + 11]
            if len(slot) < 11:
                break
            distance_m = float(slot[6]) * distance_scale
            if distance_m <= 0.0:
                continue
            threats.append(_ParsedThreat(
                distance_m=distance_m,
                closing_mps=float(slot[7]) * speed_scale,
                bearing_rad=float(slot[8]) * math.pi,
                threat_score=float(slot[10]),
            ))
        return cls(
            altitude_m=float(observation[3]) * 100_000.0,
            speed_mps=float(observation[4]) * speed_scale,
            gamma=float(observation[5]) * math.pi,
            psi=float(observation[6]) * math.pi,
            threats=tuple(threats),
        )

    def primary_threat(self, effective_range_m: float) -> _ParsedThreat | None:
        effective = [threat for threat in self.threats if threat.distance_m <= effective_range_m]
        return min(effective, key=lambda threat: threat.distance_m, default=None)


def _observation_low_altitude_risk(obs: _ParsedObservation, config: BlueBehaviorTreeConfig) -> bool:
    vz = obs.speed_mps * math.sin(obs.gamma)
    predicted_altitude = obs.altitude_m + vz * config.altitude_prediction_s
    return (obs.altitude_m < config.low_altitude_m and vz < 0.0) or predicted_altitude < config.predicted_altitude_m or obs.altitude_m < config.min_altitude_m


def _masked(action: int, action_mask: Sequence[bool] | None) -> int:
    if action_mask is None or action >= len(action_mask) or action_mask[action]:
        return action
    for idx, allowed in enumerate(action_mask):
        if allowed:
            return idx
    return A_FORWARD
