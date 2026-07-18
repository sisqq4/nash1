"""Rule-aligned modular blue escape reward components."""
from __future__ import annotations

import math
from dataclasses import dataclass
from air_combat_rl.core.math3d import flight_velocity
from air_combat_rl.domain.states import MissileState
from air_combat_rl.simulation.snapshot import WorldSnapshot


@dataclass(frozen=True, slots=True)
class RewardConfig:
    safe_distance_m: float = 30_000.0
    short_range_m: float = 8_000.0
    narrow_bearing_rad: float = math.radians(30.0)
    near_buffer_m: float = 10_000.0
    min_altitude_m: float = 1_000.0
    max_altitude_m: float = 20_000.0
    ground_risk_altitude_m: float = 1_000.0
    speed_min_mps: float = 250.0
    speed_max_mps: float = 600.0
    missile_speed_min_mps: float = 250.0
    missile_speed_max_mps: float = 1_800.0
    terminal_ground: float = -20.0
    terminal_hit: float = -10.0
    terminal_success: float = 10.0
    terminal_exhausted: float = 10.0
    terminal_timeout: float = 0.0
    w_short_distance: float = 1.0
    w_short_roll: float = 0.4
    w_short_turn: float = 0.6
    w_short_speed: float = 0.3
    w_short_height: float = 0.4
    w_mid_small_azimuth: float = 1.0
    w_mid_small_height: float = 0.8
    w_mid_small_opposite: float = 0.8
    w_mid_small_speed: float = 0.4
    w_mid_small_level: float = 0.4
    w_mid_large_distance: float = 1.0
    w_mid_large_azimuth: float = 1.0
    w_mid_large_height: float = 0.8
    w_mid_large_speed: float = 0.4
    w_mid_large_level: float = 0.4
    w_mid_large_buffer: float = 0.6
    w_threat_relief: float = 0.25
    w_threat_high_penalty: float = 1.5
    w_multi_distance: float = 1.0
    w_multi_height: float = 0.5
    w_multi_threat_relief: float = 0.8
    w_multi_threat_increase: float = -1.2
    w_multi_encirclement: float = -1.0
    w_multi_ground: float = -1.0
    encirclement_sync_tau_s: float = 8.0
    corridor_width_ref_m: float = 2_000.0
    shaping_scale: float = 0.1
    smooth_weight: float = -0.02


class EscapeReward:
    def __init__(self, config: RewardConfig = RewardConfig()) -> None:
        self.config = config
        self.prev_min_distance: float | None = None
        self.prev_mean_distance: float | None = None
        self.prev_threat: float | None = None
        self.prev_heading: float | None = None
        self.prev_action: int | None = None

    def reset(self, snapshot: WorldSnapshot) -> None:
        self.prev_min_distance = _min_distance(snapshot)
        self.prev_mean_distance = _mean_distance(snapshot)
        self.prev_threat = _aggregate_threat(snapshot, self.config)
        self.prev_heading = snapshot.blue.kinematics.angles.psi
        self.prev_action = None

    def compute(self, snapshot: WorldSnapshot, outcome: str, action_id: int) -> tuple[float, dict[str, float]]:
        terminal = _terminal_reward(outcome, self.config)
        if outcome != "running":
            comps = _zero_components()
            comps["terminal"] = terminal
            self._update_previous(snapshot, action_id)
            return terminal, comps

        active = [m for m in snapshot.missiles if m.alive and m.locked]
        if len(active) > 1:
            comps = self._compute_multi(snapshot, action_id)
        else:
            comps = self._compute_single(snapshot, action_id)
        comps["terminal"] = terminal
        reward = float(sum(comps.values()))
        self._update_previous(snapshot, action_id)
        return reward, comps

    def _compute_single(self, snapshot: WorldSnapshot, action_id: int) -> dict[str, float]:
        comps = _zero_components()
        primary = _primary_threat(snapshot)
        if primary is None:
            comps["height"] = self.config.shaping_scale * _height_score(snapshot, self.config)
            comps["smooth"] = _smooth(self.prev_action, action_id, self.config)
            return comps
        distance = _distance(primary.kinematics, snapshot.blue.kinematics)
        bearing = abs(_bearing(snapshot, primary))
        delta_d = 0.0 if self.prev_min_distance is None or math.isinf(self.prev_min_distance) else (distance - self.prev_min_distance) / self.config.safe_distance_m
        height = _height_score(snapshot, self.config)
        speed = _speed_score(snapshot.blue.kinematics.speed, self.config)
        level_penalty = _level_penalty(snapshot)
        azimuth_score = min(bearing / self.config.narrow_bearing_rad, 1.0) - 1.0
        opposite = _opposite_motion_score(snapshot, primary)
        turn_score = _turn_score(self.prev_heading, snapshot.blue.kinematics.angles.psi)
        roll_proxy = min(abs(snapshot.blue.kinematics.angles.gamma) / math.radians(45.0), 1.0)
        if distance < self.config.short_range_m:
            mode = (
                self.config.w_short_distance * delta_d
                + self.config.w_short_roll * roll_proxy
                + self.config.w_short_turn * turn_score
                + self.config.w_short_speed * speed
                + self.config.w_short_height * height
            )
        elif bearing <= self.config.narrow_bearing_rad:
            mode = (
                self.config.w_mid_small_azimuth * azimuth_score
                + self.config.w_mid_small_height * height
                + self.config.w_mid_small_opposite * opposite
                + self.config.w_mid_small_speed * speed
                - self.config.w_mid_small_level * level_penalty
            )
        else:
            buffer_score = max(0.0, (self.config.near_buffer_m - distance) / max(self.config.near_buffer_m - self.config.short_range_m, 1.0))
            mode = (
                self.config.w_mid_large_distance * delta_d
                + self.config.w_mid_large_azimuth * azimuth_score
                + self.config.w_mid_large_height * height
                + self.config.w_mid_large_speed * speed
                - self.config.w_mid_large_level * level_penalty
                + self.config.w_mid_large_buffer * buffer_score * (1.0 - level_penalty)
            )
        threat = _aggregate_threat(snapshot, self.config)
        prev_threat = threat if self.prev_threat is None else self.prev_threat
        comps["separation"] = self.config.shaping_scale * mode
        comps["threat"] = self.config.shaping_scale * _threat_change_reward(prev_threat, threat, self.config.w_threat_relief, self.config.w_threat_relief, self.config)
        comps["height"] = self.config.shaping_scale * height
        comps["ground"] = self.config.shaping_scale * _ground_penalty(snapshot, self.config)
        comps["smooth"] = _smooth(self.prev_action, action_id, self.config)
        return comps

    def _compute_multi(self, snapshot: WorldSnapshot, action_id: int) -> dict[str, float]:
        comps = _zero_components()
        mean_d = _mean_distance(snapshot)
        prev_mean = mean_d if self.prev_mean_distance is None or math.isinf(self.prev_mean_distance) else self.prev_mean_distance
        threat = _aggregate_threat(snapshot, self.config)
        prev_threat = threat if self.prev_threat is None else self.prev_threat
        delta_d = 0.0 if math.isinf(mean_d) else (mean_d - prev_mean) / self.config.safe_distance_m
        comps["separation"] = self.config.shaping_scale * self.config.w_multi_distance * delta_d
        comps["height"] = self.config.shaping_scale * self.config.w_multi_height * _height_score(snapshot, self.config)
        comps["threat"] = self.config.shaping_scale * _threat_change_reward(prev_threat, threat, self.config.w_multi_threat_relief, self.config.w_multi_threat_increase, self.config)
        comps["encirclement"] = self.config.shaping_scale * self.config.w_multi_encirclement * _encirclement(snapshot, self.config)
        comps["ground"] = self.config.shaping_scale * self.config.w_multi_ground * _ground_risk(snapshot, self.config)
        comps["smooth"] = _smooth(self.prev_action, action_id, self.config)
        return comps

    def _update_previous(self, snapshot: WorldSnapshot, action_id: int) -> None:
        self.prev_min_distance = _min_distance(snapshot)
        self.prev_mean_distance = _mean_distance(snapshot)
        self.prev_threat = _aggregate_threat(snapshot, self.config)
        self.prev_heading = snapshot.blue.kinematics.angles.psi
        self.prev_action = action_id


def _zero_components() -> dict[str, float]:
    return {"terminal": 0.0, "separation": 0.0, "threat": 0.0, "height": 0.0, "encirclement": 0.0, "ground": 0.0, "smooth": 0.0}


def _terminal_reward(outcome: str, config: RewardConfig) -> float:
    if outcome == "hit": return config.terminal_hit
    if outcome == "crash": return config.terminal_ground
    if outcome == "success": return config.terminal_success
    if outcome == "exhausted": return config.terminal_exhausted
    if outcome == "timeout": return config.terminal_timeout
    return 0.0


def _active(snapshot: WorldSnapshot) -> list[MissileState]:
    return [m for m in snapshot.missiles if m.alive and m.locked]


def _primary_threat(snapshot: WorldSnapshot) -> MissileState | None:
    active = _active(snapshot)
    return min(active, key=lambda m: _distance(m.kinematics, snapshot.blue.kinematics), default=None)


def _min_distance(snapshot: WorldSnapshot) -> float:
    ds = [_distance(m.kinematics, snapshot.blue.kinematics) for m in _active(snapshot)]
    return min(ds) if ds else math.inf


def _mean_distance(snapshot: WorldSnapshot) -> float:
    ds = [_distance(m.kinematics, snapshot.blue.kinematics) for m in _active(snapshot)]
    return sum(ds) / len(ds) if ds else math.inf


def _distance(a, b) -> float:
    return math.sqrt((a.position.x-b.position.x)**2 + (a.position.z-b.position.z)**2 + (a.position.y-b.position.y)**2)


def _bearing(snapshot: WorldSnapshot, missile: MissileState) -> float:
    bp = snapshot.blue.kinematics.position
    mp = missile.kinematics.position
    threat_az = math.atan2(mp.z - bp.z, mp.x - bp.x)
    return _wrap(threat_az - snapshot.blue.kinematics.angles.psi)


def _height_score(snapshot: WorldSnapshot, config: RewardConfig) -> float:
    y = snapshot.blue.kinematics.position.y
    if config.min_altitude_m <= y <= config.max_altitude_m:
        return 1.0
    if y < config.min_altitude_m:
        return -min((config.min_altitude_m - y) / config.min_altitude_m, 1.0)
    return -min((y - config.max_altitude_m) / config.max_altitude_m, 1.0)


def _ground_risk(snapshot: WorldSnapshot, config: RewardConfig) -> float:
    y = max(snapshot.blue.kinematics.position.y, 0.0)
    return max(0.0, (config.ground_risk_altitude_m - y) / config.ground_risk_altitude_m)


def _ground_penalty(snapshot: WorldSnapshot, config: RewardConfig) -> float:
    return -_ground_risk(snapshot, config)


def _speed_score(speed: float, config: RewardConfig) -> float:
    return max(0.0, min((speed - config.speed_min_mps) / max(config.speed_max_mps - config.speed_min_mps, 1.0), 1.0))


def _level_penalty(snapshot: WorldSnapshot) -> float:
    return min(abs(snapshot.blue.kinematics.angles.gamma) / math.radians(45.0), 1.0)


def _turn_score(prev_heading: float | None, heading: float) -> float:
    if prev_heading is None:
        return 0.0
    return min(abs(_wrap(heading - prev_heading)) / math.radians(15.0), 1.0)


def _opposite_motion_score(snapshot: WorldSnapshot, missile: MissileState) -> float:
    bp = snapshot.blue.kinematics.position
    mp = missile.kinematics.position
    bv = flight_velocity(snapshot.blue.kinematics)
    rel_from_missile = (bp.x - mp.x, bp.z - mp.z, bp.y - mp.y)
    rel_norm = max(math.sqrt(sum(v * v for v in rel_from_missile)), 1.0e-6)
    bv_norm = max(math.sqrt(bv.x * bv.x + bv.z * bv.z + bv.y * bv.y), 1.0e-6)
    dot = (bv.x * rel_from_missile[0] + bv.z * rel_from_missile[1] + bv.y * rel_from_missile[2]) / (bv_norm * rel_norm)
    return (dot + 1.0) / 2.0


def _aggregate_threat(snapshot: WorldSnapshot, config: RewardConfig) -> float:
    threats = [_single_threat(snapshot, m, config) for m in _active(snapshot)]
    return sum(threats) / len(threats) if threats else 0.0


def _single_threat(snapshot: WorldSnapshot, missile: MissileState, config: RewardConfig) -> float:
    bp = snapshot.blue.kinematics.position
    mp = missile.kinematics.position
    bv = flight_velocity(snapshot.blue.kinematics)
    mv = flight_velocity(missile.kinematics)
    rx, rz, ry = bp.x - mp.x, bp.z - mp.z, bp.y - mp.y
    rvx, rvz, rvy = bv.x - mv.x, bv.z - mv.z, bv.y - mv.y
    distance = max(math.sqrt(rx * rx + rz * rz + ry * ry), 1.0e-6)
    radial_rate = (rvx * rx + rvz * rz + rvy * ry) / distance
    closing = max(0.0, -radial_rate)
    tgo = distance / max(closing, 1.0e-6)
    los_rate = math.sqrt((rz * rvy - ry * rvz) ** 2 + (ry * rvx - rx * rvy) ** 2 + (rx * rvz - rz * rvx) ** 2) / max(distance * distance, 1.0e-6)
    missile_energy = max(0.0, min((missile.kinematics.speed - config.missile_speed_min_mps) / max(config.missile_speed_max_mps - config.missile_speed_min_mps, 1.0), 1.0))
    score = 1.0 / distance + 1.2 * closing / config.missile_speed_max_mps + 1.1 / max(tgo, 1.0) + 0.6 * los_rate + 0.8 * missile_energy
    return 1.0 / (1.0 + math.exp(-score))


def _threat_change_reward(prev_threat: float, threat: float, relief_weight: float, increase_weight: float, config: RewardConfig) -> float:
    reward = relief_weight * threat if threat <= prev_threat else increase_weight * (threat - prev_threat)
    if threat > 0.6:
        reward -= config.w_threat_high_penalty * (threat - 0.6)
    return reward


def _encirclement(snapshot: WorldSnapshot, config: RewardConfig) -> float:
    active = _active(snapshot)
    if len(active) <= 1:
        return 0.0
    bp = snapshot.blue.kinematics.position
    bearings = sorted(math.atan2(m.kinematics.position.z - bp.z, m.kinematics.position.x - bp.x) for m in active)
    gaps = [(bearings[(i + 1) % len(bearings)] - bearings[i]) % (2.0 * math.pi) for i in range(len(bearings))]
    angular_coverage = 1.0 - max(gaps) / (2.0 * math.pi)
    tgos = [_time_to_go(snapshot, m) for m in active]
    mean_tgo = sum(tgos) / len(tgos)
    variance = sum((t - mean_tgo) ** 2 for t in tgos) / len(tgos)
    sync = math.exp(-variance / (config.encirclement_sync_tau_s ** 2))
    heading = snapshot.blue.kinematics.angles.psi
    side = (-math.sin(heading), math.cos(heading))
    safe_width = min(abs((m.kinematics.position.x - bp.x) * side[0] + (m.kinematics.position.z - bp.z) * side[1]) for m in active)
    corridor = 1.0 - max(0.0, min(safe_width / config.corridor_width_ref_m, 1.0))
    return 0.4 * angular_coverage + 0.35 * sync + 0.25 * corridor


def _time_to_go(snapshot: WorldSnapshot, missile: MissileState) -> float:
    bp = snapshot.blue.kinematics.position
    mp = missile.kinematics.position
    bv = flight_velocity(snapshot.blue.kinematics)
    mv = flight_velocity(missile.kinematics)
    rx, rz, ry = bp.x - mp.x, bp.z - mp.z, bp.y - mp.y
    distance = max(math.sqrt(rx * rx + rz * rz + ry * ry), 1.0e-6)
    rvx, rvz, rvy = bv.x - mv.x, bv.z - mv.z, bv.y - mv.y
    closing = max(0.0, -((rvx * rx + rvz * rz + rvy * ry) / distance))
    return distance / max(closing, 1.0e-6)


def _smooth(prev_action: int | None, action_id: int, config: RewardConfig) -> float:
    return 0.0 if prev_action is None else config.smooth_weight * abs(action_id - prev_action) / 28.0


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
