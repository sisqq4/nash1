import math

from air_combat_rl.core.coordinates import VecXZY, normalize_angle_rad
from air_combat_rl.core.timebase import SimulationClock
from air_combat_rl.domain.commands import ManeuverCommand


def test_vec_xzy_serializes_altitude_as_y():
    vec = VecXZY.from_xzy([1.0, 2.0, 3.0])
    assert vec.as_xzy() == (1.0, 2.0, 3.0)
    assert vec.altitude == 3.0


def test_angle_normalization_range():
    assert -math.pi <= normalize_angle_rad(3.5 * math.pi) < math.pi


def test_clock_uses_twenty_substeps():
    assert SimulationClock(physics_dt=0.005, policy_dt=0.1).substeps_per_policy_step == 20


def test_maneuver_command_resolves_nf_gamma_s():
    command = ManeuverCommand(nx=0.0, nf=2.0, gamma_s=math.pi / 2.0)
    assert abs(command.ny) < 1e-12
    assert command.nz == 2.0
