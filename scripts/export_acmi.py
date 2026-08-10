"""Convert a recorded Phase-1 trajectory JSONL file to Tacview ACMI 2.1."""
from __future__ import annotations

import argparse

from src.air_combat_rl.io.acmi_writer import trajectory_jsonl_to_acmi
from src.air_combat_rl.io.coordinate_transform import GeodeticOrigin


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--origin-lat-deg", required=True, type=float)
    parser.add_argument("--origin-lon-deg", required=True, type=float)
    parser.add_argument("--origin-alt-m", required=True, type=float)
    parser.add_argument("--reference-time", help="ISO-8601 UTC time (defaults to export time)")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    origin = GeodeticOrigin(args.origin_lat_deg, args.origin_lon_deg, args.origin_alt_m)
    trajectory_jsonl_to_acmi(args.trajectory, args.output, origin=origin,
                             reference_time=args.reference_time)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
