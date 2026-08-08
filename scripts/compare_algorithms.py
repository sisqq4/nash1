"""Compare existing evaluation outputs under validated identical conditions."""
import argparse
from air_combat_rl.visualization import PlotDataError, compare_evaluations

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--evaluations",nargs="+",required=True); p.add_argument("--output-dir",required=True)
    a=p.parse_args(argv)
    try: paths=compare_evaluations(a.evaluations,a.output_dir)
    except (PlotDataError, OSError, ValueError) as exc: raise SystemExit(str(exc))
    print("\n".join(map(str,paths)))
if __name__ == "__main__": main()
