"""Generate plots from an existing single-scenario run; never runs simulation."""
import argparse
from air_combat_rl.visualization import PlotDataError, plot_run

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--run-dir",required=True); p.add_argument("--output-dir")
    a=p.parse_args(argv)
    try: paths=plot_run(a.run_dir,a.output_dir)
    except PlotDataError as exc: raise SystemExit(str(exc))
    print("\n".join(map(str,paths)))
if __name__ == "__main__": main()
