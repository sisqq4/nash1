"""Generate plots from existing Phase 3 evaluation artifacts."""
import argparse
from air_combat_rl.visualization import PlotDataError, plot_evaluation

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--evaluation-dir",required=True); p.add_argument("--output-dir")
    a=p.parse_args(argv)
    try: paths=plot_evaluation(a.evaluation_dir,a.output_dir)
    except PlotDataError as exc: raise SystemExit(str(exc))
    print("\n".join(map(str,paths)))
if __name__ == "__main__": main()
