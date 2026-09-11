"""
run_all.py -- run both simulations.

    python run_all.py            # full run
    python run_all.py --sim a    # composition laws only
    python run_all.py --sim b    # phase codes only
    python run_all.py --figs     # build figures from existing .npz

Simulation A (composition laws) writes results.npz and Figures S1, S2.
Simulation B (phase codes)      writes results_phase.npz and Figure S3.

"""
import argparse
import os
import subprocess
import sys
import time

SIM_A = [("run_sim.py", "composition laws: associator sweeps and the depth ladder"),
         ("make_figs.py", "Figures S1 and S2")]
SIM_B = [("run_phase.py", "phase codes: angular vs containment"),
         ("make_fig3.py", "Figure S3")]


def run(script, what):
    print(f"\n{'=' * 70}\n{script}  --  {what}\n{'=' * 70}", flush=True)
    t0 = time.time()
    if subprocess.run([sys.executable, script]).returncode != 0:
        raise SystemExit(f"step failed: {script}")
    print(f"[{time.time() - t0:.1f}s]", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", choices=["a", "b", "both"], default="both")
    ap.add_argument("--figs", action="store_true",
                    help="skip the simulations, rebuild figures from existing .npz")
    args = ap.parse_args()

    steps = []
    if args.sim in ("a", "both"):
        steps += SIM_A
    if args.sim in ("b", "both"):
        steps += SIM_B
    if args.figs:
        steps = [s for s in steps if s[0].startswith("make_")]
        for npz in ("results.npz", "results_phase.npz"):
            need = (npz == "results.npz" and args.sim in ("a", "both")) or \
                   (npz == "results_phase.npz" and args.sim in ("b", "both"))
            if need and not os.path.exists(npz):
                raise SystemExit(f"missing {npz}; run without --figs first")

    for script, what in steps:
        run(script, what)
    print(f"\ndone. results and figures are in {os.path.abspath('.')}")


if __name__ == "__main__":
    main()
