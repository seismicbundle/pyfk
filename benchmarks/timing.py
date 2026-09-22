"""Time calculate_gf for the problems quoted in the README.

    python benchmarks/timing.py [local|regional ...] [--reps N] [--json PATH]

Each run is one calculate_gf call in one process. Set OMP_NUM_THREADS to choose
the number of CPU threads. With PYFK_USE_CUDA=1 the first repetition includes
the kernel compilation, so use --reps 2 or more.

Model: 9 layers over a halfspace at 32 km depth, Vp 4.30 to 7.25 km/s,
Vp/Vs = 1.70, Gardner density, Q = 200 for P and S. Source depth 9 km,
dt = 0.01 s, dk = 0.3, kmax = 15, pmin = 0, pmax = 1.
"""
import argparse
import json
import platform
import statistics
import time

import numpy as np

DEPTH_TOP_KM = np.array([0.0, 0.2, 0.5, 1.0, 2.0, 4.0, 6.0, 10.0, 20.0, 32.0])
VP_KM_S = np.array([4.30, 5.06, 5.33, 5.60, 5.87, 6.09, 6.35, 6.74, 7.05, 7.25])
MODEL = np.column_stack([
    np.append(np.diff(DEPTH_TOP_KM), 0.0),  # thickness, 0 for the halfspace
    VP_KM_S / 1.70,                          # vs
    VP_KM_S,                                 # vp
    1.741 * VP_KM_S ** 0.25,                 # density (Gardner)
    np.full(len(VP_KM_S), 200.0),            # Qs
    np.full(len(VP_KM_S), 200.0),            # Qp
])

PROBLEMS = {
    "local": dict(npt=8192, distances_km=np.linspace(2.0, 30.0, 23)),
    "regional": dict(npt=16384, distances_km=np.linspace(5.0, 130.0, 23)),
}


def build_config(name):
    from pyfk import Config, SeisModel, SourceModel
    problem = PROBLEMS[name]
    return Config(model=SeisModel(model=MODEL.copy()),
                  source=SourceModel(sdep=9.0, srcType="dc",
                                     source_mechanism=[2.0, 30.0, 80.0, 5.0]),
                  receiver_distance=list(problem["distances_km"]),
                  npt=problem["npt"], dt=0.01, dk=0.3, taper=0.3, pmin=0.0,
                  pmax=1.0, kmax=15.0, smth=1.0, rdep=0.0, updn="all",
                  samples_before_first_arrival=700)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help=", ".join(PROBLEMS))
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    unknown = sorted(set(args.names) - set(PROBLEMS))
    if unknown:
        parser.error(f"unknown problem: {', '.join(unknown)}")

    import pyfk
    from pyfk import calculate_gf
    print(f"pyfk {pyfk.__version__}, python {platform.python_version()}, {platform.platform()}")
    results = {}
    for name in args.names or list(PROBLEMS):
        times = []
        for rep in range(args.reps):
            config = build_config(name)
            started = time.perf_counter()
            calculate_gf(config)
            times.append(time.perf_counter() - started)
            print(f"{name} rep {rep + 1}: {times[-1]:.2f} s", flush=True)
        warm = times[1:] if len(times) > 1 else times
        results[name] = dict(npt=PROBLEMS[name]["npt"],
                             receivers=len(PROBLEMS[name]["distances_km"]),
                             times_s=times, median_s=statistics.median(warm))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(dict(pyfk=pyfk.__version__, python=platform.python_version(),
                           platform=platform.platform(), results=results), handle, indent=1)
            handle.write("\n")


if __name__ == "__main__":
    main()
