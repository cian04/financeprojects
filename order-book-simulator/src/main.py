"""
main.py — Entry point for the order book simulator.

Usage:
    python3 src/main.py
    python3 src/main.py --informed 0.2 --T 7200 --seed 123
"""
from __future__ import annotations
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
from simulator import Simulator, SimulationParams
from analytics import run_analytics

RESULTS_DIR = Path(__file__).parent.parent / "results"


def parse_args():
    p = argparse.ArgumentParser(description="Order Book Microstructure Simulator")
    p.add_argument("--T",          type=float, default=3600.0, help="Simulation horizon (seconds)")
    p.add_argument("--informed",   type=float, default=0.1,   help="Informed trader fraction")
    p.add_argument("--lambda_u",   type=float, default=2.0,   help="Uninformed arrival rate")
    p.add_argument("--spread",     type=float, default=0.02,  help="MM base half-spread")
    p.add_argument("--seed",       type=int,   default=42,    help="Random seed")
    return p.parse_args()


def main():
    args = parse_args()

    params = SimulationParams(
        T=args.T,
        informed_fraction=args.informed,
        lambda_uninformed=args.lambda_u,
        mm_base_spread=args.spread,
        seed=args.seed,
    )

    print("="*55)
    print("  ORDER BOOK MICROSTRUCTURE SIMULATOR")
    print("="*55)
    print(f"  Horizon          : {params.T:.0f}s ({params.T/3600:.1f}h)")
    print(f"  Uninformed rate  : {params.lambda_uninformed} orders/s")
    print(f"  Informed fraction: {params.informed_fraction:.0%}")
    print(f"  MM base spread   : {params.mm_base_spread}")
    print(f"  Seed             : {params.seed}")
    print("="*55 + "\n")

    sim     = Simulator(params)
    results = sim.run()

    run_analytics(results, RESULTS_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
