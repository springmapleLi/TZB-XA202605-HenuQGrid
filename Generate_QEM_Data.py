"""Generate the 16-qubit UC-QAOA dynamic-TeleGate QEM dataset."""
from __future__ import annotations

import argparse
import json

from UC_QAOA16 import DEFAULT_MASTER_SEED, DEFAULT_P_RANGE, SHOTS, generate_dataset, preflight


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="dataset root containing train/val/test folders")
    parser.add_argument("--train", type=int, default=1600)
    parser.add_argument("--val", type=int, default=100)
    parser.add_argument("--test", type=int, default=100)
    parser.add_argument("--shots", type=int, default=SHOTS)
    parser.add_argument("--p-min", type=int, default=DEFAULT_P_RANGE[0])
    parser.add_argument("--p-max", type=int, default=DEFAULT_P_RANGE[1])
    parser.add_argument("--seed", type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument("--probes", type=int, default=16)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(args: argparse.Namespace) -> dict:
    p_range = (args.p_min, args.p_max)
    if args.preflight_only:
        return preflight(args.seed, p_range, args.probes)
    return generate_dataset(
        args.output,
        split_sizes={"train": args.train, "val": args.val, "test": args.test},
        master_seed=args.seed,
        p_range=p_range,
        shots=args.shots,
        force=args.force,
    )


if __name__ == "__main__":
    print(json.dumps(main(build_parser().parse_args()), ensure_ascii=False, indent=2))
