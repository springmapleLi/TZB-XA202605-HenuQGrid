"""Train and evaluate the UC-QAOA16 QEM baselines and graph models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import Hierarchical_GNN_QEM as qem


MODELS = {
    "mlp": qem.QAOA16MLP,
    "global": qem.QAOA16GNN,
    "hierarchical": qem.QAOA16HierarchicalGNN,
    "edge_aware": qem.QAOA16EdgeAwareHierarchicalGNN,
    "dual_branch": qem.QAOA16DualBranchHierarchicalGNN,
}


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    noisy_error = frame["distributed_noisy"].to_numpy() - frame["logical_ideal"].to_numpy()
    mitigated_error = frame["mitigated"].to_numpy() - frame["logical_ideal"].to_numpy()
    noisy_mae = float(np.mean(np.abs(noisy_error)))
    mitigated_mae = float(np.mean(np.abs(mitigated_error)))
    return {
        "noisy_mae": noisy_mae,
        "mitigated_mae": mitigated_mae,
        "mae_reduction_percent": 100.0 * (1.0 - mitigated_mae / noisy_mae) if noisy_mae else 0.0,
        "mitigated_rmse": float(np.sqrt(np.mean(mitigated_error**2))),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("output")
    parser.add_argument("--model", choices=tuple(MODELS), default="hierarchical")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--hidden", type=int, default=160)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    data = qem.load_qft16(args.dataset)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if args.model == "mlp":
        model = MODELS[args.model](hidden=args.hidden).to(device)
        history = qem.train_mlp(
            model, data["train"], data["val"], args.epochs, args.patience,
            args.batch_size, args.learning_rate,
        )
        predictions = qem.predict_mlp(model, data["test"])
    else:
        if args.model == "global":
            builder = lambda rows: qem.build_graphs(rows, "logical_global")
        else:
            builder = qem.build_dual_graphs

        train_graphs, val_graphs, test_graphs = (
            builder(data[split]) for split in ("train", "val", "test")
        )
        model = MODELS[args.model](hidden=args.hidden).to(device)
        history = qem.train_graph(
            model, train_graphs, val_graphs, args.epochs, args.patience,
            args.batch_size, args.learning_rate,
        )
        predictions = qem.predict_graph(model, test_graphs, qem.names(data["test"][0]))

    torch.save({"model": args.model, "state_dict": model.state_dict()}, output / f"{args.model}.pt")
    history.to_csv(output / f"{args.model}_history.csv", index=False)
    predictions.to_csv(output / f"{args.model}_predictions.csv", index=False)
    summary = {"model": args.model, "device": str(device), **metrics(predictions)}
    (output / f"{args.model}_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(main(build_parser().parse_args()), ensure_ascii=False, indent=2))
