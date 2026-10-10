from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from gnn import Config as GNNConfig
from gnn import GATv2MultiLabel, MoleculeFeaturizer
from train_descriptor_xgb import AllRDKitDescriptorFeaturizer
from train_fingerprint_xgb import MorganFingerprintFeaturizer


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required model metadata is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_input(path: Path) -> tuple[pd.DataFrame, list]:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "SMILES" not in frame.columns or frame.empty:
        raise ValueError("Input CSV must contain a nonempty SMILES column.")
    frame = frame.copy()
    if "compound_name" not in frame.columns:
        frame.insert(0, "compound_name", [f"molecule_{i + 1}" for i in range(len(frame))])
    frame = frame[["compound_name", "SMILES"]].reset_index(drop=True)
    molecules = []
    for i, smiles in enumerate(frame["SMILES"]):
        if pd.isna(smiles) or not str(smiles).strip():
            raise ValueError(f"Blank SMILES in CSV row {i + 2}.")
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            raise ValueError(f"Invalid SMILES in CSV row {i + 2}: {smiles}")
        frame.at[i, "SMILES"] = str(smiles).strip()
        molecules.append(mol)
    return frame, molecules


def predict_xgb(model_dir: Path, labels: list[str], features: np.ndarray) -> np.ndarray:
    columns = []
    for index, label in enumerate(labels):
        path = model_dir / "models" / f"xgb_{index}_{label}.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"Required fitted model is missing: {path}")
        with path.open("rb") as stream:
            model = pickle.load(stream)
        columns.append(model.predict_proba(features)[:, 1])
    return np.column_stack(columns)


def predict_graph(molecules: list, checkpoint_path: Path, labels: list[str], device: torch.device) -> np.ndarray:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Required graph checkpoint is missing: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if list(checkpoint.get("label_cols", [])) != labels:
        raise ValueError("Graph checkpoint and XGBoost metadata have different label orders.")
    config = GNNConfig()
    config.gnn_input_dim = int(checkpoint["gnn_input_dim"])
    config.gnn_output_dim = int(checkpoint["gnn_output_dim"])
    if config.gnn_output_dim != len(labels):
        raise ValueError("Graph checkpoint output count differs from the label count.")
    model = GATv2MultiLabel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    featurizer = MoleculeFeaturizer()
    graphs = []
    for mol in molecules:
        x, edge_index, edge_attr = featurizer(mol)
        if x.shape[1] != config.gnn_input_dim:
            raise ValueError("Graph feature dimension differs from the training checkpoint.")
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
    values = []
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=64, shuffle=False):
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            values.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/new_molecule_predictions.csv"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    args = parser.parse_args()

    fp_dir = args.results_dir / "fp_xgb_output"
    desc_dir = args.results_dir / "desc_xgb_output"
    graph_file = args.results_dir / "gnn_output" / "gatv2_shared_backbone_independent_heads.pt"
    threshold_file = args.results_dir / "simple_avg_threshold_optimized_output" / "optimized_thresholds.npy"
    fp_meta = read_json(fp_dir / "meta.json")
    desc_meta = read_json(desc_dir / "meta.json")
    labels = list(fp_meta["label_cols"])
    if labels != list(desc_meta["label_cols"]):
        raise ValueError("Fingerprint and descriptor metadata have different label orders.")
    if not threshold_file.is_file():
        raise FileNotFoundError(f"Required validation-based thresholds are missing: {threshold_file}")
    thresholds = np.asarray(np.load(threshold_file), dtype=float).reshape(-1)
    if len(thresholds) != len(labels) or not np.all((thresholds > 0) & (thresholds < 1)):
        raise ValueError("Saved thresholds do not match the model labels.")

    frame, molecules = read_input(args.input)
    fp_featurizer = MorganFingerprintFeaturizer(int(fp_meta["fp_radius"]), int(fp_meta["fp_nbits"]))
    fp_features = np.vstack([fp_featurizer(mol) for mol in molecules]).astype(np.float32)
    fp_probabilities = predict_xgb(fp_dir, labels, fp_features)

    desc_featurizer = AllRDKitDescriptorFeaturizer()
    names = desc_featurizer.get_feature_names()
    if names != list(desc_meta["raw_descriptor_names"]):
        raise ValueError("Current RDKit descriptor order differs from the training environment.")
    index_by_name = {name: i for i, name in enumerate(names)}
    positions = [index_by_name[name] for name in desc_meta["used_descriptor_names"]]
    desc_features = np.vstack([desc_featurizer(mol) for mol in molecules])[:, positions]
    desc_features = np.where(np.isfinite(desc_features), desc_features, 0.0).astype(np.float32)
    desc_probabilities = predict_xgb(desc_dir, labels, desc_features)

    device = torch.device(args.device)
    graph_probabilities = predict_graph(molecules, graph_file, labels, device)
    probabilities = (fp_probabilities + desc_probabilities + graph_probabilities) / 3.0
    predicted = probabilities >= thresholds.reshape(1, -1)
    frame["predicted_labels"] = [
        ";".join(label for label, flag in zip(labels, row) if flag)
        for row in predicted
    ]
    for i, label in enumerate(labels):
        frame[f"prob_{label}"] = probabilities[:, i]
        frame[f"pred_{label}"] = predicted[:, i].astype(int)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(frame)} predictions to {args.output}")


if __name__ == "__main__":
    main()
