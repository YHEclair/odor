import os
import json
import math
import random
import warnings
import pandas as pd
import numpy as np

from collections import defaultdict, Counter

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D


warnings.filterwarnings("ignore")


class Config:
    train_csv = "./data/split/train.csv"
    valid_csv = "./data/split/valid.csv"
    test_csv = "./data/split/test.csv"

    smiles_col = "SMILES"


    fp_radius = 3
    fp_nbits = 2048


    per_label_top_bits_json = "./results/interpretability_analysis/02_fingerprint_shap/raw_outputs/07_fig4b_per_label_top_bits.json"


    union_matrix_csv = "./results/interpretability_analysis/02_fingerprint_shap/raw_outputs/06_fig4b_union_bit_matrix.csv"


    save_root = "./results/interpretability_analysis/02_fingerprint_shap/03_bit_substructure_mapping"


    max_examples_per_bit = 12


    max_draw_per_bit = 3


    data_mode = "all"                                            


    manual_bits = None


def ensure_dirs(cfg: Config):
    os.makedirs(cfg.save_root, exist_ok=True)
    os.makedirs(os.path.join(cfg.save_root, "images"), exist_ok=True)
    os.makedirs(os.path.join(cfg.save_root, "tables"), exist_ok=True)
    os.makedirs(os.path.join(cfg.save_root, "raw_outputs"), exist_ok=True)


def load_data(cfg: Config):
    train_df = pd.read_csv(cfg.train_csv)
    valid_df = pd.read_csv(cfg.valid_csv)
    test_df = pd.read_csv(cfg.test_csv)

    if cfg.data_mode == "train":
        return train_df.copy()
    elif cfg.data_mode == "valid":
        return valid_df.copy()
    elif cfg.data_mode == "test":
        return test_df.copy()
    elif cfg.data_mode == "valid_test":
        return pd.concat([valid_df, test_df], axis=0, ignore_index=True)
    elif cfg.data_mode == "all":
        return pd.concat([train_df, valid_df, test_df], axis=0, ignore_index=True)
    else:
        raise ValueError("data_mode 只能是 train/valid/test/valid_test/all")


def get_target_bits(cfg: Config):
    if cfg.manual_bits is not None and len(cfg.manual_bits) > 0:
        return cfg.manual_bits

    if os.path.exists(cfg.union_matrix_csv):
        union_df = pd.read_csv(cfg.union_matrix_csv)
        bit_cols = [c for c in union_df.columns if c != "label"]
        return bit_cols

    if os.path.exists(cfg.per_label_top_bits_json):
        with open(cfg.per_label_top_bits_json, "r", encoding="utf-8") as f:
            d = json.load(f)
        bits = []
        for _, bit_list in d.items():
            bits.extend(bit_list)

        seen = set()
        out = []
        for b in bits:
            if b not in seen:
                out.append(b)
                seen.add(b)
        return out

    raise FileNotFoundError("未找到 union_matrix_csv 或 per_label_top_bits_json，也没有提供 manual_bits。")


def bit_name_to_index(bit_name: str):
    if not bit_name.startswith("bit_"):
        raise ValueError(f"非法 bit 名称: {bit_name}")
    return int(bit_name.split("_")[1])


def make_generator(cfg: Config):
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=cfg.fp_radius,
        fpSize=cfg.fp_nbits
    )


def get_bit_occurrences(mol, generator):


    additional_output = rdFingerprintGenerator.AdditionalOutput()
    additional_output.AllocateBitInfoMap()

    _ = generator.GetFingerprint(mol, additionalOutput=additional_output)
    bit_info_map = additional_output.GetBitInfoMap()


    bit_occ_map = {}
    for bit_id, envs in bit_info_map.items():
        bit_occ_map[bit_id] = list(envs)

    return bit_occ_map


def get_env_atom_bond_indices(mol, center_atom_idx, radius):


    if radius == 0:
        atom_indices = [center_atom_idx]
        bond_indices = []
        return atom_indices, bond_indices

    bond_indices = list(Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center_atom_idx))
    atom_indices = set([center_atom_idx])

    for bidx in bond_indices:
        bond = mol.GetBondWithIdx(bidx)
        atom_indices.add(bond.GetBeginAtomIdx())
        atom_indices.add(bond.GetEndAtomIdx())

    atom_indices = sorted(atom_indices)
    bond_indices = sorted(bond_indices)
    return atom_indices, bond_indices


def env_to_smarts(mol, atom_indices, bond_indices):


    try:
        submol = Chem.PathToSubmol(mol, bond_indices)
        smarts = Chem.MolToSmarts(submol)
        return smarts
    except Exception:
        return None


def draw_highlighted_mol(mol, atom_indices, bond_indices, save_path, legend=""):
    drawer = rdMolDraw2D.MolDraw2DCairo(600, 450)
    draw_options = drawer.drawOptions()
    draw_options.legendFontSize = 18

    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=list(atom_indices),
        highlightBonds=list(bond_indices),
        legend=legend
    )
    drawer.FinishDrawing()

    with open(save_path, "wb") as f:
        f.write(drawer.GetDrawingText())


def map_bits_to_substructures(cfg: Config):
    ensure_dirs(cfg)

    df = load_data(cfg)
    target_bits = get_target_bits(cfg)
    target_bit_ids = [bit_name_to_index(b) for b in target_bits]

    generator = make_generator(cfg)

    raw_rows = []
    summary_counter = defaultdict(list)

    for idx in range(len(df)):
        smiles = df.iloc[idx][cfg.smiles_col]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        bit_occ_map = get_bit_occurrences(mol, generator)

        for bit_id in target_bit_ids:
            if bit_id not in bit_occ_map:
                continue

            envs = bit_occ_map[bit_id]
            for (center_atom_idx, radius) in envs:
                atom_indices, bond_indices = get_env_atom_bond_indices(mol, center_atom_idx, radius)
                smarts = env_to_smarts(mol, atom_indices, bond_indices)

                raw_rows.append({
                    "bit": f"bit_{bit_id}",
                    "smiles": smiles,
                    "sample_index": idx,
                    "center_atom_idx": center_atom_idx,
                    "radius": radius,
                    "num_atoms_in_env": len(atom_indices),
                    "num_bonds_in_env": len(bond_indices),
                    "env_smarts": smarts
                })

                summary_counter[f"bit_{bit_id}"].append({
                    "smiles": smiles,
                    "sample_index": idx,
                    "center_atom_idx": center_atom_idx,
                    "radius": radius,
                    "atom_indices": atom_indices,
                    "bond_indices": bond_indices,
                    "env_smarts": smarts
                })

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(
        os.path.join(cfg.save_root, "raw_outputs", "bit_occurrence_examples.csv"),
        index=False,
        encoding="utf-8-sig"
    )


    summary_rows = []
    for bit_name, occs in summary_counter.items():
        smarts_list = [x["env_smarts"] for x in occs if x["env_smarts"] is not None]
        smarts_counter = Counter(smarts_list)

        most_common_smarts = None
        most_common_count = 0
        if len(smarts_counter) > 0:
            most_common_smarts, most_common_count = smarts_counter.most_common(1)[0]

        summary_rows.append({
            "bit": bit_name,
            "occurrence_count": len(occs),
            "unique_smarts_count": len(set(smarts_list)) if len(smarts_list) > 0 else 0,
            "most_common_env_smarts": most_common_smarts,
            "most_common_env_count": most_common_count
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["occurrence_count", "bit"], ascending=[False, True]
    )
    summary_df.to_csv(
        os.path.join(cfg.save_root, "tables", "bit_substructure_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )


    for bit_name, occs in summary_counter.items():

        valid_occs = [x for x in occs if x["env_smarts"] is not None]
        if len(valid_occs) == 0:
            valid_occs = occs


        selected = []
        seen_smarts = set()

        for item in valid_occs:
            smarts = item["env_smarts"]
            if smarts not in seen_smarts:
                selected.append(item)
                seen_smarts.add(smarts)
            if len(selected) >= cfg.max_draw_per_bit:
                break

        if len(selected) == 0:
            continue

        for rank, item in enumerate(selected, start=1):
            mol = Chem.MolFromSmiles(item["smiles"])
            if mol is None:
                continue

            legend = f"{bit_name} | r={item['radius']}"
            save_path = os.path.join(
                cfg.save_root, "images",
                f"{bit_name}_example_{rank}.png"
            )

            draw_highlighted_mol(
                mol,
                atom_indices=item["atom_indices"],
                bond_indices=item["bond_indices"],
                save_path=save_path,
                legend=legend
            )


    with open(os.path.join(cfg.save_root, "raw_outputs", "target_bits.json"), "w", encoding="utf-8") as f:
        json.dump(target_bits, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Results saved to: {cfg.save_root}")
    print("Main outputs:")
    print("- raw_outputs/bit_occurrence_examples.csv")
    print("- tables/bit_substructure_summary.csv")
    print("- images/*.png")


if __name__ == "__main__":
    cfg = Config()
    map_bits_to_substructures(cfg)