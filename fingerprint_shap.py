import os
import json
import random
import pickle
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

warnings.filterwarnings("ignore")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)


class Config:
    seed = 2026

    train_csv = "./data/split/train.csv"
    valid_csv = "./data/split/valid.csv"
    test_csv = "./data/split/test.csv"

    smiles_col = "SMILES"


    model_root = "./results/fp_xgb_output"
    model_dir = "./results/fp_xgb_output/models"


    save_root = "./results/interpretability_analysis/02_fingerprint_shap"


    data_mode = "all"


    use_sample = False
    sample_size = 1000
    random_state = 2026


    fp_radius = 3
    fp_nbits = 2048

    label_cols = [
        "fruity",
        "green",
        "sweet",
        "floral",
        "herbal",
        "woody",
        "fatty",
        "fresh",
        "waxy",
        "spicy",
        "sulfurous",
        "oily",
        "citrus",
        "nutty",
        "rose",
        "earthy",
        "apple",
        "roasted",
    ]


def ensure_dirs(cfg: Config):
    os.makedirs(cfg.save_root, exist_ok=True)
    os.makedirs(os.path.join(cfg.save_root, "raw_outputs"), exist_ok=True)
    os.makedirs(os.path.join(cfg.save_root, "figures"), exist_ok=True)
    os.makedirs(os.path.join(cfg.save_root, "tables"), exist_ok=True)


class MorganFingerprintFeaturizer:
    def __init__(self, radius=3, n_bits=2048):
        self.radius = radius
        self.n_bits = n_bits
        self.generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius,
            fpSize=self.n_bits
        )

    def __call__(self, mol):
        fp = self.generator.GetFingerprint(mol)
        arr = np.zeros((self.n_bits,), dtype=np.float32)
        for idx in fp.GetOnBits():
            arr[idx] = 1.0
        return arr


def build_fp_matrix(df, smiles_col, featurizer):
    features = []

    for i in range(len(df)):
        smiles = df.iloc[i][smiles_col]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"无效 SMILES: {smiles}")
        fp = featurizer(mol)
        features.append(fp)

    return np.vstack(features).astype(np.float32)


def prepare_fingerprint_data(cfg: Config):
    train_df = pd.read_csv(cfg.train_csv)
    valid_df = pd.read_csv(cfg.valid_csv)
    test_df = pd.read_csv(cfg.test_csv)

    featurizer = MorganFingerprintFeaturizer(
        radius=cfg.fp_radius,
        n_bits=cfg.fp_nbits
    )

    X_train = build_fp_matrix(train_df, cfg.smiles_col, featurizer)
    X_valid = build_fp_matrix(valid_df, cfg.smiles_col, featurizer)
    X_test = build_fp_matrix(test_df, cfg.smiles_col, featurizer)

    bit_names = [f"bit_{i}" for i in range(cfg.fp_nbits)]

    if cfg.data_mode == "train":
        X = X_train
        df_used = train_df.copy()
    elif cfg.data_mode == "valid":
        X = X_valid
        df_used = valid_df.copy()
    elif cfg.data_mode == "test":
        X = X_test
        df_used = test_df.copy()
    elif cfg.data_mode == "valid_test":
        X = np.vstack([X_valid, X_test])
        df_used = pd.concat([valid_df, test_df], axis=0, ignore_index=True)
    elif cfg.data_mode == "all":
        X = np.vstack([X_train, X_valid, X_test])
        df_used = pd.concat([train_df, valid_df, test_df], axis=0, ignore_index=True)
    else:
        raise ValueError("data_mode 只能是 train/valid/test/valid_test/all")

    return {
        "X": X,
        "df_used": df_used,
        "bit_names": bit_names
    }


def maybe_subsample(X: np.ndarray, df_used: pd.DataFrame, cfg: Config):
    if not cfg.use_sample:
        return X, df_used

    if len(df_used) <= cfg.sample_size:
        return X, df_used

    rng = np.random.RandomState(cfg.random_state)
    idx = rng.choice(len(df_used), size=cfg.sample_size, replace=False)
    idx = np.sort(idx)

    X_sub = X[idx]
    df_sub = df_used.iloc[idx].reset_index(drop=True)
    return X_sub, df_sub


def load_xgb_model(model_path: str):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def compute_shap_for_one_model(model, X_df: pd.DataFrame):
    booster = model.get_booster()
    dmatrix = xgb.DMatrix(X_df)

    shap_values = booster.predict(dmatrix, pred_contribs=True)
    shap_values = np.array(shap_values)

    if shap_values.ndim != 2:
        raise ValueError(f"SHAP 输出维度异常: {shap_values.shape}")


    shap_values = shap_values[:, :-1]

    if shap_values.shape[1] != X_df.shape[1]:
        raise ValueError(
            f"SHAP 特征维度不一致: shap={shap_values.shape}, X={X_df.shape}"
        )

    return shap_values


def main():
    cfg = Config()
    ensure_dirs(cfg)
    seed_everything(cfg.seed)

    raw_dir = os.path.join(cfg.save_root, "raw_outputs")
    table_dir = os.path.join(cfg.save_root, "tables")

    print("Preparing fingerprint data...")
    data_dict = prepare_fingerprint_data(cfg)

    X = data_dict["X"]
    df_used = data_dict["df_used"]
    bit_names = data_dict["bit_names"]

    print(f"Data mode: {cfg.data_mode}")
    print(f"Fingerprint radius: {cfg.fp_radius}")
    print(f"Fingerprint bits: {cfg.fp_nbits}")
    print(f"Num samples before sampling: {len(df_used)}")

    X, df_used = maybe_subsample(X, df_used, cfg)
    print(f"Num samples used for SHAP: {len(df_used)}")

    X_df = pd.DataFrame(X, columns=bit_names)

    per_label_rows = []

    print("Computing SHAP importance for each label model...")
    for i, label_name in enumerate(cfg.label_cols):
        model_path = os.path.join(cfg.model_dir, f"xgb_{i}_{label_name}.pkl")

        if not os.path.exists(model_path):
            print(f"[Skip] 未找到模型文件: {model_path}")
            continue

        print(f"Processing label {i:02d}: {label_name}")
        model = load_xgb_model(model_path)

        shap_values = compute_shap_for_one_model(model, X_df)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        label_df = pd.DataFrame({
            "label": label_name,
            "bit": bit_names,
            "mean_abs_shap": mean_abs_shap
        }).sort_values("mean_abs_shap", ascending=False)

        per_label_rows.append(label_df)

    if len(per_label_rows) == 0:
        raise RuntimeError("没有成功加载任何标签模型，请检查 model_dir 或模型命名。")


    per_label_importance_df = pd.concat(per_label_rows, axis=0, ignore_index=True)
    per_label_importance_df.to_csv(
        os.path.join(raw_dir, "01_per_label_fingerprint_shap_importance.csv"),
        index=False,
        encoding="utf-8-sig"
    )


    global_importance_df = (
        per_label_importance_df
        .groupby("bit", as_index=False)["mean_abs_shap"]
        .mean()
        .rename(columns={"mean_abs_shap": "global_mean_abs_shap"})
        .sort_values("global_mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    global_importance_df.to_csv(
        os.path.join(raw_dir, "02_global_fingerprint_shap_importance.csv"),
        index=False,
        encoding="utf-8-sig"
    )


    top20_df = global_importance_df.head(20).copy()
    top20_df.to_csv(
        os.path.join(raw_dir, "03_top20_global_fingerprint_shap_importance.csv"),
        index=False,
        encoding="utf-8-sig"
    )


    top20_df.to_csv(
        os.path.join(table_dir, "table_top20_global_fingerprint_shap_importance.csv"),
        index=False,
        encoding="utf-8-sig"
    )


    df_used[[cfg.smiles_col] + cfg.label_cols].to_csv(
        os.path.join(raw_dir, "04_shap_input_samples_and_labels.csv"),
        index=False,
        encoding="utf-8-sig"
    )


    meta = {
        "data_mode": cfg.data_mode,
        "use_sample": cfg.use_sample,
        "sample_size_used": int(len(df_used)),
        "fingerprint_type": "MorganGenerator",
        "fp_radius": cfg.fp_radius,
        "fp_nbits": cfg.fp_nbits,
        "bit_names": bit_names,
        "label_cols": cfg.label_cols,
        "model_dir": cfg.model_dir,
        "notes": {
            "shap_definition": "Global importance = average over labels of mean absolute SHAP per fingerprint bit",
            "shap_backend": "XGBoost booster.predict(pred_contribs=True)"
        }
    }
    with open(os.path.join(raw_dir, "05_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"All outputs saved under: {cfg.save_root}")
    print("Main files:")
    print("- raw_outputs/01_per_label_fingerprint_shap_importance.csv")
    print("- raw_outputs/02_global_fingerprint_shap_importance.csv")
    print("- raw_outputs/03_top20_global_fingerprint_shap_importance.csv")
    print("- tables/table_top20_global_fingerprint_shap_importance.csv")


if __name__ == "__main__":
    main()