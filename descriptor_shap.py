import os
import json
import random
import pickle
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb

from rdkit import Chem
from rdkit.Chem import Descriptors

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


    model_root = "./results/desc_xgb_output"
    model_dir = "./results/desc_xgb_output/models"


    save_root = "./results/interpretability_analysis/01_descriptor_shap"


    data_mode = "all"


    use_sample = False
    sample_size = 1000
    random_state = 2026


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


class AllRDKitDescriptorFeaturizer:
    def __init__(self):
        self.desc_list = Descriptors._descList
        self.feature_names = [name for name, func in self.desc_list]

    def __call__(self, mol):
        values = []
        for name, func in self.desc_list:
            try:
                v = func(mol)
            except Exception:
                v = np.nan
            values.append(v)
        return np.array(values, dtype=np.float32)

    def get_feature_names(self):
        return self.feature_names


def build_descriptor_matrix(df: pd.DataFrame, smiles_col: str, featurizer):
    raw_features = []

    for i in range(len(df)):
        smiles = df.iloc[i][smiles_col]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"无效 SMILES: {smiles}")

        desc = featurizer(mol)
        raw_features.append(desc)

    return np.vstack(raw_features).astype(np.float32)


def prepare_descriptor_data(cfg: Config):
    train_df = pd.read_csv(cfg.train_csv)
    valid_df = pd.read_csv(cfg.valid_csv)
    test_df = pd.read_csv(cfg.test_csv)

    featurizer = AllRDKitDescriptorFeaturizer()
    raw_desc_names = featurizer.get_feature_names()

    X_train_raw = build_descriptor_matrix(train_df, cfg.smiles_col, featurizer)
    X_valid_raw = build_descriptor_matrix(valid_df, cfg.smiles_col, featurizer)
    X_test_raw = build_descriptor_matrix(test_df, cfg.smiles_col, featurizer)


    valid_mask = np.all(np.isfinite(X_train_raw), axis=0)

    X_train = X_train_raw[:, valid_mask]
    X_valid = X_valid_raw[:, valid_mask]
    X_test = X_test_raw[:, valid_mask]


    X_train = np.where(np.isfinite(X_train), X_train, 0.0).astype(np.float32)
    X_valid = np.where(np.isfinite(X_valid), X_valid, 0.0).astype(np.float32)
    X_test = np.where(np.isfinite(X_test), X_test, 0.0).astype(np.float32)

    used_feature_names = [
        name for name, keep in zip(raw_desc_names, valid_mask) if keep
    ]

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
        "raw_desc_names": raw_desc_names,
        "used_feature_names": used_feature_names,
        "valid_mask": valid_mask
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
    fig_dir = os.path.join(cfg.save_root, "figures")
    table_dir = os.path.join(cfg.save_root, "tables")

    print("Preparing descriptor data...")
    data_dict = prepare_descriptor_data(cfg)

    X = data_dict["X"]
    df_used = data_dict["df_used"]
    raw_desc_names = data_dict["raw_desc_names"]
    used_feature_names = data_dict["used_feature_names"]
    valid_mask = data_dict["valid_mask"]

    print(f"Data mode: {cfg.data_mode}")
    print(f"Raw descriptor count: {len(raw_desc_names)}")
    print(f"Used descriptor count: {len(used_feature_names)}")
    print(f"Num samples before sampling: {len(df_used)}")

    X, df_used = maybe_subsample(X, df_used, cfg)
    print(f"Num samples used for SHAP: {len(df_used)}")

    X_df = pd.DataFrame(X, columns=used_feature_names)

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
            "descriptor": used_feature_names,
            "mean_abs_shap": mean_abs_shap
        }).sort_values("mean_abs_shap", ascending=False)

        per_label_rows.append(label_df)

    if len(per_label_rows) == 0:
        raise RuntimeError("没有成功加载任何标签模型，请检查 model_dir 或模型命名。")


    per_label_importance_df = pd.concat(per_label_rows, axis=0, ignore_index=True)
    per_label_importance_df.to_csv(
        os.path.join(raw_dir, "01_per_label_descriptor_shap_importance.csv"),
        index=False,
        encoding="utf-8-sig"
    )


    global_importance_df = (
        per_label_importance_df
        .groupby("descriptor", as_index=False)["mean_abs_shap"]
        .mean()
        .rename(columns={"mean_abs_shap": "global_mean_abs_shap"})
        .sort_values("global_mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    global_importance_df.to_csv(
        os.path.join(raw_dir, "02_global_descriptor_shap_importance.csv"),
        index=False,
        encoding="utf-8-sig"
    )


    top20_df = global_importance_df.head(20).copy()
    top20_df.to_csv(
        os.path.join(raw_dir, "03_top20_global_descriptor_shap_importance.csv"),
        index=False,
        encoding="utf-8-sig"
    )


    top20_df.to_csv(
        os.path.join(table_dir, "table_top20_global_descriptor_shap_importance.csv"),
        index=False,
        encoding="utf-8-sig"
    )


    pd.DataFrame({"descriptor": used_feature_names}).to_csv(
        os.path.join(raw_dir, "04_used_descriptor_columns.csv"),
        index=False,
        encoding="utf-8-sig"
    )


    df_used[[cfg.smiles_col] + cfg.label_cols].to_csv(
        os.path.join(raw_dir, "05_shap_input_samples_and_labels.csv"),
        index=False,
        encoding="utf-8-sig"
    )


    meta = {
        "data_mode": cfg.data_mode,
        "use_sample": cfg.use_sample,
        "sample_size_used": int(len(df_used)),
        "raw_descriptor_count": len(raw_desc_names),
        "used_descriptor_count": len(used_feature_names),
        "used_feature_names": used_feature_names,
        "valid_mask": valid_mask.tolist(),
        "label_cols": cfg.label_cols,
        "model_dir": cfg.model_dir,
        "notes": {
            "cleaning_rule": "Only columns finite in training set were kept; remaining invalid values in valid/test/all were replaced with 0.0",
            "scaling": "No scaling",
            "constant_columns": "Retained",
            "shap_definition": "Global importance = average over labels of mean absolute SHAP per descriptor",
            "shap_backend": "XGBoost booster.predict(pred_contribs=True)"
        }
    }
    with open(os.path.join(raw_dir, "06_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"All outputs saved under: {cfg.save_root}")
    print("Created folders:")
    print(f"- {raw_dir}")
    print(f"- {fig_dir}")
    print(f"- {table_dir}")
    print("Main files:")
    print("- raw_outputs/01_per_label_descriptor_shap_importance.csv")
    print("- raw_outputs/02_global_descriptor_shap_importance.csv")
    print("- raw_outputs/03_top20_global_descriptor_shap_importance.csv")
    print("- tables/table_top20_global_descriptor_shap_importance.csv")


if __name__ == "__main__":
    main()