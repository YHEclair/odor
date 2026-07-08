import os
import random
import logging
import json
import pickle
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
    accuracy_score
)

from xgboost import XGBClassifier


def get_logger(filename):
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if logger.handlers:
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


    return logger


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)


class Config:
    seed = 2026

    train_csv = "data/split/train.csv"
    valid_csv = "data/split/valid.csv"
    test_csv = "data/split/test.csv"

    save_dir = "results/fp_xgb_output"
    smiles_col = "SMILES"
    threshold = 0.5


    fp_radius = 3
    fp_nbits = 2048


    xgb_params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "min_child_weight": 1,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "random_state": 2026,
        "n_jobs": -1
    }


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
            raise ValueError(f"Invalid SMILES: {smiles}")
        fp = featurizer(mol)
        features.append(fp)

    return np.vstack(features).astype(np.float32)


def prepare_datasets(train_df, valid_df, test_df, smiles_col, label_cols, cfg, logger):
    featurizer = MorganFingerprintFeaturizer(
        radius=cfg.fp_radius,
        n_bits=cfg.fp_nbits
    )

    X_train = build_fp_matrix(train_df, smiles_col, featurizer)
    X_valid = build_fp_matrix(valid_df, smiles_col, featurizer)
    X_test = build_fp_matrix(test_df, smiles_col, featurizer)

    y_train = train_df[label_cols].values.astype(np.int32)
    y_valid = valid_df[label_cols].values.astype(np.int32)
    y_test = test_df[label_cols].values.astype(np.int32)

    logger.info(f"Fingerprint type: MorganGenerator(radius={cfg.fp_radius}, nBits={cfg.fp_nbits})")
    logger.info(f"Train fingerprint matrix shape: {X_train.shape}")
    logger.info(f"Valid fingerprint matrix shape: {X_valid.shape}")
    logger.info(f"Test fingerprint matrix shape : {X_test.shape}")

    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test
    }


def binary_metrics_per_label(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    n_labels = y_true.shape[1]
    per_label = []

    aucs, accs, f1s, mccs, kappas, sns, sps = [], [], [], [], [], [], []

    for i in range(n_labels):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        yp_prob = y_prob[:, i]

        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        acc = accuracy_score(yt, yp)
        f1 = f1_score(yt, yp, zero_division=0)

        try:
            mcc = matthews_corrcoef(yt, yp)
        except Exception:
            mcc = np.nan

        try:
            ck = cohen_kappa_score(yt, yp)
        except Exception:
            ck = np.nan

        sn = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        sp = tn / (tn + fp) if (tn + fp) > 0 else np.nan

        if len(np.unique(yt)) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(yt, yp_prob)

        per_label.append({
            "label_idx": i,
            "acc": acc,
            "f1": f1,
            "mcc": mcc,
            "ck": ck,
            "auc": auc,
            "sn": sn,
            "sp": sp,
            "positive_count": int(yt.sum())
        })

        accs.append(acc)
        f1s.append(f1)
        mccs.append(mcc)
        kappas.append(ck)
        aucs.append(auc)
        sns.append(sn)
        sps.append(sp)

    exact_match_acc = np.mean(np.all(y_true == y_pred, axis=1))

    summary = {
        "exact_match_acc": exact_match_acc,
        "acc_macro": np.nanmean(accs),
        "f1_macro": np.nanmean(f1s),
        "mcc_macro": np.nanmean(mccs),
        "ck_macro": np.nanmean(kappas),
        "auc_macro": np.nanmean(aucs),
        "sn_macro": np.nanmean(sns),
        "sp_macro": np.nanmean(sps),
    }

    return summary, per_label


def train_single_label_xgb(X_train, y_train, X_valid, y_valid, params):
    model = XGBClassifier(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )

    return model


def train_all_labels(X_train, y_train, X_valid, y_valid, label_cols, cfg, logger):
    models = []
    valid_probs = []

    models_dir = os.path.join(cfg.save_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    for i, label_name in enumerate(label_cols):
        logger.info(f"========== Training label {i}/{len(label_cols)-1}: {label_name} ==========")

        yi_train = y_train[:, i]
        yi_valid = y_valid[:, i]

        model = train_single_label_xgb(
            X_train=X_train,
            y_train=yi_train,
            X_valid=X_valid,
            y_valid=yi_valid,
            params=cfg.xgb_params
        )

        prob_valid = model.predict_proba(X_valid)[:, 1]
        valid_probs.append(prob_valid)

        with open(os.path.join(models_dir, f"xgb_{i}_{label_name}.pkl"), "wb") as f:
            pickle.dump(model, f)

        if len(np.unique(yi_valid)) >= 2:
            auc_i = roc_auc_score(yi_valid, prob_valid)
        else:
            auc_i = np.nan

        logger.info(
            f"[Label {i:02d} | {label_name}] "
            f"Valid AUC={auc_i:.4f} | Positive train={int(yi_train.sum())} | Positive valid={int(yi_valid.sum())}"
        )

        models.append(model)

    valid_probs = np.vstack(valid_probs).T
    return models, valid_probs


def predict_all_labels(models, X):
    probs = []
    for model in models:
        prob = model.predict_proba(X)[:, 1]
        probs.append(prob)
    return np.vstack(probs).T


def main():
    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)

    logger = get_logger("train")
    seed_everything(cfg.seed)

    logger.info("Using model: FP-XGBoost")
    logger.info(f"XGBoost params: {json.dumps(cfg.xgb_params, ensure_ascii=False)}")

    train_df = pd.read_csv(cfg.train_csv)
    valid_df = pd.read_csv(cfg.valid_csv)
    test_df = pd.read_csv(cfg.test_csv)

    label_cols = [c for c in train_df.columns if c != cfg.smiles_col]

    logger.info(f"Train size: {len(train_df)}")
    logger.info(f"Valid size: {len(valid_df)}")
    logger.info(f"Test size : {len(test_df)}")
    logger.info(f"Num labels: {len(label_cols)}")
    logger.info(f"Label cols: {label_cols}")

    data_dict = prepare_datasets(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        smiles_col=cfg.smiles_col,
        label_cols=label_cols,
        cfg=cfg,
        logger=logger
    )

    X_train = data_dict["X_train"]
    X_valid = data_dict["X_valid"]
    X_test = data_dict["X_test"]
    y_train = data_dict["y_train"]
    y_valid = data_dict["y_valid"]
    y_test = data_dict["y_test"]

    models, valid_probs = train_all_labels(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        label_cols=label_cols,
        cfg=cfg,
        logger=logger
    )

    valid_summary, valid_per_label = binary_metrics_per_label(
        y_true=y_valid,
        y_prob=valid_probs,
        threshold=cfg.threshold
    )

    logger.info("========== VALID RESULTS ==========")
    logger.info(
        f"ACC={valid_summary['acc_macro']:.4f} | "
        f"F1={valid_summary['f1_macro']:.4f} | "
        f"MCC={valid_summary['mcc_macro']:.4f} | "
        f"CK={valid_summary['ck_macro']:.4f} | "
        f"AUC={valid_summary['auc_macro']:.4f} | "
        f"Sn={valid_summary['sn_macro']:.4f} | "
        f"Sp={valid_summary['sp_macro']:.4f} | "
        f"ExactAcc={valid_summary['exact_match_acc']:.4f}"
    )

    test_probs = predict_all_labels(models, X_test)
    test_summary, test_per_label = binary_metrics_per_label(
        y_true=y_test,
        y_prob=test_probs,
        threshold=cfg.threshold
    )

    logger.info("========== TEST RESULTS ==========")
    logger.info(
        f"ACC={test_summary['acc_macro']:.4f} | "
        f"F1={test_summary['f1_macro']:.4f} | "
        f"MCC={test_summary['mcc_macro']:.4f} | "
        f"CK={test_summary['ck_macro']:.4f} | "
        f"AUC={test_summary['auc_macro']:.4f} | "
        f"Sn={test_summary['sn_macro']:.4f} | "
        f"Sp={test_summary['sp_macro']:.4f} | "
        f"ExactAcc={test_summary['exact_match_acc']:.4f}"
    )


    per_label_df = pd.DataFrame(test_per_label)
    per_label_df["label_name"] = label_cols
    per_label_df.to_csv(
        os.path.join(cfg.save_dir, "test_per_label_metrics.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    logger.info("Saved per-label test metrics: test_per_label_metrics.csv")


    np.save(os.path.join(cfg.save_dir, "valid_probs.npy"), valid_probs)
    np.save(os.path.join(cfg.save_dir, "test_probs.npy"), test_probs)
    np.save(os.path.join(cfg.save_dir, "valid_targets.npy"), y_valid)
    np.save(os.path.join(cfg.save_dir, "test_targets.npy"), y_test)
    logger.info("Saved valid/test probabilities and targets: *.npy")


    meta = {
        "fingerprint_type": "MorganGenerator",
        "fp_radius": cfg.fp_radius,
        "fp_nbits": cfg.fp_nbits,
        "label_cols": label_cols,
        "xgb_params": cfg.xgb_params
    }
    with open(os.path.join(cfg.save_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info("Saved metadata: meta.json")


if __name__ == "__main__":
    main()
