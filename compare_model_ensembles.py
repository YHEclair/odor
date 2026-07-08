import os
import json
import logging
import random
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
    accuracy_score
)


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
    threshold = 0.5
    save_dir = "results/pairwise_simple_avg_output"


    gnn_valid_probs = "results/gnn_output/gnn_valid_probs.npy"
    gnn_test_probs = "results/gnn_output/gnn_test_probs.npy"
    gnn_valid_targets = "results/gnn_output/gnn_valid_targets.npy"
    gnn_test_targets = "results/gnn_output/gnn_test_targets.npy"


    fp_valid_probs = "results/fp_xgb_output/valid_probs.npy"
    fp_test_probs = "results/fp_xgb_output/test_probs.npy"
    fp_valid_targets = "results/fp_xgb_output/valid_targets.npy"
    fp_test_targets = "results/fp_xgb_output/test_targets.npy"


    desc_valid_probs = "results/desc_xgb_output/valid_probs.npy"
    desc_test_probs = "results/desc_xgb_output/test_probs.npy"
    desc_valid_targets = "results/desc_xgb_output/valid_targets.npy"
    desc_test_targets = "results/desc_xgb_output/test_targets.npy"

    train_csv = "data/split/train.csv"
    smiles_col = "SMILES"


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


def load_branch_arrays(cfg, logger):
    gnn_valid_probs = np.load(cfg.gnn_valid_probs)
    gnn_test_probs = np.load(cfg.gnn_test_probs)

    fp_valid_probs = np.load(cfg.fp_valid_probs)
    fp_test_probs = np.load(cfg.fp_test_probs)

    desc_valid_probs = np.load(cfg.desc_valid_probs)
    desc_test_probs = np.load(cfg.desc_test_probs)

    gnn_valid_targets = np.load(cfg.gnn_valid_targets)
    gnn_test_targets = np.load(cfg.gnn_test_targets)

    fp_valid_targets = np.load(cfg.fp_valid_targets)
    fp_test_targets = np.load(cfg.fp_test_targets)

    desc_valid_targets = np.load(cfg.desc_valid_targets)
    desc_test_targets = np.load(cfg.desc_test_targets)

    if not np.array_equal(gnn_valid_targets, fp_valid_targets) or not np.array_equal(gnn_valid_targets, desc_valid_targets):
        raise ValueError("valid_targets are inconsistent across branches. Please check the files.")

    if not np.array_equal(gnn_test_targets, fp_test_targets) or not np.array_equal(gnn_test_targets, desc_test_targets):
        raise ValueError("test_targets are inconsistent across branches. Please check the files.")

    logger.info(f"GNN valid probs shape : {gnn_valid_probs.shape}")
    logger.info(f"FP valid probs shape  : {fp_valid_probs.shape}")
    logger.info(f"DESC valid probs shape: {desc_valid_probs.shape}")

    logger.info(f"GNN test probs shape  : {gnn_test_probs.shape}")
    logger.info(f"FP test probs shape   : {fp_test_probs.shape}")
    logger.info(f"DESC test probs shape : {desc_test_probs.shape}")

    return {
        "gnn_valid_probs": gnn_valid_probs,
        "gnn_test_probs": gnn_test_probs,
        "fp_valid_probs": fp_valid_probs,
        "fp_test_probs": fp_test_probs,
        "desc_valid_probs": desc_valid_probs,
        "desc_test_probs": desc_test_probs,
        "valid_targets": gnn_valid_targets,
        "test_targets": gnn_test_targets,
    }


def build_fusions(arrays):
    fusion_dict = {
        "gnn_fp": {
            "valid_probs": (arrays["gnn_valid_probs"] + arrays["fp_valid_probs"]) / 2.0,
            "test_probs": (arrays["gnn_test_probs"] + arrays["fp_test_probs"]) / 2.0,
        },
        "gnn_desc": {
            "valid_probs": (arrays["gnn_valid_probs"] + arrays["desc_valid_probs"]) / 2.0,
            "test_probs": (arrays["gnn_test_probs"] + arrays["desc_test_probs"]) / 2.0,
        },
        "fp_desc": {
            "valid_probs": (arrays["fp_valid_probs"] + arrays["desc_valid_probs"]) / 2.0,
            "test_probs": (arrays["fp_test_probs"] + arrays["desc_test_probs"]) / 2.0,
        },
        "gnn_fp_desc": {
            "valid_probs": (
                arrays["gnn_valid_probs"] +
                arrays["fp_valid_probs"] +
                arrays["desc_valid_probs"]
            ) / 3.0,
            "test_probs": (
                arrays["gnn_test_probs"] +
                arrays["fp_test_probs"] +
                arrays["desc_test_probs"]
            ) / 3.0,
        }
    }
    return fusion_dict


def main():
    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)

    logger = get_logger("pairwise_simple_avg")
    seed_everything(cfg.seed)

    logger.info("Using pairwise + three-way simple average fusion")

    arrays = load_branch_arrays(cfg, logger)

    train_df = pd.read_csv(cfg.train_csv)
    label_cols = [c for c in train_df.columns if c != cfg.smiles_col]

    y_valid = arrays["valid_targets"]
    y_test = arrays["test_targets"]

    fusion_dict = build_fusions(arrays)

    valid_rows = []
    test_rows = []

    for fusion_name, fusion_data in fusion_dict.items():
        logger.info(f"========== {fusion_name.upper()} ==========")

        valid_probs = fusion_data["valid_probs"]
        test_probs = fusion_data["test_probs"]

        valid_summary, _ = binary_metrics_per_label(
            y_true=y_valid,
            y_prob=valid_probs,
            threshold=cfg.threshold
        )

        test_summary, test_per_label = binary_metrics_per_label(
            y_true=y_test,
            y_prob=test_probs,
            threshold=cfg.threshold
        )

        logger.info("VALID RESULTS")
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

        logger.info("TEST RESULTS")
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


        np.save(os.path.join(cfg.save_dir, f"{fusion_name}_valid_probs.npy"), valid_probs)
        np.save(os.path.join(cfg.save_dir, f"{fusion_name}_test_probs.npy"), test_probs)


        pd.DataFrame(test_per_label).assign(label_name=label_cols).to_csv(
            os.path.join(cfg.save_dir, f"{fusion_name}_test_per_label_metrics.csv"),
            index=False,
            encoding="utf-8-sig"
        )


        valid_rows.append({
            "fusion_name": fusion_name,
            "valid_acc": valid_summary["acc_macro"],
            "valid_f1": valid_summary["f1_macro"],
            "valid_mcc": valid_summary["mcc_macro"],
            "valid_ck": valid_summary["ck_macro"],
            "valid_auc": valid_summary["auc_macro"],
            "valid_sn": valid_summary["sn_macro"],
            "valid_sp": valid_summary["sp_macro"],
            "valid_exactacc": valid_summary["exact_match_acc"],
        })


        test_rows.append({
            "fusion_name": fusion_name,
            "test_acc": test_summary["acc_macro"],
            "test_f1": test_summary["f1_macro"],
            "test_mcc": test_summary["mcc_macro"],
            "test_ck": test_summary["ck_macro"],
            "test_auc": test_summary["auc_macro"],
            "test_sn": test_summary["sn_macro"],
            "test_sp": test_summary["sp_macro"],
            "test_exactacc": test_summary["exact_match_acc"],
        })


    valid_df = pd.DataFrame(valid_rows)
    test_df = pd.DataFrame(test_rows)
    summary_df = pd.merge(valid_df, test_df, on="fusion_name", how="inner")

    summary_df.to_csv(
        os.path.join(cfg.save_dir, "pairwise_and_threeway_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )


    np.save(os.path.join(cfg.save_dir, "valid_targets.npy"), y_valid)
    np.save(os.path.join(cfg.save_dir, "test_targets.npy"), y_test)


    meta = {
        "fusion_method": "simple_average",
        "fusions": list(fusion_dict.keys()),
        "gnn_valid_probs": cfg.gnn_valid_probs,
        "gnn_test_probs": cfg.gnn_test_probs,
        "fp_valid_probs": cfg.fp_valid_probs,
        "fp_test_probs": cfg.fp_test_probs,
        "desc_valid_probs": cfg.desc_valid_probs,
        "desc_test_probs": cfg.desc_test_probs,
        "label_cols": label_cols
    }

    with open(os.path.join(cfg.save_dir, "pairwise_and_threeway_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info("Saved summary results: pairwise_and_threeway_summary.csv")
    logger.info("Saved metadata: pairwise_and_threeway_meta.json")


if __name__ == "__main__":
    main()
