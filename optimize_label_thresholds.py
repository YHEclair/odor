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
    save_dir = "results/simple_avg_threshold_optimized_output"


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


    default_threshold = 0.5


    threshold_min = 0.10
    threshold_max = 0.90
    threshold_step = 0.01


    optimize_metric = "f1"


def calc_binary_stats(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except Exception:
        mcc = np.nan

    try:
        ck = cohen_kappa_score(y_true, y_pred)
    except Exception:
        ck = np.nan

    sn = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    sp = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    if len(np.unique(y_true)) < 2:
        auc = np.nan
    else:
        auc = roc_auc_score(y_true, y_prob)

    return {
        "acc": acc,
        "f1": f1,
        "mcc": mcc,
        "ck": ck,
        "auc": auc,
        "sn": sn,
        "sp": sp,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }


def binary_metrics_per_label(y_true, y_prob, threshold=0.5):
    n_labels = y_true.shape[1]
    per_label = []

    aucs, accs, f1s, mccs, kappas, sns, sps = [], [], [], [], [], [], []

    if isinstance(threshold, (list, tuple, np.ndarray)):
        thresholds = np.array(threshold, dtype=float)
        assert len(thresholds) == n_labels, "Per-label threshold length must match the number of labels"
    else:
        thresholds = np.array([float(threshold)] * n_labels)

    y_pred_all = np.zeros_like(y_true, dtype=int)

    for i in range(n_labels):
        stats = calc_binary_stats(
            y_true=y_true[:, i],
            y_prob=y_prob[:, i],
            threshold=thresholds[i]
        )

        y_pred_all[:, i] = (y_prob[:, i] >= thresholds[i]).astype(int)

        per_label.append({
            "label_idx": i,
            "threshold": thresholds[i],
            "acc": stats["acc"],
            "f1": stats["f1"],
            "mcc": stats["mcc"],
            "ck": stats["ck"],
            "auc": stats["auc"],
            "sn": stats["sn"],
            "sp": stats["sp"],
            "positive_count": int(y_true[:, i].sum())
        })

        accs.append(stats["acc"])
        f1s.append(stats["f1"])
        mccs.append(stats["mcc"])
        kappas.append(stats["ck"])
        aucs.append(stats["auc"])
        sns.append(stats["sn"])
        sps.append(stats["sp"])

    exact_match_acc = np.mean(np.all(y_true == y_pred_all, axis=1))

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


def get_metric_value(stats, metric_name):
    if metric_name == "f1":
        return stats["f1"]
    elif metric_name == "mcc":
        return stats["mcc"]
    elif metric_name == "sn":
        return stats["sn"]
    else:
        raise ValueError(f"Unsupported optimization metric: {metric_name}")


def optimize_thresholds_per_label(y_valid, prob_valid, cfg, logger):
    n_labels = y_valid.shape[1]
    thresholds = np.arange(
        cfg.threshold_min,
        cfg.threshold_max + 1e-12,
        cfg.threshold_step
    )

    best_thresholds = []
    best_records = []

    for i in range(n_labels):
        yt = y_valid[:, i]
        yp = prob_valid[:, i]

        best_thr = cfg.default_threshold
        best_score = -np.inf
        best_stats = None

        for thr in thresholds:
            stats = calc_binary_stats(yt, yp, threshold=thr)
            score = get_metric_value(stats, cfg.optimize_metric)


            if np.isnan(score):
                continue

            if score > best_score:
                best_score = score
                best_thr = float(thr)
                best_stats = stats


        if best_stats is None:
            best_thr = cfg.default_threshold
            best_stats = calc_binary_stats(yt, yp, threshold=best_thr)
            best_score = get_metric_value(best_stats, cfg.optimize_metric)

        best_thresholds.append(best_thr)
        best_records.append({
            "label_idx": i,
            "best_threshold": best_thr,
            "optimize_metric": cfg.optimize_metric,
            "best_metric_value": best_score,
            "valid_auc": best_stats["auc"],
            "valid_f1": best_stats["f1"],
            "valid_mcc": best_stats["mcc"],
            "valid_sn": best_stats["sn"],
            "valid_sp": best_stats["sp"],
            "positive_count": int(yt.sum())
        })

        logger.info(
            f"[Label {i:02d}] best_threshold={best_thr:.2f} | "
            f"{cfg.optimize_metric}={best_score:.4f} | "
            f"AUC={best_stats['auc']:.4f} | "
            f"F1={best_stats['f1']:.4f} | "
            f"Sn={best_stats['sn']:.4f} | "
            f"Sp={best_stats['sp']:.4f}"
        )

    return np.array(best_thresholds, dtype=float), pd.DataFrame(best_records)


def main():
    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)

    logger = get_logger("threshold_optimized_simple_avg")
    seed_everything(cfg.seed)

    logger.info("Using fusion method: simple average + label-wise threshold optimization")
    logger.info(f"Optimize metric: {cfg.optimize_metric}")
    logger.info(f"Threshold search range: [{cfg.threshold_min}, {cfg.threshold_max}] step={cfg.threshold_step}")

    arrays = load_branch_arrays(cfg, logger)

    train_df = pd.read_csv(cfg.train_csv)
    label_cols = [c for c in train_df.columns if c != cfg.smiles_col]

    y_valid = arrays["valid_targets"]
    y_test = arrays["test_targets"]


    simple_avg_valid = (
        arrays["gnn_valid_probs"] +
        arrays["fp_valid_probs"] +
        arrays["desc_valid_probs"]
    ) / 3.0

    simple_avg_test = (
        arrays["gnn_test_probs"] +
        arrays["fp_test_probs"] +
        arrays["desc_test_probs"]
    ) / 3.0


    default_valid_summary, _ = binary_metrics_per_label(
        y_true=y_valid,
        y_prob=simple_avg_valid,
        threshold=cfg.default_threshold
    )
    default_test_summary, default_test_per_label = binary_metrics_per_label(
        y_true=y_test,
        y_prob=simple_avg_test,
        threshold=cfg.default_threshold
    )

    logger.info("========== DEFAULT THRESHOLD VALID RESULTS ==========")
    logger.info(
        f"ACC={default_valid_summary['acc_macro']:.4f} | "
        f"F1={default_valid_summary['f1_macro']:.4f} | "
        f"MCC={default_valid_summary['mcc_macro']:.4f} | "
        f"CK={default_valid_summary['ck_macro']:.4f} | "
        f"AUC={default_valid_summary['auc_macro']:.4f} | "
        f"Sn={default_valid_summary['sn_macro']:.4f} | "
        f"Sp={default_valid_summary['sp_macro']:.4f} | "
        f"ExactAcc={default_valid_summary['exact_match_acc']:.4f}"
    )

    logger.info("========== DEFAULT THRESHOLD TEST RESULTS ==========")
    logger.info(
        f"ACC={default_test_summary['acc_macro']:.4f} | "
        f"F1={default_test_summary['f1_macro']:.4f} | "
        f"MCC={default_test_summary['mcc_macro']:.4f} | "
        f"CK={default_test_summary['ck_macro']:.4f} | "
        f"AUC={default_test_summary['auc_macro']:.4f} | "
        f"Sn={default_test_summary['sn_macro']:.4f} | "
        f"Sp={default_test_summary['sp_macro']:.4f} | "
        f"ExactAcc={default_test_summary['exact_match_acc']:.4f}"
    )


    best_thresholds, best_threshold_df = optimize_thresholds_per_label(
        y_valid=y_valid,
        prob_valid=simple_avg_valid,
        cfg=cfg,
        logger=logger
    )


    optimized_valid_summary, _ = binary_metrics_per_label(
        y_true=y_valid,
        y_prob=simple_avg_valid,
        threshold=best_thresholds
    )
    optimized_test_summary, optimized_test_per_label = binary_metrics_per_label(
        y_true=y_test,
        y_prob=simple_avg_test,
        threshold=best_thresholds
    )

    logger.info("========== OPTIMIZED THRESHOLD VALID RESULTS ==========")
    logger.info(
        f"ACC={optimized_valid_summary['acc_macro']:.4f} | "
        f"F1={optimized_valid_summary['f1_macro']:.4f} | "
        f"MCC={optimized_valid_summary['mcc_macro']:.4f} | "
        f"CK={optimized_valid_summary['ck_macro']:.4f} | "
        f"AUC={optimized_valid_summary['auc_macro']:.4f} | "
        f"Sn={optimized_valid_summary['sn_macro']:.4f} | "
        f"Sp={optimized_valid_summary['sp_macro']:.4f} | "
        f"ExactAcc={optimized_valid_summary['exact_match_acc']:.4f}"
    )

    logger.info("========== OPTIMIZED THRESHOLD TEST RESULTS ==========")
    logger.info(
        f"ACC={optimized_test_summary['acc_macro']:.4f} | "
        f"F1={optimized_test_summary['f1_macro']:.4f} | "
        f"MCC={optimized_test_summary['mcc_macro']:.4f} | "
        f"CK={optimized_test_summary['ck_macro']:.4f} | "
        f"AUC={optimized_test_summary['auc_macro']:.4f} | "
        f"Sn={optimized_test_summary['sn_macro']:.4f} | "
        f"Sp={optimized_test_summary['sp_macro']:.4f} | "
        f"ExactAcc={optimized_test_summary['exact_match_acc']:.4f}"
    )


    np.save(os.path.join(cfg.save_dir, "simple_avg_valid_probs.npy"), simple_avg_valid)
    np.save(os.path.join(cfg.save_dir, "simple_avg_test_probs.npy"), simple_avg_test)
    np.save(os.path.join(cfg.save_dir, "valid_targets.npy"), y_valid)
    np.save(os.path.join(cfg.save_dir, "test_targets.npy"), y_test)
    np.save(os.path.join(cfg.save_dir, "optimized_thresholds.npy"), best_thresholds)


    pd.DataFrame([default_valid_summary]).to_csv(
        os.path.join(cfg.save_dir, "default_threshold_valid_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    pd.DataFrame([default_test_summary]).to_csv(
        os.path.join(cfg.save_dir, "default_threshold_test_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    pd.DataFrame([optimized_valid_summary]).to_csv(
        os.path.join(cfg.save_dir, "optimized_threshold_valid_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    pd.DataFrame([optimized_test_summary]).to_csv(
        os.path.join(cfg.save_dir, "optimized_threshold_test_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )


    pd.DataFrame(default_test_per_label).assign(label_name=label_cols).to_csv(
        os.path.join(cfg.save_dir, "default_threshold_test_per_label_metrics.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    pd.DataFrame(optimized_test_per_label).assign(label_name=label_cols).to_csv(
        os.path.join(cfg.save_dir, "optimized_threshold_test_per_label_metrics.csv"),
        index=False,
        encoding="utf-8-sig"
    )


    best_threshold_df["label_name"] = label_cols
    best_threshold_df.to_csv(
        os.path.join(cfg.save_dir, "labelwise_best_thresholds.csv"),
        index=False,
        encoding="utf-8-sig"
    )


    compare_df = pd.DataFrame([
        {
            "method": "default_threshold_0.5",
            "valid_acc": default_valid_summary["acc_macro"],
            "valid_f1": default_valid_summary["f1_macro"],
            "valid_mcc": default_valid_summary["mcc_macro"],
            "valid_ck": default_valid_summary["ck_macro"],
            "valid_auc": default_valid_summary["auc_macro"],
            "valid_sn": default_valid_summary["sn_macro"],
            "valid_sp": default_valid_summary["sp_macro"],
            "valid_exactacc": default_valid_summary["exact_match_acc"],
            "test_acc": default_test_summary["acc_macro"],
            "test_f1": default_test_summary["f1_macro"],
            "test_mcc": default_test_summary["mcc_macro"],
            "test_ck": default_test_summary["ck_macro"],
            "test_auc": default_test_summary["auc_macro"],
            "test_sn": default_test_summary["sn_macro"],
            "test_sp": default_test_summary["sp_macro"],
            "test_exactacc": default_test_summary["exact_match_acc"],
        },
        {
            "method": f"optimized_threshold_{cfg.optimize_metric}",
            "valid_acc": optimized_valid_summary["acc_macro"],
            "valid_f1": optimized_valid_summary["f1_macro"],
            "valid_mcc": optimized_valid_summary["mcc_macro"],
            "valid_ck": optimized_valid_summary["ck_macro"],
            "valid_auc": optimized_valid_summary["auc_macro"],
            "valid_sn": optimized_valid_summary["sn_macro"],
            "valid_sp": optimized_valid_summary["sp_macro"],
            "valid_exactacc": optimized_valid_summary["exact_match_acc"],
            "test_acc": optimized_test_summary["acc_macro"],
            "test_f1": optimized_test_summary["f1_macro"],
            "test_mcc": optimized_test_summary["mcc_macro"],
            "test_ck": optimized_test_summary["ck_macro"],
            "test_auc": optimized_test_summary["auc_macro"],
            "test_sn": optimized_test_summary["sn_macro"],
            "test_sp": optimized_test_summary["sp_macro"],
            "test_exactacc": optimized_test_summary["exact_match_acc"],
        }
    ])

    compare_df.to_csv(
        os.path.join(cfg.save_dir, "threshold_optimized_vs_default.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    logger.info("========== THRESHOLD OPTIMIZED VS DEFAULT ==========")
    logger.info("\n" + compare_df.to_string(index=False))


    meta = {
        "fusion_method": "simple_average",
        "threshold_optimization": True,
        "optimize_metric": cfg.optimize_metric,
        "threshold_min": cfg.threshold_min,
        "threshold_max": cfg.threshold_max,
        "threshold_step": cfg.threshold_step,
        "default_threshold": cfg.default_threshold,
        "gnn_valid_probs": cfg.gnn_valid_probs,
        "gnn_test_probs": cfg.gnn_test_probs,
        "fp_valid_probs": cfg.fp_valid_probs,
        "fp_test_probs": cfg.fp_test_probs,
        "desc_valid_probs": cfg.desc_valid_probs,
        "desc_test_probs": cfg.desc_test_probs,
        "label_cols": label_cols
    }

    with open(os.path.join(cfg.save_dir, "threshold_optimized_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info("Saved threshold optimization metadata: threshold_optimized_meta.json")


if __name__ == "__main__":
    main()
