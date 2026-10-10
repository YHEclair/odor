import os
import json
import numpy as np
import pandas as pd


RAW_CSV = "./data/OpenPOM_labels_smiles_only.csv"
SMILES_COL = "SMILES"


TRAIN_CSV = "./data/split/train.csv"
VALID_CSV = "./data/split/valid.csv"
TEST_CSV = "./data/split/test.csv"

SAVE_DIR = "./results/dataset_analysis_output"


SELECTED_LABELS = [
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


USE_SELECTED_LABELS_ONLY = True


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_label_cols(df: pd.DataFrame, smiles_col: str):
    return [c for c in df.columns if c != smiles_col]


def summarize_df(df: pd.DataFrame, label_cols: list, split_name: str):


    y = df[label_cols].values.astype(int)
    labels_per_sample = y.sum(axis=1)

    summary = {
        "split": split_name,
        "num_samples": int(len(df)),
        "num_labels": int(len(label_cols)),
        "avg_labels_per_sample": float(labels_per_sample.mean()),
        "min_labels_per_sample": int(labels_per_sample.min()),
        "max_labels_per_sample": int(labels_per_sample.max()),
        "num_zero_label_samples": int((labels_per_sample == 0).sum()),
    }
    return summary


def calc_label_frequency(df: pd.DataFrame, label_cols: list, split_name: str):


    freq = df[label_cols].sum(axis=0).astype(int)
    out = pd.DataFrame({
        "label": label_cols,
        "positive_count": freq.values,
        "split": split_name
    }).sort_values("positive_count", ascending=False).reset_index(drop=True)
    return out


def calc_label_cooccurrence(df: pd.DataFrame, label_cols: list):


    y = df[label_cols].values.astype(int)                 
    co = y.T @ y
    co_df = pd.DataFrame(co, index=label_cols, columns=label_cols)
    return co_df


def calc_label_cooccurrence_ratio(df: pd.DataFrame, label_cols: list):


    y = df[label_cols].values.astype(int)
    co = y.T @ y
    pos_counts = y.sum(axis=0)

    ratio = np.zeros_like(co, dtype=float)
    for i in range(len(label_cols)):
        if pos_counts[i] > 0:
            ratio[i, :] = co[i, :] / pos_counts[i]
        else:
            ratio[i, :] = 0.0

    ratio_df = pd.DataFrame(ratio, index=label_cols, columns=label_cols)
    return ratio_df


def validate_selected_labels(df: pd.DataFrame, selected_labels: list):
    missing = [c for c in selected_labels if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"以下标签列在原始文件中不存在: {missing}")


def save_summary_json(summary_dict, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, ensure_ascii=False, indent=2)


def pretty_print_summary(summary: dict):
    print("-" * 60)
    print(f"Split: {summary['split']}")
    print(f"num_samples           : {summary['num_samples']}")
    print(f"num_labels            : {summary['num_labels']}")
    print(f"avg_labels_per_sample : {summary['avg_labels_per_sample']:.4f}")
    print(f"min_labels_per_sample : {summary['min_labels_per_sample']}")
    print(f"max_labels_per_sample : {summary['max_labels_per_sample']}")
    print(f"num_zero_label_samples: {summary['num_zero_label_samples']}")
    print("-" * 60)


def main():
    ensure_dir(SAVE_DIR)


    raw_df = pd.read_csv(RAW_CSV)

    if SMILES_COL not in raw_df.columns:
        raise ValueError(f"原始文件中未找到 SMILES 列: {SMILES_COL}")

    all_label_cols = get_label_cols(raw_df, SMILES_COL)

    if USE_SELECTED_LABELS_ONLY:
        validate_selected_labels(raw_df, SELECTED_LABELS)
        label_cols = SELECTED_LABELS
    else:
        label_cols = all_label_cols


    raw_df = raw_df[[SMILES_COL] + label_cols].copy()


    for col in label_cols:
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce").fillna(0).astype(int)


    raw_summary = summarize_df(raw_df, label_cols, split_name="raw_all")
    pretty_print_summary(raw_summary)

    pd.DataFrame([raw_summary]).to_csv(
        os.path.join(SAVE_DIR, "raw_dataset_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    save_summary_json(
        raw_summary,
        os.path.join(SAVE_DIR, "raw_dataset_summary.json")
    )


    raw_freq_df = calc_label_frequency(raw_df, label_cols, split_name="raw_all")
    raw_freq_df.to_csv(
        os.path.join(SAVE_DIR, "raw_label_frequency.csv"),
        index=False,
        encoding="utf-8-sig"
    )


    raw_co_df = calc_label_cooccurrence(raw_df, label_cols)
    raw_co_df.to_csv(
        os.path.join(SAVE_DIR, "raw_label_cooccurrence_count.csv"),
        encoding="utf-8-sig"
    )

    raw_co_ratio_df = calc_label_cooccurrence_ratio(raw_df, label_cols)
    raw_co_ratio_df.to_csv(
        os.path.join(SAVE_DIR, "raw_label_cooccurrence_ratio.csv"),
        encoding="utf-8-sig"
    )


    split_paths = {
        "train": TRAIN_CSV,
        "valid": VALID_CSV,
        "test": TEST_CSV
    }

    split_summaries = []
    split_freq_list = []

    split_files_exist = all(os.path.exists(p) for p in split_paths.values())

    if split_files_exist:
        print("检测到 train/valid/test 文件，开始额外统计划分数据。")

        for split_name, csv_path in split_paths.items():
            df = pd.read_csv(csv_path)

            if SMILES_COL not in df.columns:
                raise ValueError(f"{split_name} 文件中未找到 SMILES 列: {SMILES_COL}")

            validate_selected_labels(df, label_cols)
            df = df[[SMILES_COL] + label_cols].copy()

            for col in label_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

            summary = summarize_df(df, label_cols, split_name=split_name)
            split_summaries.append(summary)
            pretty_print_summary(summary)

            freq_df = calc_label_frequency(df, label_cols, split_name=split_name)
            split_freq_list.append(freq_df)

            co_df = calc_label_cooccurrence(df, label_cols)
            co_df.to_csv(
                os.path.join(SAVE_DIR, f"{split_name}_label_cooccurrence_count.csv"),
                encoding="utf-8-sig"
            )

            co_ratio_df = calc_label_cooccurrence_ratio(df, label_cols)
            co_ratio_df.to_csv(
                os.path.join(SAVE_DIR, f"{split_name}_label_cooccurrence_ratio.csv"),
                encoding="utf-8-sig"
            )


        pd.DataFrame(split_summaries).to_csv(
            os.path.join(SAVE_DIR, "split_dataset_summary.csv"),
            index=False,
            encoding="utf-8-sig"
        )


        split_freq_df = pd.concat(split_freq_list, axis=0, ignore_index=True)
        split_freq_df.to_csv(
            os.path.join(SAVE_DIR, "split_label_frequency.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    else:
        print("未检测到完整的 train/valid/test 文件，仅输出原始全数据统计结果。")


    pd.DataFrame({"label": label_cols}).to_csv(
        os.path.join(SAVE_DIR, "used_labels.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n已完成数据分析，输出目录：", SAVE_DIR)
    print("主要文件如下：")
    print("1) raw_dataset_summary.csv")
    print("2) raw_label_frequency.csv")
    print("3) raw_label_cooccurrence_count.csv")
    print("4) raw_label_cooccurrence_ratio.csv")
    if split_files_exist:
        print("5) split_dataset_summary.csv")
        print("6) split_label_frequency.csv")
        print("7) train/valid/test 各自的共现矩阵")


if __name__ == "__main__":
    main()