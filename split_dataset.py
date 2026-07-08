import os
import random
import logging
import numpy as np
import pandas as pd
import torch

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


def get_logger(filename):
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)


    return logger


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    print(f"Seed set to {seed}")


def multilabel_split_811(
    input_csv,
    output_dir,
    seed=2026,
    smiles_col="SMILES"
):
    os.makedirs(output_dir, exist_ok=True)

    logger = get_logger("split")
    seed_everything(seed)

    logger.info(f"Loading data: {input_csv}")
    df = pd.read_csv(input_csv)

    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column not found: {smiles_col}")


    label_cols = [c for c in df.columns if c != smiles_col]

    if len(label_cols) == 0:
        raise ValueError("No label columns found. Please check the input file.")

    logger.info(f"Total rows: {len(df)}")
    logger.info(f"Number of labels: {len(label_cols)}")
    logger.info(f"Label columns: {label_cols}")

    X = df[[smiles_col]]
    y = df[label_cols].values


    splitter1 = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=seed
    )

    train_idx, temp_idx = next(splitter1.split(X, y))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    logger.info(f"First split completed: train={len(train_df)}, temp={len(temp_df)}")


    X_temp = temp_df[[smiles_col]]
    y_temp = temp_df[label_cols].values

    splitter2 = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.5,
        random_state=seed
    )

    valid_idx, test_idx = next(splitter2.split(X_temp, y_temp))
    valid_df = temp_df.iloc[valid_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    logger.info(f"Second split completed: valid={len(valid_df)}, test={len(test_df)}")


    train_path = os.path.join(output_dir, "train.csv")
    valid_path = os.path.join(output_dir, "valid.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    valid_df.to_csv(valid_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    logger.info(f"Saved train: {train_path}")
    logger.info(f"Saved valid: {valid_path}")
    logger.info(f"Saved test : {test_path}")


    def log_label_stats(name, sub_df):
        label_sum = sub_df[label_cols].sum().sort_values(ascending=False)
        logger.info(f"\n{name} label frequency:\n{label_sum.to_string()}")

    log_label_stats("TRAIN", train_df)
    log_label_stats("VALID", valid_df)
    log_label_stats("TEST", test_df)

    logger.info("Split completed.")


if __name__ == "__main__":
    input_csv = "data/OpenPOM_labels_smiles_only.csv"
    output_dir = "data/split"
    seed = 42

    multilabel_split_811(
        input_csv=input_csv,
        output_dir=output_dir,
        seed=seed,
        smiles_col="SMILES"
    )
