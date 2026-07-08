import os
import random
import logging
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message=".*The usage of `scatter.*")
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from rdkit import Chem
from scipy.sparse import coo_matrix

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
    accuracy_score
)

from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool


def get_logger(filename):
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)


    return logger


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    seed = 2026

    train_csv = "data/split/train.csv"
    valid_csv = "data/split/valid.csv"
    test_csv = "data/split/test.csv"

    save_dir = "results/gnn_output"
    model_name = "gatv2_shared_backbone_independent_heads.pt"

    smiles_col = "SMILES"

    batch_size = 64
    num_workers = 0

    lr = 3e-4
    weight_decay = 1e-5
    epochs = 200
    patience = 20

    gnn_input_dim = 64
    gnn_hidden_dim = 64
    gnn_head = 4
    gnn_dropout = 0.2
    gnn_output_dim = 18

    threshold = 0.5
    device = "cuda" if torch.cuda.is_available() else "cpu"


def one_hot_encoding(value, choices):
    encoding = [0] * len(choices)
    if value in choices:
        encoding[choices.index(value)] = 1
    return encoding


class MoleculeFeaturizer(object):
    def __init__(self, bond_features=True):
        self.bond_features = bond_features

    def _safe_one_hot(self, value, choices):
        encoding = [0] * len(choices)
        if value in choices:
            encoding[choices.index(value)] = 1
        return encoding

    def _atom_featurizer(self, atom):
        atomic_numer = [
            1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21,
            28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 46, 47, 48, 49, 50,
            51, 52, 53
        ]
        atom_type_features = self._safe_one_hot(atom.GetAtomicNum(), atomic_numer)

        degree_features = self._safe_one_hot(atom.GetTotalDegree(), list(range(6)))

        hybridization_choices = list(range(len(Chem.HybridizationType.names) - 1))
        hybrid_features = self._safe_one_hot(int(atom.GetHybridization()), hybridization_choices)

        chiral_tag_choices = list(range(len(Chem.ChiralType.names) - 1))
        chiral_tag_features = self._safe_one_hot(int(atom.GetChiralTag()), chiral_tag_choices)

        num_Hs_features = self._safe_one_hot(atom.GetTotalNumHs(), list(range(6)))

        aromatic_features = [int(atom.GetIsAromatic())]


        formal_charge_choices = [-2, -1, 0, 1, 2]
        formal_charge_features = self._safe_one_hot(atom.GetFormalCharge(), formal_charge_choices)

        is_in_ring_feature = [int(atom.IsInRing())]

        explicit_val = atom.GetValence(Chem.rdchem.ValenceType.EXPLICIT)
        implicit_val = atom.GetValence(Chem.rdchem.ValenceType.IMPLICIT)

        explicit_valence_features = self._safe_one_hot(int(explicit_val), list(range(9)))
        implicit_valence_features = self._safe_one_hot(int(implicit_val), list(range(9)))

        radical_e_features = self._safe_one_hot(int(atom.GetNumRadicalElectrons()), list(range(5)))

        features = (
            atom_type_features
            + degree_features
            + hybrid_features
            + chiral_tag_features
            + num_Hs_features
            + aromatic_features
            + formal_charge_features
            + is_in_ring_feature
            + explicit_valence_features
            + implicit_valence_features
            + radical_e_features
        )
        return features

    def _bond_featurizer(self, bond):
        bond_type = bond.GetBondType()
        bond_type_features = [
            int(bond_type == Chem.rdchem.BondType.SINGLE),
            int(bond_type == Chem.rdchem.BondType.DOUBLE),
            int(bond_type == Chem.rdchem.BondType.TRIPLE),
            int(bond_type == Chem.rdchem.BondType.AROMATIC)
        ]

        conjugated_and_ring = [
            int(bond.GetIsConjugated()),
            int(bond.IsInRing())
        ]


        stereo_choices = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS
        ]
        stereo_features = self._safe_one_hot(bond.GetStereo(), stereo_choices)

        direction_choices = [
            Chem.rdchem.BondDir.NONE,
            Chem.rdchem.BondDir.ENDUPRIGHT,
            Chem.rdchem.BondDir.ENDDOWNRIGHT,
            Chem.rdchem.BondDir.EITHERDOUBLE
        ]
        direction_features = self._safe_one_hot(bond.GetBondDir(), direction_choices)

        return bond_type_features + conjugated_and_ring + stereo_features + direction_features

    def __call__(self, mol):
        atom_features = [self._atom_featurizer(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(atom_features, dtype=torch.float)

        adj = Chem.GetAdjacencyMatrix(mol)
        coo_adj = coo_matrix(adj)
        edge_index = torch.tensor(
            np.array([coo_adj.row, coo_adj.col]),
            dtype=torch.long
        )


        edge_feature_dim = 16

        bond_features = []
        if self.bond_features and edge_index.size(1) > 0:
            for i, j in zip(coo_adj.row, coo_adj.col):
                bond = mol.GetBondBetweenAtoms(int(i), int(j))
                if bond is not None:
                    bond_features.append(self._bond_featurizer(bond))
                else:
                    bond_features.append([0] * edge_feature_dim)
            edge_attr = torch.tensor(bond_features, dtype=torch.float)
        else:
            edge_attr = torch.zeros((0, edge_feature_dim), dtype=torch.float)

        return x, edge_index, edge_attr


class GraphDataset(Dataset):
    def __init__(self, df, smiles_col, label_cols, featurizer=None):
        self.df = df.reset_index(drop=True)
        self.smiles_col = smiles_col
        self.label_cols = label_cols
        self.featurizer = featurizer if featurizer is not None else MoleculeFeaturizer()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = row[self.smiles_col]

        y = torch.tensor(
            row[self.label_cols].values.astype(np.float32),
            dtype=torch.float
        ).unsqueeze(0)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        x, edge_index, edge_attr = self.featurizer(mol)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y
        )
        data.smiles = smiles
        return data


class GATv2MultiLabel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        gat_heads = self.config.gnn_head
        edge_dim = 16
        hidden_dim_all = self.config.gnn_hidden_dim * gat_heads

        self.cconv1 = GATv2Conv(
            self.config.gnn_input_dim,
            self.config.gnn_hidden_dim,
            heads=gat_heads,
            edge_dim=edge_dim
        )
        self.cconv2 = GATv2Conv(
            hidden_dim_all,
            self.config.gnn_hidden_dim,
            heads=gat_heads,
            edge_dim=edge_dim
        )
        self.cconv3 = GATv2Conv(
            hidden_dim_all,
            self.config.gnn_hidden_dim,
            heads=gat_heads,
            edge_dim=edge_dim
        )

        self.norm1 = nn.LayerNorm(hidden_dim_all)
        self.norm2 = nn.LayerNorm(hidden_dim_all)
        self.norm3 = nn.LayerNorm(hidden_dim_all)

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(self.config.gnn_dropout)

        mlp_input_dim = hidden_dim_all * 2
        mlp_hidden_dim = mlp_input_dim // 2


        self.shared_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.gnn_dropout)
        )


        self.classifier_heads = nn.ModuleList([
            nn.Linear(mlp_hidden_dim, 1) for _ in range(self.config.gnn_output_dim)
        ])

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.cconv1(x, edge_index, edge_attr)
        x = self.dropout(x)
        x = self.norm1(x)
        x = self.relu(x)

        identity = x
        x = self.cconv2(x, edge_index, edge_attr)
        x = self.dropout(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = x + identity

        identity = x
        x = self.cconv3(x, edge_index, edge_attr)
        x = self.dropout(x)
        x = self.norm3(x)
        x = self.relu(x)
        x = x + identity

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_pooled = torch.cat([x_mean, x_max], dim=1)

        shared_features = self.shared_mlp(x_pooled)

        logits = torch.cat(
            [head(shared_features) for head in self.classifier_heads],
            dim=1
        )
        return logits


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


def train_one_epoch(model, loader, optimizer, criterion, device, epoch=None, total_epochs=None):
    model.train()
    running_loss = 0.0

    desc = f"Train Epoch {epoch}/{total_epochs}" if epoch is not None else "Training"
    pbar = tqdm(loader, desc=desc, leave=False)

    for batch_data in pbar:
        batch_data = batch_data.to(device)

        optimizer.zero_grad()

        logits = model(
            batch_data.x,
            batch_data.edge_index,
            batch_data.edge_attr,
            batch_data.batch
        )

        y = batch_data.y.view(batch_data.num_graphs, -1)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_data.num_graphs
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5, desc="Evaluating"):
    model.eval()

    running_loss = 0.0
    all_probs = []
    all_true = []

    pbar = tqdm(loader, desc=desc, leave=False)

    for batch_data in pbar:
        batch_data = batch_data.to(device)

        logits = model(
            batch_data.x,
            batch_data.edge_index,
            batch_data.edge_attr,
            batch_data.batch
        )

        y = batch_data.y.view(batch_data.num_graphs, -1)
        loss = criterion(logits, y)
        running_loss += loss.item() * batch_data.num_graphs

        probs = torch.sigmoid(logits).cpu().numpy()
        y_true = y.cpu().numpy()

        all_probs.append(probs)
        all_true.append(y_true)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / len(loader.dataset)

    all_probs = np.vstack(all_probs)
    all_true = np.vstack(all_true)

    summary, per_label = binary_metrics_per_label(
        y_true=all_true,
        y_prob=all_probs,
        threshold=threshold
    )

    return epoch_loss, summary, per_label


def main():
    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)


    logger = get_logger("train")
    seed_everything(cfg.seed)
    logger.info(f"Using device: {cfg.device}")


    train_df = pd.read_csv(cfg.train_csv)
    valid_df = pd.read_csv(cfg.valid_csv)
    test_df = pd.read_csv(cfg.test_csv)


    label_cols = [c for c in train_df.columns if c != cfg.smiles_col]
    cfg.gnn_output_dim = len(label_cols)

    logger.info(f"Train size: {len(train_df)}")
    logger.info(f"Valid size: {len(valid_df)}")
    logger.info(f"Test size : {len(test_df)}")
    logger.info(f"Num labels: {len(label_cols)}")
    logger.info(f"Label cols: {label_cols}")


    featurizer = MoleculeFeaturizer()

    sample_smiles = train_df.iloc[0][cfg.smiles_col]
    sample_mol = Chem.MolFromSmiles(sample_smiles)
    if sample_mol is None:
        raise ValueError(f"Invalid first training SMILES: {sample_smiles}")

    sample_x, sample_edge_index, sample_edge_attr = featurizer(sample_mol)

    cfg.gnn_input_dim = sample_x.shape[1]
    edge_dim_detected = sample_edge_attr.shape[1] if sample_edge_attr.numel() > 0 else 16

    logger.info(f"Auto-detected atom feature dim: {cfg.gnn_input_dim}")
    logger.info(f"Auto-detected edge feature dim: {edge_dim_detected}")


    train_dataset = GraphDataset(train_df, cfg.smiles_col, label_cols, featurizer)
    valid_dataset = GraphDataset(valid_df, cfg.smiles_col, label_cols, featurizer)
    test_dataset = GraphDataset(test_df, cfg.smiles_col, label_cols, featurizer)


    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers
    )


    model = GATv2MultiLabel(cfg).to(cfg.device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )


    criterion = nn.BCEWithLogitsLoss()

    logger.info(model)
    logger.info(f"Criterion: {criterion}")
    logger.info(f"Optimizer: Adam(lr={cfg.lr}, weight_decay={cfg.weight_decay})")
    logger.info("Architecture: shared backbone + independent binary heads")
    logger.info("Feature setting: enhanced atom features + enhanced bond features")


    best_valid_auc = -np.inf
    best_epoch = -1
    wait = 0

    for epoch in range(1, cfg.epochs + 1):
        logger.info(f"========== Epoch {epoch}/{cfg.epochs} ==========")

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=cfg.device,
            epoch=epoch,
            total_epochs=cfg.epochs
        )

        valid_loss, valid_summary, valid_per_label = evaluate(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=cfg.device,
            threshold=cfg.threshold,
            desc=f"Valid Epoch {epoch}/{cfg.epochs}"
        )

        logger.info(
            f"[Epoch {epoch:03d}] "
            f"TrainLoss={train_loss:.6f} | "
            f"ValidLoss={valid_loss:.6f} | "
            f"ACC={valid_summary['acc_macro']:.4f} | "
            f"F1={valid_summary['f1_macro']:.4f} | "
            f"MCC={valid_summary['mcc_macro']:.4f} | "
            f"CK={valid_summary['ck_macro']:.4f} | "
            f"AUC={valid_summary['auc_macro']:.4f} | "
            f"Sn={valid_summary['sn_macro']:.4f} | "
            f"Sp={valid_summary['sp_macro']:.4f} | "
            f"ExactAcc={valid_summary['exact_match_acc']:.4f}"
        )

        current_valid_auc = valid_summary["auc_macro"]

        if current_valid_auc > best_valid_auc:
            best_valid_auc = current_valid_auc
            best_epoch = epoch
            wait = 0

            torch.save(
                {
                    "epoch": int(epoch),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_valid_auc": float(best_valid_auc),
                    "label_cols": list(label_cols),
                    "gnn_input_dim": int(cfg.gnn_input_dim),
                    "gnn_output_dim": int(cfg.gnn_output_dim),
                },
                os.path.join(cfg.save_dir, cfg.model_name)
            )
            logger.info(
                f"New best model saved at epoch {epoch}, "
                f"valid AUC={best_valid_auc:.4f}"
            )
        else:
            wait += 1
            if wait >= cfg.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    logger.info(f"Best epoch: {best_epoch}, Best valid AUC: {best_valid_auc:.4f}")


    ckpt = torch.load(
        os.path.join(cfg.save_dir, cfg.model_name),
        map_location=cfg.device,
        weights_only=False
    )
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_summary, test_per_label = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=cfg.device,
        threshold=cfg.threshold,
        desc="Testing"
    )

    logger.info("========== TEST RESULTS ==========")
    logger.info(
        f"TestLoss={test_loss:.6f} | "
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


if __name__ == "__main__":
    print("main entered")
    main()
