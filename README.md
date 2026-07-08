## Reproduction Order

Download the OpenPOM dataset from its original source and place it at:

```text
data/OpenPOM.csv
```

Then run:

```bash
pip install -r requirements.txt
python prepare_openpom_labels.py
python split_dataset.py
python train_descriptor_xgb.py
python train_fingerprint_xgb.py
python train_gnn_gatv2.py
python ensemble_simple_average.py
python optimize_label_thresholds.py
python compare_model_ensembles.py
```

Generated outputs are written to `results/`, which is intentionally excluded from the release files.
Processed data and split files are also generated locally and are intentionally excluded from version control.
