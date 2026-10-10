## Reproduction Order

Download the OpenPOM dataset from its original source and place it at:

```text
data/OpenPOM.csv
```

Then run from this folder:

```bash
python -m pip install -r requirements.txt
python prepare_openpom_labels.py
python split_dataset.py
python train_descriptor_xgb.py
python train_fingerprint_xgb.py
python gnn.py
python gnnnpy.py
python ensemble_simple_average.py
python optimize_label_thresholds.py
python compare_model_ensembles.py
```

Generated models and results are written to `results/`. The source data, processed splits, trained models, and generated results are not included in this upload folder. Exact reproduction of the reported values also requires the study's original data version and split files.

## Interpretation

After training, run:

```bash
python analysis_dataset.py
python descriptor_shap.py
python fingerprint_shap.py
python map_fingerprint_bits_to_substructures.py
python plot_gnn2.py
```

These scripts generate dataset summaries, descriptor and fingerprint contributions, substructure examples, and atom-level attributions. The corresponding figure scripts are included as `plot_fig*.py`. Their outputs are saved under `results/`.

## Predict New Molecules

The example input is `examples/new_molecules.csv`. It requires a `SMILES` column; `compound_name` is optional. After training the three branches and saving the label-specific thresholds, run:

```bash
python predict_new_molecules.py --input examples/new_molecules.csv --output results/example_predictions.csv
```

The output includes predicted labels and per-label probabilities for the original 18 odor categories. A different label set requires new training and evaluation. Fine-tuning is not provided.

For questions about code reproduction or implementation, please contact nemozjc@163.com.
