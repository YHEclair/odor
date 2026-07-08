import pandas as pd


openpom_file = "data/OpenPOM.csv"


selected_labels = [
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


df = pd.read_csv(openpom_file)


if "nonStereoSMILES" not in df.columns:
    raise ValueError("nonStereoSMILES column not found in OpenPOM.csv.")

smiles_col = "nonStereoSMILES"


existing_labels = [col for col in selected_labels if col in df.columns]
missing_labels = [col for col in selected_labels if col not in df.columns]

print(f"Found {len(existing_labels)} label columns.")
if missing_labels:
    print("Missing labels:")
    print(missing_labels)

if len(existing_labels) == 0:
    raise ValueError("No matching label columns found.")


filtered_df = df[df[existing_labels].sum(axis=1) > 0].copy()


result_df = filtered_df[[smiles_col] + existing_labels].copy()


result_df = result_df.rename(columns={smiles_col: "SMILES"})


result_df.to_csv("data/OpenPOM_labels_smiles_only.csv", index=False, encoding="utf-8-sig")
print("Saved: OpenPOM_labels_smiles_only.csv")
print(f"Filtered molecule count：{len(result_df)}")

print(result_df[existing_labels].sum().sort_values(ascending=False))
