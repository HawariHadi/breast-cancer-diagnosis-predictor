from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
import pathlib, os

# ensure we're in the project folder (adjust if your path is different)
os.chdir(r"C:\Users\hadi2\Downloads\breast_cancer_repo_starter\breast-cancer-diagnosis-predictor")

# load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.DataFrame(data.target, columns=["target"])

# split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# make output folder
out_dir = pathlib.Path("data/processed")
out_dir.mkdir(parents=True, exist_ok=True)

# save csvs
X_train.to_csv(out_dir / "X_train.csv", index=False)
y_train.to_csv(out_dir / "y_train.csv", index=False)
X_test.to_csv(out_dir / "X_test.csv", index=False)
y_test.to_csv(out_dir / "y_test.csv", index=False)

print("✅ Created:")
for p in (out_dir / "X_train.csv", out_dir / "y_train.csv", out_dir / "X_test.csv", out_dir / "y_test.csv"):
    print(" -", p.resolve())
