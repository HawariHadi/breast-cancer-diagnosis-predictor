
import argparse, pathlib, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

def main(data_dir, outdir):
    data_dir = pathlib.Path(data_dir)
    X = pd.read_csv(data_dir / "X_train.csv")
    y = pd.read_csv(data_dir / "y_train.csv").squeeze()

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42))
    ])

    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_val)
    print(classification_report(y_val, y_pred))

    out = pathlib.Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / "model.joblib")
    print(f"Saved model to {out / 'model.joblib'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="directory containing X_train.csv and y_train.csv")
    ap.add_argument("--outdir", default="models")
    args = ap.parse_args()
    main(args.data, args.outdir)
