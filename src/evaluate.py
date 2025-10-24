
import argparse, joblib, pathlib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

def main(data_dir, model_path):
    data_dir = pathlib.Path(data_dir)
    X = pd.read_csv(data_dir / "X_test.csv")
    y = pd.read_csv(data_dir / "y_test.csv").squeeze()

    model = joblib.load(model_path)
    y_pred = model.predict(X)

    print("Classification report:")
    print(classification_report(y, y_pred))

    try:
        # If the classifier supports predict_proba
        y_prob = model.predict_proba(X)[:,1]
        auc = roc_auc_score(y, y_prob)
        print(f"ROC AUC: {auc:.4f}")
        RocCurveDisplay.from_predictions(y, y_prob)
        plt.title("ROC Curve")
        plt.savefig("assets/screenshots/roc_curve.png", bbox_inches="tight")
        print("Saved ROC curve to assets/screenshots/roc_curve.png")
    except Exception as e:
        print("Could not compute ROC curve (no predict_proba?):", e)

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    import numpy as np
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha="center", va="center")
    plt.savefig("assets/screenshots/confusion_matrix.png", bbox_inches="tight")
    print("Saved confusion matrix to assets/screenshots/confusion_matrix.png")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="directory containing X_test.csv and y_test.csv")
    ap.add_argument("--model", required=True, help="path to model.joblib")
    args = ap.parse_args()
    main(args.data, args.model)
