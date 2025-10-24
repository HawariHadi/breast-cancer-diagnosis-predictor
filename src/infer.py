
import argparse, joblib, pandas as pd

def main(model_path, input_csv, output_csv):
    model = joblib.load(model_path)
    X = pd.read_csv(input_csv)
    preds = model.predict(X)
    pd.DataFrame({"prediction": preds}).to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="predictions.csv")
    args = ap.parse_args()
    main(args.model, args.input, args.output)
