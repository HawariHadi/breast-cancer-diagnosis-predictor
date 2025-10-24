
# Optional script stub if you want to automate raw->processed.
# Fill this based on your dataset's actual format.
import argparse, pathlib, pandas as pd

def main(src, dst):
    src = pathlib.Path(src)
    dst = pathlib.Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    # TODO: load your raw dataset, clean it, split features/labels
    # This is just an example scaffold:
    # raw = pd.read_csv(src / "raw.csv")
    # X = raw.drop(columns=["target"])
    # y = raw["target"]
    # X.to_csv(dst / "X_train.csv", index=False)
    # y.to_csv(dst / "y_train.csv", index=False)
    # (Repeat for test set)

    print("Fill in src/data_prep.py with your dataset logic.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    main(args.input, args.output)
