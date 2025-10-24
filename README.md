
# Breast Cancer Diagnosis Predictor

Predicts benign/malignant diagnosis from tabular features extracted from breast X-ray scans using an Artificial Neural Network (ANN).



## 🔍 Goals
- Clean/preprocess dataset
- Train an ANN baseline (scikit-learn MLP)
- Evaluate with a held-out test set
- Simple inference for new CSV inputs

## 📦 Tech
Python, scikit-learn, NumPy, Pandas, Matplotlib

## 🚀 Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Prepare your processed CSVs (see data/README.md), then:
python src/train.py --data data/processed --outdir models/
python src/evaluate.py --data data/processed --model models/model.joblib
python src/infer.py --model models/model.joblib --input samples/new_patients.csv --output predictions.csv
```

## 📁 Layout
- `src/` training & inference
- `data/` dataset notes (no raw data in Git)
- `models/` saved artifacts (ignored in Git)
- `assets/` charts/figures
- `tests/` quick smoke tests
- `.github/workflows/` CI

## 📚 Dataset
Add the dataset link & license in `data/README.md`.
Data is **not** included in this repo.

## 🧾 License
MIT
