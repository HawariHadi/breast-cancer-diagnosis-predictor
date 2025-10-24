
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
This project uses the **Breast Cancer Diagnostic Dataset**.

- **Source:** [Alister Baroi’s Breast Cancer Dataset](https://raw.githubusercontent.com/AlisterBaroi/breast-cancer-diagnosis-predictor/main/Breast_Cancer_Dataset.csv)
- **Description:** Contains tabular features extracted from X-ray scans of breast tissue (e.g., radius, texture, compactness, etc.) and a diagnosis label (`M` = malignant, `B` = benign).
- **License:** The dataset is publicly available for educational and research use.
- **Format:** CSV file with 12 columns and 80 rows.

⚠️ Data is **not included directly** in this repository.  
You can download it using the link above or replace it with a local file at runtime.

## 🧾 License
MIT
