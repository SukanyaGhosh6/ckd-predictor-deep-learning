# Chronic Kidney Disease Detection Using Deep Learning

A deep learning-based project built entirely in VS Code to detect Chronic Kidney Disease (CKD) early using structured clinical data. This project explores how machine learning and healthcare can intersect to support early-stage diagnostics through automation, explainability, and data-driven insight.

---

##  Overview

Chronic Kidney Disease (CKD) is often called a silent killer, progressing without symptoms until it's too late. This project leverages a neural network to predict the presence of CKD from basic clinical indicators, trained on the UCI CKD dataset.

Built entirely with Python (no Jupyter notebooks!), this repository includes:

* Data preprocessing
* A deep learning model (TensorFlow/Keras)
* SHAP-based explainability
* Evaluation metrics and visualization
* Modular Python scripts

---

##  Model Features

* Fully connected neural network
* Input: 24 clinical features (age, albumin, blood pressure, etc.)
* Architecture: 2 hidden layers + dropout + sigmoid output
* Early stopping and validation monitoring
* SHAP for model explainability

---

##  Project Structure

```
ckd-predictor-deep-learning/
├── data/                  # Raw and cleaned datasets (UCI CKD format)
├── scripts/               # Python scripts for pipeline steps
│   ├── preprocess.py      # Data cleaning and feature engineering
│   ├── train_model.py     # Training pipeline
│   └── evaluate.py        # Model evaluation and visualization
├── models/                # Saved trained models (.h5 or .pkl)
├── utils/                 # Helper functions and tools
├── results/               # Confusion matrix, plots, logs
├── main.py                # CLI entry point to trigger training and evaluation
├── RESEARCH_REPORT.md     # Detailed research methodology
├── README.md              # You’re here!
└── requirements.txt       # Dependencies
```

---

##  How to Run the Project

1. **Clone the repository**

```bash
git clone https://github.com/SukanyaGhosh6/ckd-predictor-deep-learning.git
cd ckd-predictor-deep-learning
```

2. **Create a virtual environment (optional but recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install requirements**

```bash
pip install -r requirements.txt
```

4. **Run the training pipeline**

```bash
python main.py
```

---

##  Dataset

* Source: [UCI CKD Dataset](https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease)
* 400 records with 24 features
* Preprocessing includes label encoding, normalization, and missing value handling

---

##  Model Performance

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 96.5% |
| Precision | 95.8% |
| Recall    | 97.3% |
| F1-Score  | 96.5% |

* Confusion matrix and ROC curve visualizations are saved in `/results`

---

##  Explainability

We used SHAP (SHapley Additive exPlanations) to understand how each feature contributes to the model's prediction.

* Top features: serum creatinine, albumin, hemoglobin, specific gravity
* Helps bridge model transparency with medical interpretability

Learn more about SHAP: [SHAP GitHub](https://github.com/slundberg/shap)

---

##  Future Work

* Deploy model as a REST API (Flask/FastAPI)
* Integrate with real-time health monitoring systems
* Expand to time-series data using LSTM
* Collaborate with medical practitioners for real-world testing

---

##  Contributions

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

##  References

* [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease)
* [TensorFlow Documentation](https://www.tensorflow.org/)
* [SHAP GitHub](https://github.com/slundberg/shap)
* [National Kidney Foundation](https://www.kidney.org/)

