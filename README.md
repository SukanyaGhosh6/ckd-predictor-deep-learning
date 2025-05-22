# CKD Predictor – A Deep Learning Based Detection System

Welcome to my project on **Chronic Kidney Disease Detection using Deep Learning**. I built this entire solution in **Visual Studio Code** using only **Python scripts** (no notebooks), with the intention of exploring how deep learning can assist in **early medical diagnosis**—especially for conditions like CKD, which often go unnoticed until the later stages.

This repository reflects my journey in applying machine learning techniques to a real-world, critical health problem. From preprocessing structured clinical data to interpreting model decisions using SHAP, everything here is developed from scratch to reinforce practical implementation skills.

---

##  Project Purpose

Chronic Kidney Disease (CKD) is often known as a silent killer. It progresses without obvious symptoms until it's in an advanced stage. The goal of this project is:

* To enable early detection of CKD using basic clinical features
* To demonstrate how structured health data can be used in deep learning pipelines
* To build an end-to-end, reproducible, and modular machine learning workflow

---

##  Project Structure

```
ckd-predictor-deep-learning/
├── data/                # Contains the raw and cleaned dataset
├── models/              # Saved deep learning model (.h5 format)
├── results/             # Plots, confusion matrix, ROC curves, logs
├── scripts/             # Python scripts for modular pipeline
│   ├── preprocess.py    # Handles cleaning, encoding, scaling
│   ├── train_model.py   # Builds and trains the model
│   └── evaluate.py      # Model evaluation and SHAP-based explainability
├── utils/               # (Optional) Common helper functions
├── main.py              # CLI entry point to trigger training and evaluation
├── requirements.txt     # All dependencies
└── README.md            # You're reading it!
```

---

##  Technologies Used

* Python
* TensorFlow & Keras
* Pandas, NumPy
* Scikit-learn
* Matplotlib & Seaborn
* SHAP (Explainability)

---

##  Dataset Details

* Source: [UCI Chronic Kidney Disease Dataset](https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease)
* Total Records: 400
* Features: 24 (age, blood pressure, albumin, sugar, hemoglobin, serum creatinine, etc.)
* Target: Presence or absence of CKD

I performed extensive data cleaning including:

* Handling missing values
* Label encoding categorical features
* Scaling numerical values

The cleaned dataset is saved as `ckd.csv` under `/data`.

---

##  Model Architecture

A **fully connected feedforward neural network** built using TensorFlow/Keras:

* Input Layer: 24 clinical features
* Hidden Layers: 2 dense layers with ReLU activation
* Dropout Layer: To avoid overfitting
* Output Layer: Sigmoid activation for binary classification

Includes **EarlyStopping** for monitoring validation loss and halting training when needed.

---

##  Evaluation Metrics

After training, the model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* ROC AUC Score

These outputs are stored in the `results/` folder.

---

##  Explainability with SHAP

To bring transparency to the model's decisions, I used **SHAP (SHapley Additive exPlanations)**. It helps interpret the feature contributions and makes the model's predictions more trustworthy, especially in a domain like healthcare.

---

##  How to Run the Project

> Make sure you have Python 3.10+ and pip installed

### 1. Clone the Repository

```bash
git clone https://github.com/SukanyaGhosh6/ckd-predictor-deep-learning.git
cd ckd-predictor-deep-learning
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv .venv
.\.venv\Scripts\activate     # Windows
# or
source .venv/bin/activate    # macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Pipeline

```bash
python main.py
```

This script handles preprocessing, training, saving the model, and then evaluating it.

---

##  Sample Outputs

* Confusion matrix and performance plots saved in `/results`
* Trained `.h5` model saved in `/models`
* SHAP values and feature importance graphs (optional)

---

##  Future Work

Some things I’d love to expand this project into:

* Deploy the model via a REST API (Flask or FastAPI)
* Add a frontend UI for real-time CKD checks
* Train using time-series health records
* Get feedback from healthcare professionals for refinement

---

##  About Me

I’m Sukanya Ghosh and I love working at the intersection of **AI and healthcare**. This project is a part of my continuous learning journey in data science and deep learning, and I hope to expand it with more health-focused ML models.

If you’re working on similar projects or want to collaborate, feel free to connect via [GitHub](https://github.com/SukanyaGhosh6).

---

## License

This repository is open-sourced under the **MIT License**. Feel free to use it for learning, academic work, or real-world prototyping.

