# Early Detection of Chronic Kidney Disease Using Deep Learning

## Abstract

Chronic Kidney Disease (CKD) is a silent yet severe condition that progresses over time and can lead to kidney failure if not detected early. Early diagnosis is key, but traditional methods often rely heavily on medical history and lab tests, which may not be readily available or interpretable by all healthcare professionals. In this research project, we explore a deep learning-based approach for early CKD detection using patient health indicators. Leveraging open-source tools and working exclusively in VS Code, we developed, trained, and evaluated a neural network model to identify patterns indicative of CKD. This research aims to bridge clinical insights with data science, making early detection smarter, scalable, and more accessible.


GitHub Repository: [github.com/SukanyaGhosh6/ckd-predictor-deep-learning](https://github.com/SukanyaGhosh6/ckd-predictor-deep-learning)

---

## 1. Introduction

Chronic Kidney Disease affects over 10% of the global population, often going unnoticed until it reaches advanced stages. According to the [National Kidney Foundation](https://www.kidney.org/), early detection can significantly delay progression, yet many patients are diagnosed late due to non-specific symptoms. With the rise of digital health records and advances in machine learning, predictive modeling offers a promising pathway to flag early signs of CKD using simple clinical features.

We chose to approach this as a deep learning project to experiment with multi-layered representations and build a model capable of capturing subtle relationships in the data — ones that traditional models might miss.

## 2. Related Work

Several studies ([IEEE Xplore](https://ieeexplore.ieee.org/), [PubMed](https://pubmed.ncbi.nlm.nih.gov/)) have investigated ML models for CKD prediction using logistic regression, decision trees, and ensemble techniques. Our approach differentiates itself by focusing on:

* End-to-end deep learning using TensorFlow/Keras
* Emphasis on explainability with SHAP or LIME
* Modern Python tooling (VS Code, Jupyter extensions, GitHub CI/CD)

## 3. Dataset

We used the [Chronic Kidney Disease dataset](https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease) from the UCI Machine Learning Repository. It contains 400 patient records and 24 features including:

* Age, Blood Pressure (bp), Specific Gravity (sg), Albumin (al), Sugar (su)
* Red blood cells (rbc), Pus cell (pc), Serum creatinine (sc), Hemoglobin (hemo), Packed cell volume (pcv), etc.

The target variable is `classification` (ckd or notckd).

Data preprocessing was critical. We handled:

* Missing values using median/mode imputation
* Label encoding for categorical variables
* MinMaxScaler for normalization

## 4. Methodology

### 4.1 Tools and Environment

* **Editor**: VS Code (Extensions: Python, Jupyter, GitLens)
* **Language**: Python 3.12
* **Libraries**: pandas, numpy, scikit-learn, TensorFlow/Keras, matplotlib, seaborn
* **Versioning**: git + GitHub

### 4.2 Model Architecture

We built a fully connected feedforward neural network with the following structure:

* Input layer (24 features)
* Hidden Layer 1: 64 neurons + ReLU
* Hidden Layer 2: 32 neurons + ReLU
* Dropout (0.3) to prevent overfitting
* Output layer: Sigmoid (binary classification)

```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

### 4.3 Training & Evaluation

* Loss function: Binary Crossentropy
* Optimizer: Adam
* Metrics: Accuracy, Precision, Recall, F1-Score
* Epochs: 100
* Batch size: 32
* Validation Split: 0.2

We used early stopping to prevent overfitting.

### 4.4 Explainability

We used [SHAP](https://github.com/slundberg/shap) to interpret the model predictions and highlight the most influential features.

## 5. Results

Our model achieved the following performance on the test set:

* Accuracy: 96.5%
* Precision: 95.8%
* Recall: 97.3%
* F1-Score: 96.5%

Feature importance showed that serum creatinine, albumin, hemoglobin, and specific gravity were the strongest indicators of CKD.

We plotted ROC curves and confusion matrices to visualize the performance.

## 6. Conclusion

The proposed deep learning model proved highly effective in detecting early signs of CKD using basic clinical indicators. Not only does this streamline diagnostic workflows, but it also lays the foundation for building scalable, real-time health monitoring systems.

We strongly believe that combining clinical data with intelligent algorithms can redefine early diagnostics and personalized medicine.

## 7. Future Work

* Deploy the model as a Flask API or FastAPI for hospital integration
* Extend the dataset via clinical partnerships or synthetic data generation
* Explore hybrid models combining CNNs or transformers for time-series lab records
* Integrate with wearable/IoT devices for real-time monitoring

## 8. References

* [UCI ML Repository: CKD Dataset](https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease)
* [TensorFlow Docs](https://www.tensorflow.org/)
* [SHAP GitHub](https://github.com/slundberg/shap)
* [National Kidney Foundation](https://www.kidney.org/)
* [WHO: Kidney Disease](https://www.who.int/news-room/fact-sheets/detail/kidney-disease)

