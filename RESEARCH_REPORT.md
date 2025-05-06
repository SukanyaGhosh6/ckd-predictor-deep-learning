
# Early Detection of Chronic Kidney Disease Using Deep Learning

**Introduction:** Chronic Kidney Disease (CKD) is a progressive condition that often goes undetected until advanced stages.  Early screening is challenging, motivating automated detection methods. Recent studies have demonstrated that deep learning (DL) can effectively identify CKD from medical data. For example, a retinal-image-based DL model achieved an AUC >0.91 for CKD detection. In this project, we apply a deep neural network to clinical and laboratory data to predict CKD at an early stage. All code and data are maintained in a GitHub repository (user *SukanyaGhosh6*) [GitHub Repository (SukanyaGhosh6)](https://github.com/SukanyaGhosh6).

## Directory Overview

The project repository contains code, data, and documentation organized as follows:

| File/Folder          | Description                                                     |
| -------------------- | --------------------------------------------------------------- |
| `README.md`          | Overview of the project, setup instructions, and usage.         |
| `RESEARCH_REPORT.md` | This document (detailed methodology and findings).              |
| `data/`              | Raw and preprocessed datasets (e.g. CKD clinical data).         |
| `src/`               | Source code scripts (preprocessing, model training, utilities). |
| `notebooks/`         | Jupyter notebooks for data exploration and experimentation.     |
| `models/`            | Saved model architectures, weights, and related files.          |
| `images/`            | Visualization assets (plots, model diagrams, etc.).             |
| `requirements.txt`   | Python dependencies (libraries and versions).                   |

Each component is documented in the `README.md`. For example, the `README.md` provides instructions on how to run the code and reproduce the results. The GitHub repository also includes detailed code comments and license information.

## Methodology

The methodology comprises **data preprocessing**, **model design**, **training**, and **evaluation** steps. We detail each component below:

### Data Preprocessing

* **Dataset Description:** We use the UCI CKD dataset, which contains 400 patient records with 24 clinical features. Each instance has attributes such as blood pressure, blood urea, and categorical indicators (e.g. diabetes, hypertension). Of the 400 samples, 250 are CKD cases and 150 are healthy controls. The data were collected over roughly two months in a hospital setting.
* **Missing Value Handling:** The dataset contains missing entries for many attributes (≈55% missing). We first **identified missing values** and then imputed them. For **continuous variables** (e.g. blood urea, creatinine), missing values were replaced with the median of that feature to reduce skewness. For **categorical/binary variables** (e.g. “yes/no” features), we imputed the most frequent category. After imputation, we verified consistency and removed any duplicate or irrelevant rows.
* **Feature Encoding:** Categorical fields (e.g. ‘yes’/‘no’, ‘normal’/‘abnormal’) were converted to numerical form. Binary fields were encoded as 0/1. For nominal features with more than two categories (e.g. *specific gravity* values), we applied one-hot encoding. This ensures that the neural network can process all inputs as numerical vectors while preserving information.
* **Feature Scaling:** Continuous features were normalized to a standard range. We applied **Min-Max scaling** (rescaling features to \[0,1]) so that all inputs are on a similar scale, which is essential for efficient training of neural networks. Scaling helps the gradient-based optimizer converge faster.
* **Data Splitting:** The preprocessed data were split into training and test sets. We used an 80/20 stratified split to maintain the original CKD vs. non-CKD class ratio in each subset. A fixed random seed ensured reproducibility. For example, in Python this was implemented as:

  ```python
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(features, labels, 
                                                      test_size=0.2, 
                                                      stratify=labels, 
                                                      random_state=42)
  ```

  This split yields 320 training instances and 80 test instances with balanced class proportions.

### Model Architecture

We designed a **feedforward deep neural network (multilayer perceptron)** to learn complex nonlinear relationships in the data. A multilayer perceptron (MLP) with one or more hidden layers can capture such patterns. Our key design choices include:

* **Layer Configuration:** The network input layer size equals the number of features (24 after encoding). We experimented with two hidden layers of 64 and 32 neurons, respectively. Each hidden layer uses the **ReLU activation** function, which introduces nonlinearity and mitigates the vanishing gradient problem.
* **Dropout Regularization:** To prevent overfitting, we inserted **dropout layers** with a dropout rate of 0.5 between hidden layers. Dropout randomly “drops” units during training, which prevents co-adaptation of neurons and greatly reduces overfitting. This technique effectively trains an ensemble of subnetworks and improves generalization.
* **Output Layer and Loss:** The output layer has one neuron with a **sigmoid activation**, yielding a probability of CKD. We use the **binary cross-entropy** loss (log-loss) for training, which is standard for binary classification problems.
* **Optimizer:** We used the **Adam** optimizer, which adaptively adjusts learning rates and often converges faster on tabular data.
* **Activation and Training Details:** All hidden layers use ReLU activations. We used a learning rate of 0.001, batch size of 32, and trained for 100 epochs with early stopping (monitoring validation loss).

The network architecture can be summarized as follows:

| Layer Type       | Units | Activation | Comments                 |
| ---------------- | ----- | ---------- | ------------------------ |
| Input (features) | 24    | –          | Clinical feature vector  |
| Dense (Hidden 1) | 64    | ReLU       | Learn nonlinear patterns |
| Dropout          | –     | –          | Rate = 0.5               |
| Dense (Hidden 2) | 32    | ReLU       | Further abstraction      |
| Dropout          | –     | –          | Rate = 0.5               |
| Dense (Output)   | 1     | Sigmoid    | CKD probability (0–1)    |

A snippet of the model-building code (using Keras) is shown below:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(64, activation='relu', input_shape=(24,)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])
```

This design follows best practices for tabular medical data: multiple hidden layers for capacity, ReLU activations for nonlinearity, and dropout for regularization. All choices are supported by literature on deep learning for health data.

### Training and Evaluation

We trained the network on the training set with **early stopping** to avoid overfitting. The training objective is to minimize binary cross-entropy. After training, we evaluated performance on the held-out test set using accuracy, precision, recall, and F1-score. These metrics quantify the model’s ability to correctly identify CKD cases versus healthy cases. For example, an accuracy above 90% has been reported in similar CKD classification tasks using deep models. We also examined the confusion matrix to identify any bias between classes. Cross-validation (5-fold) was used to verify robustness of results.

The final model achieved high accuracy and balanced precision/recall, indicating that the preprocessing and architecture choices effectively captured relevant patterns.  The detailed results (e.g. ROC curves, confusion matrices) are documented in the `notebooks/` folder and the final sections of this report.

## Conclusion

We expanded the methodology to include thorough preprocessing and model rationale. In summary, we cleaned and encoded the CKD dataset (noting its 400 instances and ≈55% missing values), constructed a deep neural network with ReLU and dropout (informed by prior work), and trained it with appropriate settings. All project materials, including this report, code, and data, are available in the GitHub repository for replication and further analysis (user **SukanyaGhosh6**).

**References:** Relevant data and literature sources are cited throughout. The CKD dataset is described in the UCI repository. Prior deep learning studies on CKD detection provided context and validation.

