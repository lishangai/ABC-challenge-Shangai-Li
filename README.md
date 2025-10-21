# 🧠 Human Activity Recognition Using CNN-LSTM with Attention

This project focuses on **classifying human activity types** based on triaxial acceleration data collected from wearable sensors.
It integrates **data preprocessing**, **feature extraction**, and **multi-model evaluation**—ranging from traditional machine learning methods to deep learning architectures with attention mechanisms.

The core goal is to develop a **robust and interpretable model** capable of recognizing human activities with high accuracy, despite individual variability and sensor noise.

---

## 🚀 Overview

* **Task:** Activity classification from acceleration signals
* **Techniques:** CNN, LSTM, Transformer, Attention Mechanism, Random Forest, XGBoost
* **Frameworks:** PyTorch, NumPy, Scikit-learn, Matplotlib
* **Key Outputs:** Model performance visualization, prediction tagging for unseen test data

---

## 🧩 Project Structure

### 1️⃣ Data Preprocessing

| File                 | Description                                                                              |
| -------------------- | ---------------------------------------------------------------------------------------- |
| **merge_data.ipynb** | Merges all acceleration data and assigns activity labels according to `trainactivities`. |
| **getdata.py**       | Splits merged data into separate files based on activity types for downstream modeling.  |

---

### 2️⃣ Model Development

#### 🧮 Deep Learning Models

| Script                                    | Description                                                            |
| ----------------------------------------- | ---------------------------------------------------------------------- |
| **cnn_lstm_attention_classifier_all.py**  | Main training pipeline for the CNN-LSTM Attention model.               |
| **cnn_lstm_attention_classifier_plot.py** | Generates performance plots (loss curves, accuracy, confusion matrix). |

#### ⚙️ Comparison Models — *Machine Learning Baselines*

Located in `machinelearning_comparison_models/`

* **Random Forest**

  * `model_trainer.py` – trains Random Forest classifier
  * `feature_extractor.py` – extracts statistical and time-domain features
  * `predict.py` – generates predictions and evaluation metrics
* **XGBoost**

  * `XGboost.ipynb` – trains and evaluates the XGBoost classifier

#### 🤖 Comparison Models — *Deep Learning Architectures*

Located in `deeplearning_comparison_models/`

Includes various hybrid designs combining **CNN, LSTM, and Transformer** structures:
`transformer_classifier_128.py`, `CNN_transformer_LSTM_classifier.py`, `transformer_cnn_classifier.py`, etc.
These models are used for comparative experiments and performance benchmarking.

---

### 3️⃣ Test Data Retrieval

| File                     | Description                                                 |
| ------------------------ | ----------------------------------------------------------- |
| **match_testdata.ipynb** | Matches test samples to corresponding acceleration records. |

---

### 4️⃣ Prediction Pipeline

| File                                       | Description                                                                                    |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| **cnn_lstm_attention_predict.py**          | Runs inference on new data using the best-trained model (`best_cnn_lstm_attention_model.pth`). |
| **cnn_lstm_attention_predict_testdata.py** | Predicts activity types for the test dataset and saves outputs as `predictions_by_minute.pkl`. |

---

### 5️⃣ Result Tagging

| File              | Description                                                             |
| ----------------- | ----------------------------------------------------------------------- |
| **get_result.py** | Updates the `testactivities` file by tagging predicted activity labels. |

---

## 📂 Supporting Folders

| Folder                    | Contents                                                                                           |
| ------------------------- | -------------------------------------------------------------------------------------------------- |
| **processed_activities/** | Preprocessed training and test datasets (`TrainActivities.csv`, `TestActivities.csv`).             |
| **plot_scripts/**         | Scripts for visualizing activity distribution, model accuracy, and feature importance.             |
| **experiment_results/**   | Evaluation outputs and visualizations — e.g., `classification_report.png`, `confusion_matrix.png`. |
| **saved_models/**         | Stored model checkpoints, trained scalers, and prediction files.                                   |

---

## 🧭 Execution Order

To ensure smooth execution, place all files and folders in the same working directory.
If path issues occur, flatten the folder structure so that all scripts are accessible from the root directory.

> **Acceleration data folder:** `users_timeXYZ/` must be located in the working directory.

**Run in the following order:**

1. `merge_data.ipynb` — merge and label acceleration data
2. `getdata.py` — split data by activity type
3. `cnn_lstm_attention_classifier_all.py` — train the CNN-LSTM Attention model
4. `match_testdata.ipynb` — match test data
5. `cnn_lstm_attention_predict_testdata.py` — generate predictions
6. `get_result.py` — produce final tagged test file

---

## ⚙️ Dependencies

Ensure the following Python libraries are installed:

```bash
pip install torch numpy pandas scikit-learn matplotlib xgboost
```

---

## 📊 Sample Outputs

The following figures are generated during model evaluation:

* **Confusion Matrix** – performance across activity classes
* **Loss Curve** – model convergence visualization
* **Feature Importance** – Random Forest and XGBoost interpretability
* **Accuracy Comparison** – between CNN-LSTM, Transformer, and baseline models

---

## 🧠 Key Highlights

* **End-to-end pipeline:** from raw data to labeled prediction output
* **Multi-model benchmarking:** comparison between ML and DL methods
* **Attention mechanism integration:** enhances interpretability and temporal focus
* **Reproducible workflow:** modular scripts and clear execution order

---

## ✨ Future Work

* Incorporate multimodal sensor fusion (e.g., gyroscope + accelerometer)
* Explore lightweight Transformer architectures for mobile deployment
* Implement explainable AI (XAI) modules for activity interpretability

---

## 👤 Author

**Shanga Li**
Huazhong University of Science and Technology

---

Would you like me to make this version **Markdown-formatted for GitHub** (with emoji icons, clean typography, and collapsible sections) so that it looks visually professional when uploaded? That style can significantly impress professors reviewing your profile.

