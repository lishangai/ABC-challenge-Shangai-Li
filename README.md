# Activity Recognition Project

This project processes acceleration data to classify various activity types using both machine learning and deep learning models. The workflow includes data preprocessing, model training, prediction generation, and result tagging. Models range from CNN-LSTM with attention mechanisms to traditional methods like Random Forest and XGBoost.

---

## File Structure

### 1. Preprocessing

- **merge_data.ipynb**  
  Merges all acceleration data and assigns activity type labels according to the `trainactivities` file.
- **getdata.py**  
  Splits the labeled acceleration data into separate files based on different activity types.

### 2. Models

#### CNN-LSTM Attention Models

- **cnn_lstm_attention_classifier_all.py**  
  Main script for training the CNN-LSTM attention model.
- **cnn_lstm_attention_classifier_plot.py**  
  Generates plots for model performance and evaluation.

#### Machine Learning Comparison Models (Folder: `machinelearning_comparison_models`)

- **Random Forest Model:**
  - **model_trainer.py**  
    Trains the Random Forest model.
  - **data_loader.py**  
    Loads data for model training and evaluation.
  - **main.py**  
    Main execution script for the Random Forest model.
  - **predict.py**  
    Generates predictions using the Random Forest model.
  - **feature_extractor.py**  
    Extracts features from the acceleration data.
  - **test.py**  
    Tests and evaluates the Random Forest model.
- **XGBoost Model:**
  - **XGboost.ipynb**  
    Notebook for training and evaluating the XGBoost model.

#### Deep Learning Comparison Models (Folder: `deeplearning_comparison_models`)

- **lstm_CNN_classifier.py**
- **lstm_classifier copy.py**
- **lstm_CNN_classifier copy.py**
- **transformer_classifier_128.py**
- **transformer_classifier_128_print.py**
- **lstm_predict.py**
- **transformer_classifier.py**
- **lstm_classifier.py**
- **CNN_LSTM_transformer_classifier.py**
- **CNN_transformer_LSTM_classifier.py**
- **transformer_cnn_classifier.py**
- **transformer_predict.py**

*These scripts include various implementations combining CNN, LSTM, and Transformer architectures.*

### 3. Test Data Acquisition

- **match_testdata.ipynb**  
  Retrieves the corresponding acceleration data for the test set.

### 4. Prediction

- **cnn_lstm_attention_predict.py**  
  Uses `best_cnn_lstm_attention_model.pth` to generate predictions on acceleration data.
- **cnn_lstm_attention_predict_testdata.py**  
  Predicts activity types for the test data and saves results in `predictions_by_minute.pkl`.

### 5. Tagging Test Data

- **get_result.py**  
  Uses `predictions_by_minute.pkl` to update the `testactivities` file with the predicted activity labels.

### Other Folders

- **processed_activities**  
  Contains processed activity files:
  
  - **TrainActivities.csv**: Preprocessed training activities (minutes with mixed activities have been removed).
  - **TestActivities.csv**
  - **readme.txt**

- **plot_scripts**  
  Contains scripts for generating various plots:
  
  - **plot_post_match_activity_label_counts.py**
  - **plot_model_accuracy_comparison.py**

- **experiment_results**  
  Contains evaluation outputs and figures:
  
  - **classification_report.png**
  - **confusion_matrix.png**
  - **loss_curve.png**
  - **acceleration_record_counts_gray.png**
  - **model_accuracies.png**
  - **feature_importance.png**
  - **model_results.txt**

- **saved_models**  
  Contains saved model files and scalers:
  
  - **lstm_scaler.pkl**
  - **transformer_scaler.pkl**
  - **best_cnn_lstm_attention_model.pth**
  - **cnn_lstms_attention_classifier_all_scaler.pkl**
  - **predictions_by_minute.pkl**
  - **saved_scaler.pkl**
  - **best_transformer_lstm_model.pth**
  - **transformer_lstm_scaler.pkl**

---

## Execution Order

Please ensure that all files are placed in the same working directory to avoid path issues.

Recommendation: Unpack all folders in the directory (i.e., move the contents of each folder directly into the working directory) so that every file can be accessed directly.

Note on acceleration data: Please note that the acceleration data files are not included here. The folder containing the acceleration data, **users_timeXYZ**, should be placed directly in the working directory.

1. **merge_data.ipynb**  
   Merge all acceleration data and label them.
2. **getdata.py**  
   Split the labeled data according to activity type.
3. **cnn_lstm_attention_classifier_all.py**  
   Train the CNN-LSTM attention model.
4. **match_testdata.ipynb**  
   Retrieve the corresponding acceleration data for the test set.
5. **cnn_lstm_attention_predict_testdata.py**  
   Generate predictions for the test data and save them in `predictions_by_minute.pkl`.
6. **get_result.py**  
   Tag the test data with the predicted activity labels.

---

## Additional Information

- **Dependencies:**  
  Ensure that all necessary libraries (e.g., PyTorch, NumPy, etc.) are installed before running the scripts.

- **Folder Names:**  
  The folder names (such as `processed_activities`, `saved_models`, etc.) are chosen to clearly represent their contents and facilitate easy navigation through the project.

- **Customization:**  
  You may adjust or extend any part of this project according to your specific requirements. Refer to individual scripts and notebooks for detailed configuration options and parameters.
