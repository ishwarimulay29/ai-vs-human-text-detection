# How to Use This Bi-LSTM Python Script

This `.py` script demonstrates the process of building, training, and evaluating a Bi-directional Long Short-Term Memory (Bi-LSTM) model for a binary classification task. It includes data preprocessing, model architecture definition, hyperparameter tuning, and performance evaluation.

## Table of Contents

1.  [Purpose of the Script](#purpose-of-the-script)
2.  [Prerequisites](#prerequisites)
3.  [Running the Script](#running-the-script)
4.  [Understanding the Code](#understanding-the-code)
5.  [Interpreting Results](#interpreting-results)

## 1. Purpose of the Script

The primary goal of this script is to:

*   Preprocess text data for a Bi-LSTM model.
*   Define a Bi-LSTM neural network architecture using TensorFlow/Keras.
*   Perform hyperparameter tuning to find optimal model configurations.
*   Evaluate the model's performance using metrics like accuracy, precision, recall, and F1-score.
*   Identify the best-performing model based on the F1-score.

## 2. Prerequisites

Before running this script, ensure you have:

*   **Python Environment:** Run locally or on a server with Python 3.x. A GPU-accelerated environment (e.g., via CUDA) is recommended for faster training but not required.
*   **Required Libraries:** Install dependencies such as `pandas`, `numpy`, `tensorflow`, and `sklearn` (`pip install -r requirements.txt` if provided, or install individually).
*   **Data Files:** The script expects `train_data_with_features.csv`, `dev_data_with_features.csv`, and `test_data_with_features.csv` to be available in the working directory or the path referenced in the code. Update any hardcoded paths if your data lives elsewhere.

## 3. Running the Script

1.  Ensure the data files are in the expected path.
2.  From the project directory, run the script:
    ```bash
    python milestone3.py
    ```
3.  Watch the console logs for progress through preprocessing, training, and evaluation. Adjust paths or hyperparameters in the script as needed before rerunning.

## 4. Understanding the Code

### Data Preparation

*   **Loading Data:** Reads `train_df`, `dev_df`, and `test_df` from CSV files.
*   **Text/Label Separation:** `X_train`, `X_dev`, `X_test` contain the text data, and `y_train`, `y_dev`, `y_test` contain the corresponding labels (`0` or `1`).
*   **Tokenization:** A `Tokenizer` is initialized with `max_words=20000` and an OOV (Out-Of-Vocabulary) token. It learns the vocabulary from `X_train`.
*   **Sequence Conversion:** Text data is converted into numerical sequences based on the learned vocabulary.
*   **Sequence Length Analysis:** The distribution of sequence lengths in the training data is analyzed to determine an appropriate `max_len` for padding.
*   **Padding:** Sequences are padded (`post` padding and `post` truncation) to a uniform `max_len` to ensure all inputs to the Bi-LSTM have the same dimension.
*   **Label Conversion:** Labels are converted to NumPy arrays.

### Model Architecture (`Bi-LSTM`)

*   **`Embedding` Layer:** Converts input integer sequences into dense vectors of fixed size (`embedding_dim=128`). This layer learns word embeddings.
*   **`SpatialDropout1D`:** Applies dropout to entire feature maps rather than individual elements, which is effective for sequence data by dropping out random entire words (or embeddings).
*   **`Bidirectional(LSTM(128, return_sequences=False))`:** This is the core of the model. A Bidirectional wrapper allows the LSTM to process the sequence in both forward and backward directions, capturing dependencies from both past and future contexts. `128` is the number of LSTM units (hidden state dimension). `return_sequences=False` means only the output of the last time step is returned.
*   **`Dropout`:** A standard dropout layer applied to the output of the LSTM to prevent overfitting.
*   **`Dense(1, activation='sigmoid')`:** The output layer with a single neuron and a sigmoid activation function, suitable for binary classification (outputting a probability between 0 and 1).
*   **Compilation:** The model is compiled with the `adam` optimizer, `binary_crossentropy` loss (for binary classification), and `accuracy` as a metric.

### Hyperparameter Tuning

*   **`hyperparameters` List:** A list of dictionaries, each specifying a unique combination of `lstm_units`, `spatial_dropout_rate`, `dropout_rate`, `batch_size`, and `epochs` to experiment with.
*   **Training Loop:** The script iterates through each set of hyperparameters:
    *   A new Bi-LSTM model is created and compiled for each set.
    *   The model is trained (`model.fit`) on `X_train_padded` and `y_train`, with `validation_data` set to `(X_dev_padded, y_dev)` to monitor performance on the development set.
    *   After training, predictions (`y_pred_probs`) are made on the development set.
    *   Probabilities are converted to binary predictions (`y_pred`).
    *   Evaluation metrics (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`) are calculated.
    *   All results are stored in the `results` list.

## 5. Interpreting Results

After the hyperparameter tuning loop completes, the script will print a summary of the F1-scores for each hyperparameter set on the development data. It will then identify and print the details of the **best-performing hyperparameter set** based on its F1-score.

*   **`Set X: Accuracy=Y.YYY, F1-score=Z.ZZZ`**: This output indicates the performance of each hyperparameter set during the tuning process.
*   **`--- Best Hyperparameter Set ---`**: This section highlights the hyperparameters and all evaluation metrics (Accuracy, Precision, Recall, F1-score) for the model that achieved the highest F1-score on the development set.

**Key Metrics to Look For:**

*   **F1-score:** This is the primary metric used to select the best model, as it balances precision and recall, especially important for imbalanced datasets.
*   **Accuracy:** The proportion of correctly classified instances.
*   **Precision:** The proportion of positive identifications that were actually correct.
*   **Recall:** The proportion of actual positives that were identified correctly.