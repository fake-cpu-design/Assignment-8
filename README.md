# Handwritten Digit Recognition using Artificial Neural Network (ANN)

## Objective
To automate the recognition of handwritten digits (0–9) on postal codes by developing and evaluating an Artificial Neural Network (ANN) classification model using the MNIST dataset.

## Dataset Link
- **Dataset**: MNIST Handwritten Digit Database
- **Link**: [MNIST in CSV on Kaggle](https://www.kaggle.com/datasets/odd-yplot/mnist-in-csv) / [Yann LeCun's MNIST Database](http://yann.lecun.com/exdb/mnist/)
- **Details**:
  - **Input Features**: 784 pixel values (`pixel0` to `pixel783`), representing a $28 \times 28$ grayscale image.
  - **Target Variable**: `label` (integer values ranging from 0 to 9).
  - **Train Set**: 60,000 samples (`mnist_train.csv`).
  - **Test Set**: 10,000 samples (`mnist_test.csv`).

## Libraries Used
- **Pandas**: Data loading, manipulation, and structural inspection.
- **NumPy**: Numerical operations, array reshaping, and vector matrix operations.
- **Matplotlib & Seaborn**: Visualization of sample digits, training history curves (loss and accuracy), and confusion matrix heatmap.
- **Scikit-Learn**: Model evaluation metrics, confusion matrix computation, and classification report generation.
- **TensorFlow / Keras**: Neural network design, layer configuration, compilation, model training, and predictions.

## Methodology
1. **Data Understanding**:
   - Loaded training (`mnist_train.csv`) and testing (`mnist_test.csv`) datasets into Pandas DataFrames.
   - Extracted 784 input pixel features and 1 target label column.
   - Checked dimensions and data types, and rendered sample digits using Matplotlib.

2. **Data Preprocessing**:
   - Verified that no missing or null values existed across training and testing datasets.
   - Separated features ($X$) and target labels ($y$).
   - **Feature Scaling**: Normalized pixel values from $[0, 255]$ to $[0.0, 1.0]$ by dividing pixel values by $255.0$ for optimal gradient descent optimization.
   - **Target Encoding**: Applied One-Hot Encoding to convert target class integers ($0-9$) into 10-dimensional binary vectors via Keras `to_categorical`.

3. **Model Development**:
   - Built a Sequential ANN model using TensorFlow/Keras.
   - Configured architecture with Dense layers using ReLU activations for hidden processing and Softmax for class probabilities.
   - Compiled model with **Adam** optimizer, **Categorical Crossentropy** loss, and **Accuracy** metric.
   - Trained model for 10 epochs with a batch size of 32.

4. **Model Evaluation**:
   - Evaluated final performance on 10,000 unseen test images.
   - Computed overall test accuracy and generated confusion matrix and class-wise classification report.
   - Rendered loss and accuracy trajectory graphs across 10 training epochs.

## Model Architecture
| Layer | Layer Type | Neurons / Units | Activation | Input Shape | Output Shape | Parameters |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Input** | Flatten / Direct | 784 | - | `(784,)` | `(784,)` | 0 |
| **Hidden Layer 1** | Dense | 128 | ReLU | `(784,)` | `(128,)` | 100,480 |
| **Hidden Layer 2** | Dense | 64 | ReLU | `(128,)` | `(64,)` | 8,256 |
| **Output Layer** | Dense | 10 | Softmax | `(64,)` | `(10,)` | 650 |

- **Total Parameters**: 109,386 (All trainable)

## Results
The model achieved an overall **Test Accuracy of 98.00%** ($0.9800$) on 10,000 test samples.

### Classification Report
```text
              precision    recall  f1-score   support

           0       0.98      0.99      0.99       980
           1       0.98      0.99      0.99      1135
           2       0.97      0.98      0.98      1032
           3       0.96      0.99      0.97      1010
           4       0.98      0.98      0.98       982
           5       0.99      0.97      0.98       892
           6       0.98      0.98      0.98       958
           7       0.99      0.96      0.97      1028
           8       0.97      0.97      0.97       974
           9       0.98      0.96      0.97      1009

    accuracy                           0.98     10000
   macro avg       0.98      0.98      0.98     10000
weighted avg       0.98      0.98      0.98     10000
```

### Key Performance Insights
1. **High Overall Precision & Recall**: Precision and recall consistently ranged between $0.96$ and $0.99$ across all individual digit categories (0 through 9).
2. **Fast & Stable Convergence**: Training and validation loss dropped rapidly within the first 3 epochs without significant oscillations.
3. **Strong Generalization**: The validation accuracy closely matched the training accuracy throughout all 10 epochs, proving minimal overfitting.
4. **Minor Confusion Areas**: Misclassifications primarily occurred between visually similar digits such as 4 vs 9 and 3 vs 8.

## Conclusion
The Artificial Neural Network achieved an overall test accuracy of 98% across 10,000 test samples, maintaining consistently strong performance across all digit classes with precision and recall ranging between 0.96 and 0.99. The two hidden layers (128 and 64 neurons with ReLU) were critical in enabling the network to learn non-linear patterns and construct abstract hierarchical representations from high-dimensional pixel inputs. A major advantage of Deep Learning over traditional Machine Learning is automated feature extraction directly from raw data, eliminating the need for complex, manual feature engineering. However, a key limitation of standard dense ANNs is that flattening 2D image matrices into 1D vectors destroys spatial relationships and localized structural context between adjacent pixels.
