# logistic_regression
Classification model example  -Machine Learning (Supervised)
# README

This code performs binary classification using logistic regression to predict diabetes based on several input features. It utilizes the scikit-learn library for logistic regression, model evaluation, and data preprocessing. Here's a breakdown of the code and its functionality:

1. Importing the necessary libraries:
   - `pandas` for data manipulation and analysis
   - `LogisticRegression` from `sklearn.linear_model` for logistic regression
   - `accuracy_score`, `recall_score`, and `precision_score` from `sklearn.metrics` for model evaluation
   - `train_test_split` from `sklearn.model_selection` to split the dataset into training and testing sets
   - `matplotlib.pyplot` and `seaborn` for data visualization
   - `confusion_matrix` from `sklearn.metrics` to calculate the confusion matrix
   - `roc_curve` and `auc` from `sklearn.metrics` to plot the ROC curve

2. Loading the dataset:
   - The code reads the CSV file named "diabetes3.csv" using `pd.read_csv()` from the pandas library.
   - The dataset is assumed to have nine columns, with the last column representing the target variable (diabetes).

3. Extracting features and labels:
   - The code separates the input features (X) and the target variable (y) from the dataset.

4. Splitting the dataset:
   - The code uses `train_test_split()` to split the dataset into training and testing sets.
   - It assigns 80% of the data to the training set and 20% to the testing set.
   - The `random_state` parameter is set to 42 for reproducibility.

5. Creating and training the logistic regression model:
   - The code creates an instance of the logistic regression model using `LogisticRegression()`.
   - It trains the model on the training data using `fit()`.

6. Making predictions and evaluating the model:
   - The code uses the trained model to make predictions on the testing data using `predict()`.
   - It calculates accuracy, recall, and precision scores using `accuracy_score()`, `recall_score()`, and `precision_score()`.

7. Visualizing the results:
   - The code plots a heatmap of the confusion matrix using `seaborn.heatmap()`.
   - It plots the ROC curve using `roc_curve()` and `auc()`.
   - The x-axis represents the false positive rate, and the y-axis represents the true positive rate.
   - The area under the ROC curve (AUC) is also displayed.

Note: Make sure to update the file path in `data = pd.read_csv("D:\Jamal's\diabetes3.csv")` to the correct location of your "diabetes3.csv" file.

This code provides a basic implementation of logistic regression for diabetes prediction. You can modify and expand upon it to suit your specific requirements.
