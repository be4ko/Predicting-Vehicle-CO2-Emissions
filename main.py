import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score

# Load the data
data = pandas.read_csv("co2_emissions_data.csv")

# Check whether there are missing values
missing_values = data.isna().sum()
print(missing_values)

# Check whether numeric features have the same scale
numericData = data.select_dtypes(include=['float64', 'int64'])
print("\nRange of each numeric feature:")
for column in numericData.columns:
    range = numericData[column].max() - numericData[column].min()
    print(f"{column}: {range}")

# Visualize a pair-plot in which diagonal subplots are histograms
sns.pairplot(numericData, diag_kind="hist")
plt.show()

# Visualize a correlation heatmap between numeric columns
correlation_matrix = numericData.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

################# Data Preprocessing #################

# Separate features and targets
features = data.drop(columns=["CO2 Emissions(g/km)", "Emission Class"])
target_regression = data["CO2 Emissions(g/km)"]
target_classification = data["Emission Class"]

# Identify categorical columns
categorical_columns = features.select_dtypes(include=['object']).columns

# Encode categorical columns
label_encoders = {}
for column in categorical_columns:
    encoder = LabelEncoder()
    features[column] = encoder.fit_transform(features[column])  # Encode categorical values
    label_encoders[column] = encoder

# Encode the classification target
classification_encoder = LabelEncoder()
target_classification = classification_encoder.fit_transform(target_classification)

# Split features and targets for regression model
features_train, features_test, target_train_reg, target_test_reg = train_test_split(
    features, target_regression, test_size=0.2, random_state=42
)

# Split features and targets for classification model
_, _, target_train_class, target_test_class = train_test_split(
    features, target_classification, test_size=0.2, random_state=42
)

# Identify numeric columns
numeric_columns = features.select_dtypes(include=['float64', 'int64']).columns

# Scale features using the training set statistics
scaler = StandardScaler()
features_train[numeric_columns] = scaler.fit_transform(features_train[numeric_columns])  # Fit and transform training data
features_test[numeric_columns] = scaler.transform(features_test[numeric_columns])  # Transform testing data

# Select two independent variables based on the correlation heatmap
# These should have a strong relationship with the target but not with each other
selected_features = ["Feature1", "Feature2"]  # Replace with actual column names based on the heatmap
X_train = features_train[selected_features].values
X_test = features_test[selected_features].values
y_train = target_train_reg.values
y_test = target_test_reg.values

################# Linear Regression Using Gradient Descent #################

# Initialize parameters
def initialize_weights(n):
    return np.zeros((n, 1)), 0  # Weights (W) and bias (b)

# Hypothesis function
def predict(X, W, b):
    return np.dot(X, W) + b

# Cost function
def compute_cost(y, y_pred):
    return np.mean((y_pred - y) ** 2) / 2

# Gradient descent function
def gradient_descent(X, y, W, b, alpha, num_iterations):
    m = len(y)
    costs = []
    for i in range(num_iterations):
        y_pred = predict(X, W, b)
        error = y_pred - y
        dW = np.dot(X.T, error) / m
        db = np.sum(error) / m
        W -= alpha * dW
        b -= alpha * db
        cost = compute_cost(y, y_pred)
        costs.append(cost)
    return W, b, costs

# Prepare data for gradient descent
X_train_gd = np.c_[np.ones(X_train.shape[0]), X_train]  # Add bias term
y_train_gd = y_train.reshape(-1, 1)
X_test_gd = np.c_[np.ones(X_test.shape[0]), X_test]

# Train the model
num_features = X_train.shape[1]
W, b = initialize_weights(num_features)
alpha = 0.01  # Learning rate
num_iterations = 1000
W, b, costs = gradient_descent(X_train, y_train_gd, W, b, alpha, num_iterations)

# Plot the cost reduction
plt.plot(range(num_iterations), costs)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost Reduction Over Iterations")
plt.show()

# Predict on the test set
y_pred = predict(X_test_gd, W, b)

# Evaluate the model using R2 score
r2 = r2_score(y_test, y_pred.flatten())
print(f"R2 Score on the test set: {r2:.2f}")
