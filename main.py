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


################# e) Logistic Regression Using Stochastic Gradient Descent #################

# Selecting two features 
features = data[["Fuel Consumption Comb (L/100 km)", "Engine Size(L)"]]

# Selecting the target column (Emission class)
target = data["Emission Class"]

# Converting the categorical target values (Low, Moderate, High) into numerical values (0, 1, 2)
encoder = LabelEncoder()
target_encoded = encoder.fit_transform(target)

print("Encoded Classes:", list(encoder.classes_))
print("Example From The Encoded Targets:", target_encoded[:5])

# Splitting data with 80% training, 20% test
X_train, X_test, y_train, y_test = train_test_split(features, target_encoded, test_size=0.2, random_state=0)

print("Training Features:", X_train.shape)
print("Testing Features:", X_test.shape)

# Standardize the features to have mean 0 and standard deviation 1
scaler = StandardScaler()

# Fit the scaler on the training data and transform both train and test sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaled Features (Training):", X_train_scaled[:5])

# Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Model Parameters
m, n = X_train_scaled.shape  # Number of training samples and features
theta = np.zeros(n)          # Start with weights of 0
learning_rate = 0.01         # Learning rate for gradient descent
iterations = 2000            # Number of training iterations

# List to store cost
cost_history = []

# Perform training for the specified number of iterations
for iteration in range(iterations):

    random_index = np.random.randint(0, m)  # Randomly choose a sample index
    xi = X_train_scaled[random_index, :]    # Features for the chosen sample
    yi = y_train[random_index]              # True label for the chosen sample

    z = np.dot(xi, theta)          # Weighted sum
    prediction = sigmoid(z)        # Convert to probability

    error = prediction - yi        # Difference between prediction and true label
    gradient = error * xi          # Gradient for the weights
    theta -= learning_rate * gradient  # Update weights using the gradient

    
    if iteration % 100 == 0:
        cost = -yi * np.log(prediction) - (1 - yi) * np.log(1 - prediction)  # Binary cross-entropy loss
        cost_history.append(cost)


# Plotting the cost over time
plt.plot(range(0, iterations, 100), cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Reduction Over Time")
plt.show()

# Using the trained weights to predict test set
z_test = np.dot(X_test_scaled, theta)  # Weighted sums for the test set
predicted_probabilities = sigmoid(z_test) 

threshold = 0.5  # Default threshold used in logistic regression
y_pred_class = predicted_probabilities >= threshold


# Accuracy
accuracy = np.mean(y_pred_class == y_test)
print("The Accuracy in this Logistic Regression is: ", accuracy)


