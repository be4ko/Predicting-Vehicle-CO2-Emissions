import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score

# Load the data
data = pd.read_csv("co2_emissions_data.csv")

# Check whether there are missing values
missing_values = data.isna().sum()
print("missing values =", missing_values)

# Check whether numeric features have the same scale
numericData = data.select_dtypes(include=['float64', 'int64'])
print("\nRange of each numeric feature:")
for column in numericData.columns:
    value_range = numericData[column].max() - numericData[column].min()
    print(f"{column}: {value_range}")

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

# Split features and targets for regression model
features_train, features_test, target_train_reg, target_test_reg = train_test_split(
    features, target_regression, test_size=0.2, random_state=42
)

################# Linear Regression #################

# Select features for linear regression
selected_features = ["Engine Size(L)", "Fuel Consumption Comb (L/100 km)", "Cylinders"]
X_train = features_train[selected_features].values
X_test = features_test[selected_features].values
y_train = target_train_reg.values
y_test = target_test_reg.values

class LinearRegression:
    def __init__(self, learning_rate=0.0001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.costs = []
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        self.costs = []
        
        # Training loop
        for _ in np.arange(self.n_iters):
            # Forward pass
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Cost calculation
            cost = np.mean((y_pred - y) ** 2) / 2
            self.costs.append(cost)
            
            # Backward pass (gradient descent)
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Create and train model
model = LinearRegression(learning_rate=0.0001, n_iters=1000)
model.fit(X_train, y_train.reshape(-1, 1))

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
linear_accuracy = r2_score(y_test, y_pred)
print("\n----------------------------------------")
print("Linear Regression Results:")
print("The Accuracy (R² Score) is: ", linear_accuracy)
print("----------------------------------------")

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(np.arange(model.n_iters), model.costs)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost Reduction Over Iterations")
plt.show()

################# Logistic Regression #################

# Prepare data for logistic regression
features_log = data[["Fuel Consumption Comb (L/100 km)", "Engine Size(L)"]]
target_log = data["Emission Class"]

# Encode target
encoder = LabelEncoder()
target_encoded = encoder.fit_transform(target_log)

print("Encoded Classes:", list(encoder.classes_))
print("Example From The Encoded Targets:", target_encoded[:5])

# Split data
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
    features_log, target_encoded, test_size=0.2, random_state=0
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_log)
X_test_scaled = scaler.transform(X_test_log)

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=2000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in np.arange(self.n_iters):
            idx = np.random.randint(0, n_samples)
            xi = X[idx]
            yi = y[idx]
            
            linear_pred = np.dot(xi, self.weights) + self.bias
            pred = self.sigmoid(linear_pred)
            
            dw = xi * (pred - yi)
            db = pred - yi
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        pred = self.sigmoid(linear_pred)
        return (pred >= 0.5).astype(int)


# Create and train logistic regression model
log_model = LogisticRegression(learning_rate=0.01, n_iters=2000)
log_model.fit(X_train_scaled, y_train_log)

# Make predictions
y_pred_log = log_model.predict(X_test_scaled)

# Calculate accuracy
logistic_accuracy = np.mean(y_pred_log == y_test_log)
print("\n----------------------------------------")
print("Logistic Regression Results:")
print("The Accuracy is: ", logistic_accuracy)
print("----------------------------------------")