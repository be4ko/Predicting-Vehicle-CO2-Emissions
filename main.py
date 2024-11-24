import pandas
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

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
features_train, features_test, tareget_train_reg, tareget_test_reg = train_test_split(
    features, target_regression, test_size=0.2, random_state=42
)

# Split features and targets for classification model
_, _, tareget_train_class, tareget_test_class = train_test_split(
    features, target_classification, test_size=0.2, random_state=42
)

# Identify numeric columns
numeric_columns = features.select_dtypes(include=['float64', 'int64']).columns

# Scale features using the training set statistics
scaler = StandardScaler()
features_train[numeric_columns] = scaler.fit_transform(features_train[numeric_columns])  # Fit and transform training data
features_test[numeric_columns] = scaler.transform(features_test[numeric_columns])  # Transform testing data

print("Preprocessing")