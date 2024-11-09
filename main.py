import pandas
import seaborn as sns
import matplotlib.pyplot as plt


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