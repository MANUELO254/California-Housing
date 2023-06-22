import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
data = pd.read_excel('C:/Projects/CaliHouse/1553768847_housing.xlsx')

# Extract input (X) and output (Y) data
X = data.drop('median_house_value', axis=1)
Y = data['median_house_value']

# Encode categorical column
categorical_column = 'ocean_proximity'
label_encoder = LabelEncoder()
X[categorical_column] = label_encoder.fit_transform(X[categorical_column])

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create a StandardScaler object
scaler = StandardScaler()

# Standardize the training data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Print the shapes of the standardized datasets
print("Training data shape:", X_train_scaled.shape, Y_train.shape)
print("Test data shape:", X_test_scaled.shape, Y_test.shape)
