import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_excel('C:/Projects/CaliHouse/1553768847_housing.xlsx')

# Encode categorical feature
data["ocean_proximity"] = data["ocean_proximity"].astype("category").cat.codes

# Split the data into features and target variable
X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform mean imputation on the training data
imputer = SimpleImputer(strategy="mean")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)

# Perform mean imputation on the test data
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the linear regression model
linear_reg = LinearRegression()
linear_reg.fit(X_train_scaled, y_train)

# Predict on the test data
y_pred = linear_reg.predict(X_test_scaled)

# Calculate the root mean squared error
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Print the root mean squared error
print("Root Mean Squared Error: ", rmse)
