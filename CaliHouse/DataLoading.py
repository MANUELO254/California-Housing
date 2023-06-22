import pandas as pd

# Read the CSV file into a pandas DataFrame
data = pd.read_excel('C:/Projects/CaliHouse/1553768847_housing.xlsx')

# Print the first few rows of the data
print(data.head())

# Extract input (X) and output (Y) data
X = data.drop('median_house_value', axis=1)  # Input data (all columns except 'median_house_value')
Y = data['median_house_value']  # Output data (the 'median_house_value' column)
