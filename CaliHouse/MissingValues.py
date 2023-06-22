import pandas as pd

# Assuming you have already loaded the CSV file into a DataFrame called 'data'
# Fill missing values with the mean of each column
data = pd.read_excel('C:/Projects/CaliHouse/1553768847_housing.xlsx')
data_filled = data.fillna(data.mean())

# Print the first few rows of the updated DataFrame
print(data_filled.head())
