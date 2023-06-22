import pandas as pd
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
data = pd.read_excel('C:/Projects/CaliHouse/1553768847_housing.xlsx')
data['ocean_proximity_encoded'] = encoder.fit_transform(data['ocean_proximity'])
