import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle

data=pd.read_csv('weather_classification_data.csv')
print(data)

data.isnull().sum()

X=data.drop(['Weather Type'],axis=1)

from sklearn.pipeline import Pipeline

#Scaling
numerical_cols=['Temperature','Humidity','Wind Speed','Precipitation (%)','Atmospheric Pressure','UV Index','Visibility (km)']
categorical_cols=['Cloud Cover','Season','Location']

numerical_pipeline=Pipeline(steps=[("scaler",StandardScaler())])
categorical_pipeline = Pipeline(steps=[("onehot",OneHotEncoder(handle_unknown='ignore'))])

ct=ColumnTransformer(transformers=[('num',numerical_pipeline,numerical_cols),
                                   ('cate',categorical_pipeline,categorical_cols)])

X_transform=ct.fit_transform(X)

# Save the ColumnTransformer
with open('column_transformer.pkl', 'wb') as file:
    pickle.dump(ct, file)