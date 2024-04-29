import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# menangani data hilang
from sklearn.impute import SimpleImputer

# encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# encoding dependent variable
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('data.csv')
x = dataset.iloc[:, :-1].values # [baris,kolom], variabel X berisi value dari fitur-fitur. -1 yaitu sampai kolom terakhir yang dimana -1 tidak akan dimasukan karena X merupakan data fitur
y = dataset.iloc[:, -1].values  # varibael Y berisii value dari depending variabel. 

print(x)
print(y)
print()

# menangani data hilang
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3]) #melaukuan pembatasan pada area x[:,1:3]
x[:, 1:3] = imputer.transform(x[:, 1:3]) #melakukan perubahan pada area x[:,1:3] dan menempatkannya pada variabel x[:,1:3]
print(x)
print()

# Encoding categorical data (independent variable)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

print(x)
print()

# Encoding dependent variable
le = LabelEncoder()
y = le.fit_transform(y)
print(y)