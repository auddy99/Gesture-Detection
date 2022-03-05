import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import pickle

df = pd.read_feather('trainingData.feather')
print(df.shape)
# df = df.iloc[: , 2:]
label_encoder = preprocessing.LabelEncoder()

y = label_encoder.fit_transform(df['Label'])
print(y)
y = y.astype('int')
X = df.drop('Label', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,y_train)

predictions = rfc.predict(X_test)
print(classification_report(y_test,predictions))

rfc.fit(X.values,y)

pickle.dump(rfc, open('RFCModel.sav', 'wb'))