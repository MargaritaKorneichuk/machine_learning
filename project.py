import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import csv

from sklearn.metrics import r2_score
import itertools
from sklearn.metrics  import mean_squared_error as mse

# Splitting data into training and testing
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 60)

# Matplotlib for visualization
%matplotlib inline

# Set default font size
plt.rcParams['font.size'] = 24

from IPython.core.pylabtools import figsize

# Seaborn for visualization
import seaborn as sns
sns.set(font_scale = 2)

# Machine Learning Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
df=pd.read_csv('eurrub_data.csv')
df['Datetime']=df['Date']+"."+df['Time']
df['Datetime']=pd.to_datetime(df['Datetime'],format="%d/%m/%Y.%H:%M:%S")
df=df.set_index('Datetime')
df=df.drop('Date',axis=1)
df=df.drop('Time',axis=1)
df=df.drop('Vol',axis=1)
plt.figure(figsize=(11, 8)) # resizing the plot
df["Open"].plot()
plt.title('Open Price History') # adding a title
plt.xlabel('Date') # x label
plt.ylabel('Open') # y label
plt.show()
df1=df[-83:]
def norm(data):
  for i in range(1,data.shape[0]-1):
    data['High'][i]=(data['High'][i-1]+data['Low'][i+1])/2
  for i in range(1,data.shape[0]-1):
    data['Low'][i]=(data['Low'][i-1]+data['Low'][i+1])/2
  for i in range(1,data.shape[0]-1):
    data['Close'][i]=(data['Close'][i-1]+data['Close'][i+1])/2
  for i in range(1,data.shape[0]-1):
    data['Open'][i]=(data['Open'][i-1]+data['Open'][i+1])/2
  return data

df1=norm(df1)
df1=norm(df1)

plt.figure(figsize=(11, 8)) # resizing the plot
df1["Open"].plot()
plt.title('Open Price History') # adding a title
plt.xlabel('Date') # x label
plt.ylabel('Open') # y label
plt.show()

import math
df2=df1.copy()
forecast_out = 4 # forcasting out 5% of the entire dataset
print(forecast_out)
df2['label'] = df1['Open'].shift(-forecast_out)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

scaler = StandardScaler()
X = np.array(df2.drop(['label'], 1))
scaler.fit(X)
X = scaler.transform(X)

X_Predictions = X[-forecast_out:] # data to be predicted
X = X[:-forecast_out] # data to be trained
df2.dropna(inplace=True)
y = np.array(df2['label'])
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.2,random_state=42)
from sklearn.metrics import mean_squared_error,mean_absolute_error

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_confidence = lr.score(X_test, y_test)
y_pred=lr.predict(X_test)
lr_mse=mean_squared_error(y_test,y_pred)
lr_mae=mean_absolute_error(y_test,y_pred)
#pred=lr.predict(X_Predictions)
print(lr_confidence,lr_mse,lr_mae)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_confidence = rf.score(X_test, y_test)
y_pred=rf.predict(X_test)
rf_mse=mean_squared_error(y_test,y_pred)
rf_mae=mean_absolute_error(y_test,y_pred)
#pred=rf.predict(X_Predictions)
print(rf_confidence,rf_mse,rf_mae)

svr = SVR()
svr.fit(X_train, y_train)
svr_confidence = svr.score(X_test, y_test)
y_pred=svr.predict(X_test)
svr_mse=mean_squared_error(y_test,y_pred)
svr_mae=mean_absolute_error(y_test,y_pred)
#pred=svr.predict(X_Predictions)
print(svr_confidence,svr_mse,svr_mae)

knn=KNeighborsRegressor()
knn.fit(X_train,y_train)
knn_confidence=knn.score(X_test,y_test)
y_pred=knn.predict(X_test)
knn_mse=mean_squared_error(y_test,y_pred)
knn_mae=mean_absolute_error(y_test,y_pred)
#pred=knn.predict(X_Predictions)
print(knn_confidence,knn_mse,knn_mae)

gr=GradientBoostingRegressor()
gr.fit(X_train,y_train)
gr_confidence=gr.score(X_test,y_test)
y_pred=gr.predict(X_test)
gr_mse=mean_squared_error(y_test,y_pred)
gr_mae=mean_absolute_error(y_test,y_pred)
#pred=rf.predict(X_Predictions)
print(gr_confidence,gr_mse,gr_mae)
