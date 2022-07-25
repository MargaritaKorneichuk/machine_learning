!pip install yfinance
!pip install statsmodels
!pip install arch
!pip install keras

import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 

from tqdm import tqdm_notebook as tqdm

from data_preparation import *

from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import itertools
from sklearn.metrics  import mean_squared_error as mse

def get_stocks(ticker, start = "1900-01-01", end = None):
    tickerData = yf.Ticker(tickerSymbol)
    df = tickerData.history(period='1d', start=start, end=end)
    if "Stock Splits" in df.columns:
        df.drop(["Stock Splits"], axis = 1, inplace = True)

    if "Dividends" in df.columns:
        df.drop(["Dividends"], axis = 1, inplace = True)
    #if "Volume" in df.columns:
        #df.drop(["Volume"], axis = 1, inplace = True)
    #if "High" in df.columns:
        #df.drop(["High"], axis = 1, inplace = True)
    #if "Low" in df.columns:
        #df.drop(["Low"], axis = 1, inplace = True)
    
    return df

tickerSymbol = 'MSFT'
df = get_stocks(tickerSymbol, end = "2022-05-10")
df

print('Кол-во строк: ',        df.shape[0])
print('Кол-во столбцов: ' , df.shape[1], end='\n\n')

df.apply(lambda x: x == 0).sum()
df.isnull().sum() 
def fill_missing(df):
    new = df.copy()
    idx = pd.date_range(new.index[0], new.index[-1])
    new = new.reindex(idx, fill_value=0)
    n = new.shape[0]

    for i in range(1, n):
        if new.iloc[i, :].sum() == 0:
            new.iloc[i, :] = new.iloc[i-1, :]

    return new
df = fill_missing(df)

def plot_open(df, tickerSymbol, start = None, end = None):
    
    if not start:
        start = df.index[0]
    
    if not end:
        end = df.index[-1]
    plt.figure(figsize=(24, 10))
    plt.title("Цена открытия акций " + tickerSymbol, fontsize = 20)
    plt.plot(df["Open"][start:end])
    plt.grid()
    plt.xlabel('Дата', fontsize=15)
    plt.ylabel("Цена", fontsize=15)
    plt.show()

plot_open(df, tickerSymbol, "2010")

years = 5
forward=40
lag_max=30

data_prep = get_dataset(df,start=-years*365, lag_max = lag_max)
data_prep

def smooth(df, ws = 7):
    df_upd = np.copy(df)

    for i in range(ws//2, len(df_upd) - ws//2):
        df_upd[i] = np.median(df_upd[i-ws//2:i+ws//2])
    return df_upd
def visualization_result(X_train,y_train, X_test,y_test, predict, name, forward, ws = None):
    if ws:
        y_train = smooth(y_train, ws) 
        y_test = smooth(y_test, ws) 
        predict = smooth(predict, ws) 
    plt.rc('figure', figsize=(20, 8))
    plt.plot(X_train[-forward*2:], y_train[-forward*2:], color = "g", label = "Train")
    plt.plot(X_test, y_test, label = "Test")
    plt.plot(X_test[2:], predict, lw=5, label = "Prediction")
    plt.grid()
    plt.title(f"{name}", fontsize = 20)
    plt.xlabel('Дата', fontsize=15)
    plt.ylabel("Цена", fontsize=15)
    plt.legend(prop={'size': 20})
    plt.show()


from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import math
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

np.random.seed(7)
daf = pd.DataFrame(data_prep["Open"].copy())
dataset = daf.values
dataset
dataset = dataset.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
test_size = forward
train_size = len(dataset) - test_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict2 = scaler.inverse_transform(trainPredict)
trainY2 = scaler.inverse_transform([trainY])
testPredict2 = scaler.inverse_transform(testPredict)
testY2 = scaler.inverse_transform([testY])
trainScore = math.sqrt(mse(trainY2[0], trainPredict2[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mse(testY2[0], testPredict2[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
r2=r2_score(testY2[0],testPredict2[:,0])
mape_res=mape(testY2[0],testPredict2[:,0])
print(r2)
print(mape_res)


def all_comb(params):
    keys = params.keys()
    values = (params[key] for key in keys)
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    return combinations

X_train, y_train, X_test, y_test = get_train_test(data_prep, forward)
train_ind, test_ind = data_prep.index[:-forward], data_prep.index[-forward:]

from sklearn.linear_model import LinearRegression

Lin = LinearRegression()

Lin.fit(X_train, y_train)

y_pred = Lin.predict(X_test)
result = mape(y_test, y_pred)

print('Test:')
print(f'mape - {result}', end='\n\n')
history_mape.append(result)

coef = pearsonr(y_test, y_pred)[0]
print(f"Коэффициент корреляции Пирсона: {coef}")
history_coef.append(coef)

r2 = r2_score(y_test, y_pred)
print(f"R^2 score(coefficient of determination): {r2} ")
history_r2.append(r2)

visualization_result(train_ind, y_train, test_ind, y_test, y_pred, "LinearRegression", forward, 4)

from sklearn.ensemble import RandomForestRegressor

RF = RandomForestRegressor(bootstrap=False,max_depth= 110,max_features='auto',min_samples_leaf= 1,min_samples_split=2,n_estimators=100)
RF.fit(X_train, y_train)

y_pred = RF.predict(X_test)

result = mape(y_test, y_pred)

print('Test:')
print(f'mape - {result}', end='\n\n')

history_mape.append(result)

coef = pearsonr(y_test, y_pred)[0]
print(f"Коэффициент корреляции Пирсона: {coef}")
history_coef.append(coef)

r2 = r2_score(y_test, y_pred)
print(f"R^2 score(coefficient of determination): {r2} ")
history_r2.append(r2)

visualization_result(train_ind, y_train, test_ind, y_test, y_pred, "RandomForestRegressor", forward,3)

from sklearn.ensemble import GradientBoostingRegressor

GBR = GradientBoostingRegressor(learning_rate= 0.4,loss= 'quantile',max_depth= 10,n_estimators= 400,subsample= 0.5)
GBR.fit(X_train, y_train)

y_pred = GBR.predict(X_test)

result = mape(y_test, y_pred)

print('Test:')
print(f'mape - {result}', end='\n\n')

history_mape.append(result)


coef = pearsonr(y_test, y_pred)[0]
print(f"Коэффициент корреляции Пирсона: {coef}")
history_coef.append(coef)


r2 = r2_score(y_test, y_pred)
print(f"R^2 score(coefficient of determination): {r2} ")
history_r2.append(r2)

visualization_result(train_ind, y_train, test_ind, y_test, y_pred, "GradientBoostingRegressor", forward, 3)

from sklearn.svm import SVR

sv = SVR(C= 100, gamma= 0.001, kernel= 'rbf')
sv.fit(X_train, y_train)

y_pred = sv.predict(X_test)

result = mape(y_test, y_pred)

print('Test:')
print(f'mape - {result}', end='\n\n')

history_mape.append(result)

coef = pearsonr(y_test, y_pred)[0]
print(f"Коэффициент корреляции Пирсона: {coef}")
history_coef.append(coef)

r2 = r2_score(y_test, y_pred)
print(f"R^2 score(coefficient of determination): {r2} ")
history_r2.append(r2)

visualization_result(train_ind, y_train, test_ind, y_test, y_pred, "SVR", forward, 4)

from sklearn.neighbors    import KNeighborsRegressor

KN = KNeighborsRegressor(n_neighbors= 6, p= 3, weights= 'uniform')
KN.fit(X_train, y_train)

y_pred = KN.predict(X_test)

result = mape(y_test, y_pred)

print('Test:')
print(f'mape - {result}', end='\n\n')

history_mape.append(result)

r2 = r2_score(y_test, y_pred)
print(f"R^2 score(coefficient of determination): {r2} ")
history_r2.append(r2)

visualization_result(train_ind, y_train, test_ind, y_test, y_pred, "KNeighborsRegressor", forward, 4)

objects = ['LinearRegression', 'RandomForestRegressor  ', 'GradientBoostingRegressor', 
            'SVR', 'KNeighborsRegressor']

y_pos = np.arange(len(objects))
colors = ["r", "g", "b", "m", "yellow", "darkorange" , 'aqua']
models, values = list(zip(*sorted(zip(objects, history_mape), key = lambda x: -x[1])))
plt.figure(figsize=(18,10))
plt.bar(y_pos, values, align='center', alpha=0.5, color =colors)
plt.xticks(y_pos, models, fontsize = 12)
plt.ylabel('MAPE')
plt.grid()
plt.title('Сравнение ошибок моделей', fontsize = 16)

plt.show()

objects = ['LinearRegression', 'RandomForestRegressor  ', 'GradientBoostingRegressor', 
            'SVR', 'KNeighborsRegressor']

y_pos = np.arange(len(objects))
colors = ["r", "g", "b", "m", "yellow", "darkorange", "lime", "aqua"]
models, values = list(zip(*sorted(zip(objects, history_coef), key = lambda x: x[1])))
plt.figure(figsize=(18,10))
plt.bar(y_pos, values, align='center', alpha=0.5, color =colors)
plt.xticks(y_pos, models, fontsize = 12, rotation = 15)
plt.ylabel('pearson')
plt.grid()
plt.title('Сравнение коэффициентов корреляции Пирсона для каждой модели', fontsize = 16)

plt.show()

objects = ['LinearRegression', 'RandomForestRegressor  ', 'GradientBoostingRegressor', 
            'SVR', 'KNeighborsRegressor']

y_pos = np.arange(len(objects))
colors = ["r", "g", "b", "m", "yellow", "darkorange", "lime", "aqua"]
models, values = list(zip(*sorted(zip(objects, history_r2), key = lambda x: x[1])))
plt.figure(figsize=(18,10))
plt.bar(y_pos, values, align='center', alpha=0.5, color =colors)
plt.xticks(y_pos, models, fontsize = 12, rotation = 15)
plt.ylabel('r2', fontsize=16)
plt.grid()
plt.title('Сравнение коэффициентов детерминации для каждой модели', fontsize = 16)

plt.show()