import numpy as np
import pandas as pd


def get_dataset(dataset, start=None, end=None, lag_max=30):
    """
    Based on the ordinary data from the field, we generate a lot of statistical
     and lag features and return a dataset with a very large number of features.
     
    :param dataset: data with stocks. (pd.DataFrame)
    :param start:   the point from which the output dataset starts to be generated. (int)
    :param end:     the point to which the output dataset will be generated. (int)
    :param lag_max: how much lags it will be generated.
    
    :return: a dataset in which all the necessary features have already been added
        and it remains only to configure training and validation on it. (pd.DataFrame)
    """

    if not start:
        start = 0

    if not end:
        end = dataset.shape[0]

    data_prep = pd.DataFrame(dataset["Open"])
    #data_prep=dataset.copy()
    if end == 0:
        data_prep = data_prep.iloc[start:]
    else:
        data_prep = data_prep.iloc[start:end]

    if lag_max > 2:
        for lag_num in range(5, lag_max, 5):
            data_prep["Open_lag_{}".format(lag_num)] = data_prep['Open'].shift(lag_num)

    #data_prep['year'] = data_prep.index.year
    #new_features = pd.get_dummies(data_prep.index.year)
    #new_features.columns = list(map(lambda x: 'year_' + str(x), new_features.columns))
    #new_features.index = data_prep.index
    #data_prep = pd.concat((data_prep, new_features), axis=1)

    #new_features = pd.get_dummies(data_prep.index.month)
    #new_features.columns = list(map(lambda x: 'month_' + str(x), new_features.columns))
    #new_features.index = data_prep.index
    #data_prep = pd.concat((data_prep, new_features), axis=1)

    #new_features = pd.get_dummies(data_prep.index.week)
    #new_features.columns = list(map(lambda x: 'week_' + str(x), new_features.columns))
    #new_features.index = data_prep.index
    #data_prep = pd.concat((data_prep, new_features), axis=1)

    #new_features = pd.get_dummies(data_prep.index.weekday)
    #new_features.columns = list(map(lambda x: 'dow_' + str(x), new_features.columns))
    #new_features.index = data_prep.index
    #data_prep = pd.concat((data_prep, new_features), axis=1)

    #new_features = pd.get_dummies(data_prep.index.hour)
    #new_features.columns = list(map(lambda x: 'h_' + str(x), new_features.columns))
    #new_features.index = data_prep.index
    #data_prep = pd.concat((data_prep, new_features), axis=1)

    
    #data_prep['day'] = data_prep.index.day

    data_prep.dropna(inplace=True)

    return data_prep


def get_train_test(prepared_dataset, th=90):
    """
    Split dataset in train and test parts.
    
    :param prepared_dataset: a dataset in which all the necessary features have already been
        added and it remains only to configure training and validation on it. (pd.DataFrame)
    :param th: threshold for allocating data for train and test. (int)
    
    :return: two pairs, where arguments and targets are located within the pair,
        the fraction corresponds to the 'th' parameter. (4x np.array)
    """
    
    X_train = prepared_dataset.drop(['Open'], axis=1).values[:-th]
    y_train = prepared_dataset['Open'].values[:-th]

    X_test = prepared_dataset.drop(['Open'], axis=1).values[-th:]
    y_test = prepared_dataset['Open'].values[-th:]

    return X_train, y_train, X_test, y_test


def mape(y_true, y_pred):
    """
    Mean absolute percentage error.
    
    :param y_true: ground truth (correct) target values. (one dim array)
    :param y_pred: estimated target values. (one dim array)    
    
    :return: the value of mape score between y_true and y_prediction.
    """ 
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    assert y_true.shape == y_pred.shape
    return np.mean(np.abs((y_true - y_pred) / y_true))




