#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import random
import datetime
import joblib
import tensorflow as tf
import dill # To save workspace
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import Adam
from keras import backend as K
from keras import regularizers
import keras.metrics
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.utils import class_weight
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


# In[3]:


test_data = pd.read_csv(test_features_link)
test_data.head()


# In[4]:


test_data = test_data[test_data.second < 600000000].sort_values(['bookingID', 'second']).reset_index(drop = True)
test_data = test_data[test_data.Speed <= 30].sort_values(['bookingID', 'second']).reset_index(drop = True) 
test_data = test_data[test_data.Accuracy <= 1500].sort_values(['bookingID', 'second']).reset_index(drop = True) 

test_data = test_data.fillna(0)
test_data['positive_speed'] = [1 if speed >= 0 else 0 for speed in test_data.Speed]
test_data['imputed_speed'] = test_data.Speed * test_data.positive_speed


# ## 2. Sequential Model:

# In[5]:


test_data['acceleration_total'] = (test_data.acceleration_x ** 2 +                                   test_data.acceleration_y ** 2 +                                   test_data.acceleration_z ** 2) **0.5

test_data['gyro_total'] = (test_data.gyro_x ** 2 +                           test_data.gyro_y ** 2 +                           test_data.gyro_z ** 2) **0.5


# In[6]:


def make_timestamp (second):
    return pd.Timestamp('2018-01-01') + datetime.timedelta(seconds = second)

test_data['timestamp'] = test_data.second.apply(make_timestamp)
id_group = test_data.groupby('bookingID').rolling('30s', on = 'timestamp', min_periods = 1)
test_data['rolling_acc_total'] = id_group.acceleration_total.mean().reset_index(drop=True)
test_data['rolling_acc_x'] = id_group.acceleration_x.mean().reset_index(drop=True)
test_data['rolling_acc_y'] = id_group.acceleration_y.mean().reset_index(drop=True)
test_data['rolling_acc_z'] = id_group.acceleration_z.mean().reset_index(drop=True)
test_data['rolling_gyro_total'] = id_group.gyro_total.mean().reset_index(drop=True)
test_data['rolling_gyro_x'] = id_group.gyro_x.mean().reset_index(drop=True)
test_data['rolling_gyro_y'] = id_group.gyro_y.mean().reset_index(drop=True)
test_data['rolling_gyro_z'] = id_group.gyro_z.mean().reset_index(drop=True)
test_data['rolling_speed'] = id_group.imputed_speed.mean().reset_index(drop=True)


# In[7]:


bearing_change_clockwise = abs(test_data.Bearing.diff()) / test_data.second.diff()
bearing_change_anticlockwise = (360 - abs(test_data.Bearing.diff())) / test_data.second.diff()
test_data['bearing_change'] = np.where(bearing_change_clockwise <= bearing_change_anticlockwise, bearing_change_clockwise, bearing_change_anticlockwise)
test_data.loc[test_data.second == 0, 'bearing_change'] = 0
del bearing_change_clockwise, bearing_change_anticlockwise
test_data['rolling_bearing_change'] = test_data.groupby('bookingID').rolling('30s', on = 'timestamp', min_periods = 1).bearing_change.mean().reset_index(drop=True)


# In[8]:


turning_indicator = test_data.bearing_change / 180
test_data['speed_during_turning'] = turning_indicator * test_data.imputed_speed
del turning_indicator

turning_indicator_x = abs(test_data.gyro_x) / 48.46
test_data['speed_during_turning_x'] = turning_indicator_x * test_data.imputed_speed

turning_indicator_y = abs(test_data.gyro_y) / 80.3
test_data['speed_during_turning_y'] = turning_indicator_y * test_data.imputed_speed

turning_indicator_z = abs(test_data.gyro_z) / 66.3
test_data['speed_during_turning_z'] = turning_indicator_z * test_data.imputed_speed
del turning_indicator_x, turning_indicator_y, turning_indicator_z


# In[9]:


min_speed = 7.5697*10**(-27)
test_data['speed_pct_change'] = (test_data.imputed_speed + min_speed).pct_change() / test_data.second.diff()
test_data.loc[test_data.second == 0, 'speed_pct_change'] = 0
del min_speed

# Raw changeof acceleration:
test_data['accel_x_raw_change'] = test_data.acceleration_x.diff() / test_data.second.diff()
test_data.loc[test_data.second == 0, 'accel_x_raw_change'] = 0
test_data['accel_y_raw_change'] = test_data.acceleration_y.diff() / test_data.second.diff()
test_data.loc[test_data.second == 0, 'accel_y_raw_change'] = 0
test_data['accel_z_raw_change'] = test_data.acceleration_z.diff() / test_data.second.diff()
test_data.loc[test_data.second == 0, 'accel_z_raw_change'] = 0
test_data['accel_total_raw_change'] = test_data.acceleration_total.diff() / test_data.second.diff()
test_data.loc[test_data.second == 0, 'accel_total_raw_change'] = 0
test_data[['imputed_speed', 'speed_pct_change']].head()


# In[10]:


test_data['high_accuracy'] = np.where(test_data.Accuracy > 15, 1, 0)


# In[14]:


n_timestamps = 50

first_obs_index = test_data.groupby('bookingID').second.idxmin()
last_obs_index = test_data.groupby('bookingID').second.idxmax()


lstm_sample = []
for idx in test_data.bookingID.unique():
    temp_array = np.linspace(first_obs_index[idx], last_obs_index[idx], n_timestamps).astype('int')
    lstm_sample.extend(temp_array)
    

test_data_lstm = test_data.loc[lstm_sample,:].sort_values(['bookingID', 'second']).reset_index(drop=True)
test_data_lstm[['bookingID', 'second']].head(15)


# In[15]:


not_useful_col = test_data_lstm.columns[test_data_lstm.columns.isin(['second', 'bookingID', 'high_accuracy', 'Accuracy', 'Bearing', 'Speed', 'timestamp',                                                       'speed_during_turning_x', 'speed_during_turning_y', 'speed_during_turning_z', 'speed_during_turning',                                                       'speed_during_high_accuracy', 'accel_x_raw_change', 'accel_y_raw_change', 'accel_z_raw_change',                                                       'acceleration_x', 'acceleration_y', 'acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z'                                                      ])]
test_data_lstm_dropped = test_data_lstm.drop(not_useful_col, 1)


scaler = joblib.load('./models/seq_scaler.pkl')
test_data_lstm_scaled = scaler.transform(test_data_lstm_dropped)


n_obs = len(test_data_lstm.bookingID.unique())
n_cols = test_data_lstm_scaled.shape[1]
test_data_lstm_reshaped = np.array(test_data_lstm_scaled)
test_data_lstm_reshaped = test_data_lstm_reshaped.reshape(n_obs, n_timestamps, n_cols)


# In[17]:


# Define optimizer:
adam_opt = Adam(lr=0.002, decay=0)

# Define model network
# As we have lots of features, we include drop out and L2 regularizers.
lstm_model = Sequential()
lstm_model.add(LSTM(32, input_shape=(n_timestamps, n_cols), kernel_regularizer=regularizers.l2(0.005), return_sequences=True))
lstm_model.add(Dropout(0.4))
lstm_model.add(LSTM(8, kernel_regularizer=regularizers.l2(0.005), return_sequences=True))
lstm_model.add(Dropout(0.4))
lstm_model.add(LSTM(16, kernel_regularizer=regularizers.l2(0.005)))
lstm_model.add(Dropout(0.4))
lstm_model.add(Dense(1, activation='sigmoid'))
lstm_model.compile(loss='binary_crossentropy', optimizer=adam_opt, metrics=['accuracy'])


lstm_model.load_weights('./models/best_final_lstm_model.hdf5')
lstm_preds = lstm_model.predict(test_data_lstm_reshaped)
lstm_preds = pd.Series([x[0] for x in lstm_preds], index = test_data_lstm.bookingID.unique(), name = 'lstm_preds')
lstm_preds.head()


# ## Non-sequential Model:

# In[18]:


test_data_nonseq = test_data.drop(['timestamp', 'Accuracy', 'Speed', 'second', 'positive_speed', 'Bearing', 'high_accuracy'], axis = 1)
test_data_nonseq = test_data_nonseq.set_index('bookingID')


# In[19]:


from scipy.stats import kurtosis, skew

def semi_std(series):
    mean = series.mean()
    high_val = series[series >= mean]
    n_obs = len(high_val)
    if n_obs == 1: return 0
    var = np.sum((high_val - mean)**2) / (n_obs - 1)
    return (var**0.5)


def mean_of_extreme_values(series):
    q95 = series.quantile(0.95, interpolation='midpoint')
    return series[series >= q95].mean()


def summary_statstics(x):
    result = []
    colnames = x.columns[x.columns != 'bookingID']
    agg_funcs = ['min', 'max', 'mean', 'median', 'std', 'q25', 'q75', 'skew', 'kurtosis', 'semi_std', 'mean_of_extreme_values']

    for col in colnames:
        if col in ['speed_during_turning', 'imputed_speed']:
            values = x[col][x[col] != 0]
        else:
            values = x[col]
        
        if values.empty:
            result.extend([0] * len(agg_funcs))
        else:
            result.append(values.min())
            result.append(values.max())        
            result.append(values.mean())
            result.append(values.quantile(0.5))
            result.append(np.std(values))      
            result.append(values.quantile(0.25))
            result.append(values.quantile(0.75))
            result.append(skew(values, bias = False))
            result.append(kurtosis(values, bias = False))
            result.append(semi_std(values))
            result.append(mean_of_extreme_values(values))
    
    return pd.Series(result, index=[[colname for colname in colnames.to_list() for repeat in range(len(agg_funcs))],
                                    [agg_func for repeat in range(len(colnames)) for agg_func in agg_funcs]])

test_data_nonseq_agg = test_data_nonseq.groupby(level = 0).apply(summary_statstics)
test_data_nonseq_agg.columns = test_data_nonseq_agg.columns.map('|'.join).str.strip('|')
test_data_nonseq_agg = test_data_nonseq_agg.sort_index()


# In[20]:


mode_speed = pd.Series(test_data[test_data.imputed_speed > 0].groupby('bookingID').imputed_speed.agg(lambda x: pd.Series.mode(x.astype('int'))[0]), name = 'imputed_speed|mode')
test_data_nonseq_agg = pd.concat([test_data_nonseq_agg, mode_speed], axis = 'columns', join_axes = [test_data_nonseq_agg.index])
del mode_speed
test_data_nonseq_agg['imputed_speed|mode'] = test_data_nonseq_agg['imputed_speed|mode'].fillna(0)


# In[21]:


trip_length = pd.Series(test_data.groupby('bookingID').second.max(), name = 'trip_length')
test_data_nonseq_agg = pd.concat([test_data_nonseq_agg, trip_length], axis = 'columns', join_axes = [test_data_nonseq_agg.index])
test_data_nonseq_agg['triplength_x_speed'] = test_data_nonseq_agg.trip_length * test_data_nonseq_agg['imputed_speed|mean']
del trip_length


# In[22]:


perc_negative_speed = pd.Series((1 - test_data.groupby('bookingID').positive_speed.mean()), name = 'perc_negative_speed')
test_data_nonseq_agg = pd.concat([test_data_nonseq_agg, perc_negative_speed], axis = 'columns', join_axes = [test_data_nonseq_agg.index])

perc_high_accuracy = pd.Series((1 - test_data.groupby('bookingID').high_accuracy.mean()), name = 'perc_high_accuracy')
test_data_nonseq_agg = pd.concat([test_data_nonseq_agg, perc_high_accuracy], axis = 'columns', join_axes = [test_data_nonseq_agg.index])

del perc_negative_speed, perc_high_accuracy


# In[23]:


accuracy_65 = pd.Series(np.where(test_data.Accuracy == 65, 1, 0))
wifi_gps = pd.Series(accuracy_65.groupby(test_data.bookingID).mean(), name = 'wifi_gps')
test_data_nonseq_agg = pd.concat([test_data_nonseq_agg, wifi_gps], axis = 'columns', join_axes = [test_data_nonseq_agg.index])

accuracy_1414 = pd.Series(np.where(test_data.Accuracy == 1414, 1, 0))
cell_gps = pd.Series(accuracy_1414.groupby(test_data.bookingID).mean(), name = 'cell_gps')
test_data_nonseq_agg = pd.concat([test_data_nonseq_agg, cell_gps], axis = 'columns', join_axes = [test_data_nonseq_agg.index])


# In[26]:


time_interval = test_data.second.diff()
time_interval[test_data.second == 0] = 0
max_time_interval = pd.Series(time_interval.groupby(test_data.bookingID).max(), name = 'max_time_interval')
test_data_nonseq_agg = pd.concat([test_data_nonseq_agg, max_time_interval], axis = 'columns', join_axes = [test_data_nonseq_agg.index])
del time_interval, max_time_interval
test_data_nonseq_agg['max_time_interval'].head()


# In[27]:


from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

preProcess = joblib.load('./models/nonseq_scaler.pkl')
train_features = preProcess.transform(test_data_nonseq_agg)


# In[32]:


final_knn_model = joblib.load('./models/final_knn_model.pkl')
full_knn_preds = final_knn_model.predict_proba(train_features)
full_knn_preds = [x[1] for x in full_knn_preds]
full_knn_preds = pd.Series(full_knn_preds, name = 'full_knn_preds', index = test_data_nonseq_agg.index)

final_enet_model = joblib.load('./models/final_enet_model.pkl')
full_enet_preds = final_enet_model.predict_proba(train_features)
full_enet_preds = [x[1] for x in full_enet_preds]
full_enet_preds = pd.Series(full_enet_preds, name = 'full_enet_preds', index = test_data_nonseq_agg.index)

final_rf_model = joblib.load('./models/final_rf_model.pkl')
full_rf_preds = final_rf_model.predict_proba(train_features)
full_rf_preds = [x[1] for x in full_rf_preds]
full_rf_preds = pd.Series(full_rf_preds, name = 'full_rf_preds', index = test_data_nonseq_agg.index)

final_xgb_model = joblib.load('./models/final_xgb_model.pkl')
full_xgb_preds = final_xgb_model.predict_proba(train_features)
full_xgb_preds = [x[1] for x in full_xgb_preds]
full_xgb_preds = pd.Series(full_xgb_preds, name = 'full_xgb_preds', index = test_data_nonseq_agg.index)


# In[35]:


full_stacked_model_data = pd.concat([lstm_preds, full_knn_preds, full_enet_preds, full_rf_preds, full_xgb_preds], axis = 'columns', join_axes = [test_data_nonseq_agg.index])
final_stacked_model = joblib.load('./models/final_stacked_model.pkl')
full_stacked_pred_labels = pd.Series(final_stacked_model.predict(full_stacked_model_data), name = 'full_stacked_pred_labels', index = test_data_nonseq_agg.index)

full_stacked_pred_proba = final_stacked_model.predict_proba(full_stacked_model_data)
full_stacked_pred_proba = [x[1] for x in full_stacked_pred_proba]
full_stacked_pred_proba = pd.Series(full_stacked_pred_proba, name = 'full_stacked_pred_proba', index = test_data_nonseq_agg.index)

full_stacked_preds = pd.concat([full_stacked_pred_labels, full_stacked_pred_proba], axis = 'columns', join_axes = [test_data_nonseq_agg.index])
full_stacked_preds.to_csv('safety_predictions.csv')

