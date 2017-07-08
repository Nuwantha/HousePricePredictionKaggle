# -*- coding: utf-8 -*-
"""
Created on Sun May 28 13:27:25 2017

@author: H.P. Asela
"""

import numpy as np
import pandas as pd
from scipy.stats import skew
from keras.models import Sequential
from keras.layers import Dense

train = pd.read_csv("D:/train.csv") # read train data
test = pd.read_csv("D:/test.csv") # read test data

        
outlier_idx = [4,11,13,20,46,66,70,167,178,185,199, 224,261, 309,313,318, 349,412,423,440,454,477,478, 523,540, 581,588,595,654,688, 691, 774, 798, 875, 898,926,970,987,1027,1109, 1169,1182,1239, 1256,1298,1324,1353,1359,1405,1442,1447]
train.drop(train.index[outlier_idx],inplace=True)
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],test.loc[:,'MSSubClass':'SaleCondition']))
to_delete = ['Alley','FireplaceQu','PoolQC','Fence','MiscFeature']
all_data = all_data.drop(to_delete,axis=1)
train["SalePrice"] = np.log1p(train["SalePrice"])
#log transform skewed numeric features
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]

y = train.SalePrice

model = Sequential()
model.add(Dense(100, input_dim=268, kernel_initializer='normal', activation='relu'))
model.add(Dense(40, kernel_initializer='normal', activation='relu'))
model.add(Dense(20, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(np.array(X_train),np.array(y), nb_epoch=20, batch_size=50)
predictions = model.predict(np.array(X_test))
np.savetxt('predict.csv', predictions, delimiter="\t")