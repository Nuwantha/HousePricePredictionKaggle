import numpy as np 
import pandas as pd

from sklearn.ensemble import BaggingRegressor
import matplotlib.pyplot as plt
from  sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor

data_frame = pd.read_csv('train.csv') 
data_frame_test=pd.read_csv('test.csv') 

categorical_fields=['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle',
                    'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
                    'BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual',
                    'GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition'
                    ]

numerical_fields=['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
                  'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
                  'TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea',
                  'MiscVal','MoSold','YrSold']

ally_dict={'NA':0,'Grvl':1,'Pave':2} 

def pre_process_data():
    for col in categorical_fields:
        data_frame[col].fillna('default',inplace=True)
        data_frame_test[col].fillna('default',inplace=True)

    for col in numerical_fields:
        data_frame[col].fillna(0,inplace=True)
        data_frame_test[col].fillna(0,inplace=True)

    encode=LabelEncoder()
    for col in categorical_fields:
        data_frame[col]=encode.fit_transform(data_frame[col])
        data_frame_test[col]=encode.fit_transform(data_frame_test[col])
    data_frame['SalePrice'].fillna(0,inplace=True)

def get_feature_importance(list_of_features):
    n_estimators=10000
    random_state=0
    n_jobs=4
    x_train=data_frame[list_of_features]
    y_train=data_frame.iloc[:,-1]
    feat_labels= data_frame.columns[1:]
    forest = BaggingRegressor(n_estimators=n_estimators,random_state=random_state,n_jobs=n_jobs) 
    forest.fit(x_train,y_train) 
    importances=forest.feature_importances_ 
    indices = np.argsort(importances)[::-1]


    for f in range(x_train.shape[1]):
        print("%2d) %-*s %f" % (f+1,30,feat_labels[indices[f]],
                                        importances[indices[f]]))

   
    plt.title("Feature Importance")
    plt.bar(range(x_train.shape[1]),importances[indices],color='lightblue',align='center')
    plt.xticks(range(x_train.shape[1]),feat_labels[indices],rotation=90)
    plt.xlim([-1,x_train.shape[1]])
    plt.tight_layout()
    plt.show()  


def get_features():
    l1=list(data_frame.columns.values)
    l2=list(data_frame_test.columns.values)
    fields=list(set(l1)& set(l2))
    fields.remove('Id')
    list_of_features=sorted(fields)
    return list_of_features

def create_model(list_of_features):

    n_estimators=10000 
    n_jobs=4 
    x_train=data_frame[list_of_features] 
    y_train=data_frame.iloc[:,-1]
    x_test=data_frame_test[list_of_features] 
    random_state=0

    forest=BaggingRegressor(base_estimator=DecisionTreeRegressor(),n_estimators=n_estimators,random_state=random_state, n_jobs=n_jobs)
    forest.fit(x_train[list_of_features],y_train)
    Y_pred=forest.predict(data_frame_test[list_of_features].as_matrix()) 

    i=0
    file=open('submission.csv','w')
    header="Id,SalePrice"
    header=header+'\n'
    file.write(header)
    for id in (data_frame_test['Id']):
        str="{},{}".format(id,Y_pred[i])
        str=str+'\n'
        file.write(str)
        i+=1




def main():
    pre_process_data()
    list_of_features=get_features()
    get_feature_importance(list_of_features)
    create_model(list_of_features)


if __name__==main():
    main()
