from google.colab import drive
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

drive.mount("/content/gdrive")

pd.options.display.max_rows = 4000

location_dataset = "gdrive/My Drive/Datasets/house-prices-advanced-regression-techniques"
location_trainData = location_dataset + "/train.csv"
trainData = pd.read_csv(location_trainData)

#print(trainData)


'''
location_dataset = "C://Users//shad_//Desktop//2. Fall 2020//CSE 6023 (Machine " \
                   "Learning)//Submission//Project//_assets//Dataset//" \
                   "house-prices-advanced-regression-techniques"
'''

location_trainData = location_dataset + "//train.csv"
location_testData = location_dataset + "//test.csv"

trainData = pd.read_csv(location_trainData)
testData = pd.read_csv(location_testData)


trainLabel = trainData["SalePrice"]

Ids = testData['Id']
Ids = pd.DataFrame(Ids)

trainData.drop(['Id', 'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature', 'SalePrice'], inplace=True, axis=1)

trainData["LotFrontage"] = trainData["LotFrontage"].fillna(trainData["LotFrontage"].mean())
trainData["MasVnrArea"] = trainData["MasVnrArea"].fillna(trainData["MasVnrArea"].mean())

trainData["BsmtQual"] = trainData["BsmtQual"].fillna(trainData["BsmtQual"].mode()[0])
trainData["BsmtCond"] = trainData["BsmtCond"].fillna(trainData["BsmtCond"].mode()[0])
trainData["BsmtExposure"] = trainData["BsmtExposure"].fillna(trainData["BsmtExposure"].mode()[0])
trainData["BsmtFinType1"] = trainData["BsmtFinType1"].fillna(trainData["BsmtFinType1"].mode()[0])
trainData["BsmtFinType2"] = trainData["BsmtFinType2"].fillna(trainData["BsmtFinType2"].mode()[0])
trainData["GarageType"] = trainData["GarageType"].fillna(trainData["GarageType"].mode()[0])
trainData["GarageYrBlt"] = trainData["GarageYrBlt"].fillna(trainData["GarageYrBlt"].mode()[0])
trainData["GarageFinish"] = trainData["GarageFinish"].fillna(trainData["GarageFinish"].mode()[0])
trainData["GarageQual"] = trainData["GarageQual"].fillna(trainData["GarageQual"].mode()[0])
trainData["GarageCond"] = trainData["GarageCond"].fillna(trainData["GarageCond"].mode()[0])
trainData["MasVnrType"] = trainData["MasVnrType"].fillna(trainData["MasVnrType"].mode()[0])
trainData["Electrical"] = trainData["Electrical"].fillna(trainData["Electrical"].mode()[0])


testData.drop(['Id', 'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], inplace=True, axis=1)

testData["LotFrontage"] = testData["LotFrontage"].fillna(testData["LotFrontage"].mean())
testData["MasVnrArea"] = testData["MasVnrArea"].fillna(testData["MasVnrArea"].mean())
testData["BsmtFinSF1"] = testData["BsmtFinSF1"].fillna(testData["BsmtFinSF1"].mean())
testData["BsmtFinSF2"] = testData["BsmtFinSF2"].fillna(testData["BsmtFinSF2"].mean())
testData["BsmtUnfSF"] = testData["BsmtUnfSF"].fillna(testData["BsmtUnfSF"].mean())
testData["TotalBsmtSF"] = testData["TotalBsmtSF"].fillna(testData["TotalBsmtSF"].mean())
testData["GarageArea"] = testData["GarageArea"].fillna(testData["GarageArea"].mean())

testData["BsmtQual"] = testData["BsmtQual"].fillna(testData["BsmtQual"].mode()[0])
testData["BsmtCond"] = testData["BsmtCond"].fillna(testData["BsmtCond"].mode()[0])
testData["BsmtExposure"] = testData["BsmtExposure"].fillna(testData["BsmtExposure"].mode()[0])
testData["BsmtFinType1"] = testData["BsmtFinType1"].fillna(testData["BsmtFinType1"].mode()[0])
testData["BsmtFinType2"] = testData["BsmtFinType2"].fillna(testData["BsmtFinType2"].mode()[0])
testData["GarageType"] = testData["GarageType"].fillna(testData["GarageType"].mode()[0])
testData["GarageYrBlt"] = testData["GarageYrBlt"].fillna(testData["GarageYrBlt"].mode()[0])
testData["GarageFinish"] = testData["GarageFinish"].fillna(testData["GarageFinish"].mode()[0])
testData["GarageQual"] = testData["GarageQual"].fillna(testData["GarageQual"].mode()[0])
testData["GarageCond"] = testData["GarageCond"].fillna(testData["GarageCond"].mode()[0])
testData["MasVnrType"] = testData["MasVnrType"].fillna(testData["MasVnrType"].mode()[0])
testData["Electrical"] = testData["Electrical"].fillna(testData["Electrical"].mode()[0])
testData["MSZoning"] = testData["MSZoning"].fillna(testData["MSZoning"].mode()[0])
testData["Utilities"] = testData["Utilities"].fillna(testData["Utilities"].mode()[0])
testData["Exterior1st"] = testData["Exterior1st"].fillna(testData["Exterior1st"].mode()[0])
testData["Exterior2nd"] = testData["Exterior2nd"].fillna(testData["Exterior2nd"].mode()[0])
testData["BsmtFullBath"] = testData["BsmtFullBath"].fillna(testData["BsmtFullBath"].mode()[0])
testData["BsmtHalfBath"] = testData["BsmtHalfBath"].fillna(testData["BsmtHalfBath"].mode()[0])
testData["KitchenQual"] = testData["KitchenQual"].fillna(testData["KitchenQual"].mode()[0])
testData["Functional"] = testData["Functional"].fillna(testData["Functional"].mode()[0])
testData["GarageCars"] = testData["GarageCars"].fillna(testData["GarageCars"].mode()[0])
testData["SaleType"] = testData["SaleType"].fillna(testData["SaleType"].mode()[0])


def encode_data(data):
    columns = list(data.select_dtypes(['object']).columns)
    x = data.iloc[:, :].values
    label_enc = LabelEncoder()
    rows = data.shape[0]

    for field in columns:
        # print(field)
        index_column = data.columns.get_loc(field)
        tmp = np.array(['shad' for _ in range(rows)])
        for i in range(rows):
            tmp[i] = x[:, index_column:index_column + 1][i][0]
        tmp = label_enc.fit_transform(tmp)
        tmp = np.reshape(tmp, (-1, 1))
        x[:, index_column:index_column + 1] = tmp

    return x


#print(type(trainData))
trainData = encode_data(trainData)
testData = encode_data(testData)
#np.savetxt("shad.csv", trainData, delimiter=",")

#print(trainLabel)

#model = DecisionTreeRegressor()
model = RandomForestRegressor(n_estimators=100, n_jobs=-1, oob_score=True)
model.fit(trainData, trainLabel)
sale_prices_predicted = model.predict(testData)

out = pd.DataFrame(sale_prices_predicted)
ds=pd.concat([Ids, out], axis=1)
ds.columns=['Id', 'SalePrice']
#ds.to_csv('1. DT_sample_submission.csv', index=False)
ds.to_csv('2. RF_sample_submission.csv', index=False)


