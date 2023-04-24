#1.Βαζω απαραίτητες βιβλιοθηκες
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

from scipy.stats import norm # for some statistics
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#2. Κατανοηση των Datasets
hous_pred_df = pd.read_csv(r'../input/house-prices-advanced-regression-techniques/train.csv')
print(hous_pred_df.info())
print(hous_pred_df.head(10))
print(hous_pred_df.describe())
print(hous_pred_df.columns)

#3.Κάνω Correlation and HeatMap
#Calculating the correlation
hous_pred_corr = hous_pred_df.corr()
hous_pred_corr

#Plotting heatmap
plt.figure(figsize = (30,20))
sns.heatmap(hous_pred_corr,data=hous_pred_df,cmap='Blues',annot=True)

#4.Διαχειριζομαι τις nulls values

hous_pred_df.loc[:, hous_pred_df.isnull().any()].isna().sum()

#Dropping the columns/attributes which have More Nulls(Nulls > 1100) και το ID δεν χρειαζεται
hous_pred_df_copy = hous_pred_df
colswithhnull = ['Id','Alley','PoolQC','Fence','MiscFeature']
for col in colswithhnull:
    hous_pred_df = hous_pred_df.drop(col,axis=1)

print(hous_pred_df.shape)
print(hous_pred_df_copy.shape)

#5.Διαχωριζω τα columns σε categorical και non-categorical

hous_cat_cols = ['MSZoning', 'Street',
                 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
                 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
                 'HouseStyle','RoofStyle', 'RoofMatl',
                 'Exterior1st', 'Exterior2nd', 'MasVnrType','ExterQual', 'ExterCond',
                 'Foundation', 'BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                 'Heating','HeatingQC', 'CentralAir', 'Electrical',
                 'KitchenQual','Functional', 'FireplaceQu',
                 'GarageType','GarageFinish', 'GarageQual','GarageCond', 'PavedDrive',
                 'SaleType','SaleCondition']

#Group categorical columns with String Values ΣΕ ΕΝΑ group
hous_pred_df_cat=hous_pred_df.loc[:,hous_cat_cols]
hous_pred_df_cat.head()

#Group rest of the columns
hous_pred_df_noncat = hous_pred_df.drop(columns=hous_cat_cols)
hous_pred_df_noncat.head()


#ΧΡΗΣΗ ΤΟΥ SimpleImputer για να γεμισω τα κενα
#non-categorical values use mean
from sklearn.impute import SimpleImputer
SI = SimpleImputer(missing_values = np.nan,strategy='mean')
hous_pred_df_noncat_new = SI.fit_transform(hous_pred_df_noncat)
hous_pred_df_noncat_new=pd.DataFrame(hous_pred_df_noncat_new,columns=hous_pred_df_noncat.columns,index=hous_pred_df_noncat.index)

hous_pred_df_noncat_new.head()

#categorical values use most_frequent(mode)
hous_pred_df_new_cat = SimpleImputer(strategy='most_frequent').fit_transform(hous_pred_df_cat)
hous_pred_df_new_cat = pd.DataFrame(hous_pred_df_new_cat, columns=hous_pred_df_cat.columns,
                                    index=hous_pred_df_cat.index)
for col in hous_pred_df_new_cat.columns:
    hous_pred_df_new_cat[col] = hous_pred_df_new_cat[col].astype('category')

for col in hous_pred_df_new_cat.columns:
    hous_pred_df_new_cat[col] = hous_pred_df_new_cat[col].cat.codes

hous_pred_df_new_cat.head()

#Ενωνω πίσω τα νεα δεδομενα σε ενα dataset
hous_pred_df_new=pd.concat([hous_pred_df_new_cat,hous_pred_df_noncat_new],axis=1)

#Ξανα κανω heatmap να δω την σχεση τωρα με τα νεα δεδομενα
hous_pred_corr = hous_pred_df_new.corr()
hous_pred_corr
plt.figure(figsize = (60,30))
sns.heatmap(hous_pred_corr,cbar=True,
                 fmt='.2f',annot=True,
                 cmap='Blues')

#Φευγω αυτες που ειναι λιγοτερο σχετικες με το saleprice
hous_no_info_cols = ['Street','Utilities','Condition2','RoofMatl','Exterior2nd','Heating','CentralAir','LowQualFinSF',
                     'BsmtHalfBath','KitchenAbvGr','Functional','GarageCond','PoolArea','MiscVal']
hous_pred_df_new.drop(columns=hous_no_info_cols)

hous_pred_corr = hous_pred_df_new.corr()
hous_pred_corr
plt.figure(figsize = (60,30))
sns.heatmap(hous_pred_corr,cbar=True,
                 fmt='.2f',annot=True,
                 cmap='Blues')


# BAR PLOT ΓΙΑ ΟΛΑ τα feautures με το saleprice
houscol = hous_pred_df_new.columns
houscol = houscol.tolist()
houscol.remove('SalePrice')
y=hous_pred_df_new['SalePrice']
plt.ylabel('SalePrice')
for col in houscol:
    x = hous_pred_df_new.loc[:,col]
    plt.xlabel(col)
    title = col+' Graph'
    plt.title(title)
    plt.bar(x,y)
    plt.show()

#6. διαχωριζω τα dataset μου
hous_pred_x = hous_pred_df_new.drop('SalePrice',axis=1)
hous_pred_y = hous_pred_df_new['SalePrice']

from sklearn.model_selection import train_test_split
hous_pred_x_train, hous_pred_x_test, hous_pred_y_train, hous_pred_y_test = train_test_split(
    hous_pred_x, hous_pred_y, test_size=0.20, random_state=0)

#7. εφαρμοζω 5 διαφορετικες τεχνικες regression και βρισκω το MEAN

#1.LINEAR
from sklearn.linear_model import LinearRegression
hous_pred_lr = LinearRegression()
hous_pred_lr_fit =hous_pred_lr.fit(hous_pred_x_train,hous_pred_y_train)

#Prediction
hous_pred_lr_predict = hous_pred_lr.predict(hous_pred_x_test)

#Comparing the predicted value with the Actual values
hous_pred_lr_compare = pd.DataFrame({'Actual': hous_pred_y_test, 'Predicted': hous_pred_lr_predict})
hous_pred_lr_compare.head(25)

#ACCURACY
from sklearn import metrics
accuracylr = metrics.r2_score(hous_pred_y_test,hous_pred_lr_predict)
print( 'Predicted Accuracy Linear Regression:', accuracylr)

#Mean Square error, root mean square error, R2-Score
from sklearn import metrics
print('MSE:', metrics.mean_squared_error(hous_pred_y_test, hous_pred_lr_predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(hous_pred_y_test, hous_pred_lr_predict)))
print("R2 score =", round(metrics.r2_score(hous_pred_y_test, hous_pred_lr_predict), 2))

#2.Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
hous_pred_gbr = GradientBoostingRegressor()
hous_pred_gbr_fit =hous_pred_gbr.fit(hous_pred_x_train,hous_pred_y_train)

#Prediction
hous_pred_gbr_predict = hous_pred_gbr.predict(hous_pred_x_test)

#Compare
hous_pred_gbr_compare = pd.DataFrame({'Actual': hous_pred_y_test, 'Predicted': hous_pred_gbr_predict})
hous_pred_gbr_compare.head(25)

#Mean Square error, root mean square error, R2-Score
print('MSE:', metrics.mean_squared_error(hous_pred_y_test, hous_pred_gbr_predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(hous_pred_y_test, hous_pred_gbr_predict)))
print("R2 score =", round(metrics.r2_score(hous_pred_y_test, hous_pred_gbr_predict), 2))

#Accuracy
from sklearn import metrics
accuracygbr = metrics.r2_score(hous_pred_y_test,hous_pred_gbr_predict)
print( 'Predicted Accuracy Gradient Boosting Regressor:', accuracygbr)

#3.Random Forest
from sklearn.ensemble import RandomForestRegressor
hous_pred_rfr = RandomForestRegressor()
hous_pred_rfr_fit =hous_pred_rfr.fit(hous_pred_x_train,hous_pred_y_train)

#Prediction
hous_pred_rfr_predict = hous_pred_rfr.predict(hous_pred_x_test)

#compare
hous_pred_rfr_compare = pd.DataFrame({'Actual': hous_pred_y_test, 'Predicted': hous_pred_rfr_predict})
hous_pred_rfr_compare.head(25)

#Accuracy
from sklearn import metrics
accuracyrfr = metrics.r2_score(hous_pred_y_test,hous_pred_rfr_predict)
print( 'Predicted Accuracy random forest:', accuracyrfr)

#Mean Square error
print('MSE:', metrics.mean_squared_error(hous_pred_y_test, hous_pred_rfr_predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(hous_pred_y_test, hous_pred_rfr_predict)))
print("R2 score =", round(metrics.r2_score(hous_pred_y_test, hous_pred_rfr_predict), 2))

#4.Extreme Gradient Boost
from xgboost.sklearn import XGBRegressor
hous_pred_xgbr = XGBRegressor()
hous_pred_xgbr_fit =hous_pred_xgbr.fit(hous_pred_x_train,hous_pred_y_train)

#prediction
hous_pred_xgbr_predict = hous_pred_xgbr.predict(hous_pred_x_test)

#Compare
hous_pred_xgbr_compare = pd.DataFrame({'Actual': hous_pred_y_test, 'Predicted': hous_pred_xgbr_predict})
hous_pred_xgbr_compare.head(25)

#Accuracy
from sklearn import metrics
accuracyxgbr = metrics.r2_score(hous_pred_y_test,hous_pred_xgbr_predict)
print( 'Predicted Accuracy XGBRegressor:', accuracyxgbr)

#Calculate
print('MSE:', metrics.mean_squared_error(hous_pred_y_test, hous_pred_xgbr_predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(hous_pred_y_test, hous_pred_xgbr_predict)))
print("R2 score =", round(metrics.r2_score(hous_pred_y_test, hous_pred_xgbr_predict), 2))

#5.SVR
from sklearn.svm import SVR
hous_pred_svr = SVR()
hous_pred_svr_fit =hous_pred_svr.fit(hous_pred_x_train,hous_pred_y_train)

#Prediction
hous_pred_svr_predict = hous_pred_svr.predict(hous_pred_x_test)

#Compare
hous_pred_svr_compare = pd.DataFrame({'Actual': hous_pred_y_test, 'Predicted': hous_pred_svr_predict})
hous_pred_svr_compare.head(25)

#Accuracy
from sklearn import metrics
accuracysvr = metrics.r2_score(hous_pred_y_test,hous_pred_svr_predict)
print( 'Predicted Accuracy SVR:', accuracysvr)

#calculat mean
print('MSE:', metrics.mean_squared_error(hous_pred_y_test, hous_pred_svr_predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(hous_pred_y_test, hous_pred_svr_predict)))
print("R2 score =", round(metrics.r2_score(hous_pred_y_test, hous_pred_svr_predict), 2))

from tabulate import tabulate


#8. ΟΛΑ ΤΑ ΑΠΟΤΕΛΕΣΜΑΤΑ ΜΑΖΙ\
from tabulate import tabulate

#MEAN TABLE
hous_score_comp = [
    ["Linear Regression", metrics.mean_squared_error(hous_pred_y_test, hous_pred_lr_predict),
     np.sqrt(metrics.mean_squared_error(hous_pred_y_test, hous_pred_lr_predict)),
     round(metrics.r2_score(hous_pred_y_test, hous_pred_lr_predict), 2)],
    ["Gradient Boosting Regression", metrics.mean_squared_error(hous_pred_y_test, hous_pred_gbr_predict),
     np.sqrt(metrics.mean_squared_error(hous_pred_y_test, hous_pred_gbr_predict)),
     round(metrics.r2_score(hous_pred_y_test, hous_pred_gbr_predict), 2)],
    ["Random Forest", metrics.mean_squared_error(hous_pred_y_test, hous_pred_rfr_predict),
     np.sqrt(metrics.mean_squared_error(hous_pred_y_test, hous_pred_rfr_predict)),
     round(metrics.r2_score(hous_pred_y_test, hous_pred_rfr_predict), 2)],
    ["Extreme Gradient Boost", metrics.mean_squared_error(hous_pred_y_test, hous_pred_xgbr_predict),
     np.sqrt(metrics.mean_squared_error(hous_pred_y_test, hous_pred_xgbr_predict)),
     round(metrics.r2_score(hous_pred_y_test, hous_pred_xgbr_predict), 2)],
    ["SVM", metrics.mean_squared_error(hous_pred_y_test, hous_pred_svr_predict),
     np.sqrt(metrics.mean_squared_error(hous_pred_y_test, hous_pred_svr_predict)),
     round(metrics.r2_score(hous_pred_y_test, hous_pred_svr_predict), 2)],

]

#header
head = ["Type of Regression", "Mean Square Error", "Root Mean Square Error", "R2-Score"]

#table
print(tabulate(hous_score_comp, headers=head, tablefmt="grid"))

#ACCURACY TABLE

# data
hous_score_accuracy = [
    ["Linear Regression", accuracylr],
    ["Gradient Boosting Regression", accuracygbr],
    ["Random Forest", accuracyrfr],
    ["Extreme Gradient Boost", accuracyxgbr],
    ["SVM", accuracysvr],

]

# header
head = ["Type of Regression", "Accuracy"]

# table
print(tabulate(hous_score_accuracy, headers=head, tablefmt="grid"))

#9. Κανω vizulize το μοντελο XGRB εχει το λιγοτερο Mean square error

#Stem Plot
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.stem(hous_pred_y_test,hous_pred_xgbr_predict)

#Scatter Plot
plt.scatter(hous_pred_y_test,hous_pred_xgbr_predict)

#Regression Plot
x, y = pd.Series(hous_pred_y_test, name="Actual Values"), pd.Series(hous_pred_xgbr_predict, name="Predicted Values")
ax = sns.regplot(x=x, y=y, marker="+")

#Rel Plot
ax = sns.relplot(x=x, y=y, marker="+")
