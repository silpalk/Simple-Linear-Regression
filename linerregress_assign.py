# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 00:28:29 2021

@author: Amarnadh Tadi
"""

import pandas as pd
import numpy as np
cal_data=pd.read_csv(r"C:\Users\Amarnadh Tadi\Desktop\datascience\assign10\calories_consumed.csv")
cal_data.columns
cal_data.rename(columns={'Weight gained (grams)': 'weight','Calories Consumed': 'calories'}, inplace=True)
cal_data.columns
cal_data.isnull().sum()##no null values

##Exploratory data analysis
cal_data.describe()
#Graphical representation
import matplotlib.pyplot as plt
%matplotlib inline
plt.bar(height=cal_data.weight,x=np.arange(0,14,1))
plt.hist(cal_data.weight)
plt.boxplot(cal_data.weight)#no outliers
plt.bar(height=cal_data.calories,x=np.arange(0,14,1))
plt.hist(cal_data.calories)
plt.boxplot(cal_data.calories)##no outliers

##Scatter plot
plt.scatter(cal_data.weight,cal_data.calories)
#Correlation
np.corrcoef(cal_data.weight,cal_data.calories)
##model building
import statsmodels.formula.api as smf
#simple linear regression
model1=smf.ols('calories ~ weight',data=cal_data).fit()
model1.summary()
pred1=model1.predict(pd.DataFrame(cal_data['weight']))

#Regression line
plt.scatter(cal_data.weight,cal_data.calories)
plt.plot(cal_data.weight,pred1,"r")
plt.legend(['Predicted line','Observed data'])

##Error caluculation
error1=cal_data.calories-pred1
error_sq1=error1*error1
##mean error square(mes)
mesq1=np.mean(error_sq1)
##root mean squared error
rmse1=np.sqrt(mesq1)
rmse1

###Model building on Transformed data
##Applying  log transformation
#x=log(weight),y=calories
plt.scatter(x=np.log(cal_data.weight),y=cal_data.calories,color='brown')
np.corrcoef(np.log(cal_data.weight),cal_data.calories)
model2=smf.ols('cal_data.calories ~ np.log(cal_data.weight)' ,data=cal_data).fit()
model2.summary
pred2=model2.predict(pd.DataFrame(cal_data['weight']))
#regression line after log transformation

plt.scatter(np.log(cal_data.weight),cal_data.calories)
plt.plot(np.log(cal_data.weight),pred2,'r')
plt.legend(['Predicted line','Observed Data'])

##Error calucualtion
error2=cal_data.calories-pred2
error_sq2=error2*error2
mesq2=np.mean(error_sq2)
rmse2=np.sqrt(mesq2)
rmse2

##Applying Exponential transformation
#x=weight,y=log(calories)
plt.scatter(x=cal_data.weight,y=np.log(cal_data.calories),color='brown')
np.corrcoef(cal_data.weight,np.log(cal_data.calories))

model3=smf.ols('np.log(calories) ~weight',data=cal_data).fit()
pred3=model3.predict(pd.DataFrame(cal_data['weight']))
pred3_cal=np.exp(pred3)
##regression line
plt.scatter(cal_data.weight,np.log(cal_data.calories))
plt.plot(cal_data.weight,pred3,"r")
plt.legend(['Predicted line','Observed Data'])

##Error calculation

error3=cal_data.calories-pred3_cal
error_sq3=error3*error3
mesq3=np.mean(error_sq3)
rmse3=np.sqrt(mesq3)
rmse3

##Applying polynomial transformation
#x=weight,x^2=weight*weight,y=log(calories)

model4=smf.ols('np.log(calories) ~ weight+I(weight*weight)',data=cal_data).fit()
model4.summary()
pred4=model4.predict(pd.DataFrame(cal_data['weight']))
pred4_cal=np.mean(pred4)

#Regression line

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
X=cal_data.iloc[:,0:1]
X_poly=poly_reg.fit_transform(X)
X_poly
plt.scatter(cal_data.weight,np.log(cal_data.calories))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()

#Error caluculation


error4=cal_data.calories-pred4_cal
error_sq4=error4*error4
mesq4=np.mean(error_sq4)
rmse4=np.sqrt(mesq4)
rmse4

#choose the best model using RMSE
data={"Model":pd.Series(['SLR','log model','exp model','poly model']),"RMSE":pd.Series([rmse1,rmse2,rmse3,rmse4])}
data
table_rmse=pd.DataFrame(data)
table_rmse

##the best model
from sklearn.model_selection import train_test_split

train, test = train_test_split(cal_data, test_size = 0.2)

finalmodel = smf.ols('calories ~ weight', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
test_pred

# Model Evaluation on Test data
error=test.calories-test_pred
error_sq=error*error
mesq=np.mean(error_sq)
rmse_final=np.sqrt(mesq)
rmse_final


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_cal = np.exp(train_pred)
pred_train_cal

# Model Evaluation on train data
train_res = train.calories - pred_train_cal
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

`

