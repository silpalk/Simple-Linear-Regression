# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 10:32:10 2021

@author: Amarnadh Tadi
"""

import pandas as pd
import numpy as np
delivery=pd.read_csv(r"C:\Users\Amarnadh Tadi\Desktop\datascience\assign10\delivery_time.csv")
delivery.rename(columns={'Delivery Time':'time','Sorting Time' :'sort_time'}, inplace=True)

delivery.isnull().sum()

##Exploratory data analysis
delivery.describe()
#Graphical representation
import matplotlib.pyplot as plt
%matplotlib inline
plt.bar(height=delivery.time,x=np.arange(0,21,1))
plt.hist(delivery.time)
plt.boxplot(delivery.time)#no outliers
plt.bar(height=delivery.sort_time,x=np.arange(0,14,1))
plt.hist(delivery.sort_time)
plt.boxplot(delivery.sort_time)##no outliers

##Scatter plot
plt.scatter(delivery.time,delivery.sort_time)
#Correlation
np.corrcoef(delivery.time,delivery.sort_time)
##model building
import statsmodels.formula.api as smf
#simple linear regression
model1=smf.ols('time ~ sort_time',data=delivery).fit()
model1.summary()
pred1=model1.predict(pd.DataFrame(delivery['sort_time']))

#Regression line
plt.scatter(delivery.time,delivery.sort_time)
plt.plot(delivery.sort_time,pred1,"r")
plt.legend(['Predicted line','Observed data'])

##Error caluculation
error1=delivery.time-pred1
error_sq1=error1*error1
##mean error square(mes)
mesq1=np.mean(error_sq1)
##root mean squared error
rmse1=np.sqrt(mesq1)
rmse1

###Model building on Transformed data
##Applying  log transformation
#x=log(sort_time),y=time
plt.scatter(x=np.log(delivery.sort_time),y=delivery.time,color='brown')
np.corrcoef(np.log(delivery.sort_time),delivery.time)
model2=smf.ols('time ~ np.log(sort_time)' ,data=delivery).fit()
model2.summary
pred2=model2.predict(pd.DataFrame(delivery['sort_time']))
#regression line after log transformation

plt.scatter(np.log(delivery.sort_time),delivery.time)
plt.plot(np.log(delivery.sort_time),pred2,'r')
plt.legend(['Predicted line','Observed Data'])

##Error calucualtion
error2=delivery.time-pred2
error_sq2=error2*error2
mesq2=np.mean(error_sq2)
rmse2=np.sqrt(mesq2)
rmse2

##Applying Exponential transformation
#x=weight,y=log(calories)
plt.scatter(x=delivery.sort_time,y=np.log(delivery.time),color='brown')
np.corrcoef(delivery.sort_time,np.log(delivery.time))

model3=smf.ols('np.log(time) ~ sort_time',data=delivery).fit()
pred3=model3.predict(pd.DataFrame(delivery['sort_time']))
pred3_time=np.exp(pred3)
##regression line
plt.scatter(delivery.sort_time,np.log(delivery.time))
plt.plot(delivery.sort_time,pred3,"r")
plt.legend(['Predicted line','Observed Data'])

##Error calculation

error3=delivery.time-pred3_time
error_sq3=error3*error3
mesq3=np.mean(error_sq3)
rmse3=np.sqrt(mesq3)
rmse3

##Applying polynomial transformation
#x=weight,x^2=weight*weight,y=log(calories)

model4=smf.ols('np.log(time) ~ sort_time+I(sort_time*sort_time)',data=delivery).fit()
model4.summary()
pred4=model4.predict(pd.DataFrame(delivery['sort_time']))
pred4_time=np.mean(pred4)

#Regression line

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
X=delivery.iloc[:,[1]]
X
X_poly=poly_reg.fit_transform(X)
X_poly
plt.scatter(delivery.sort_time,np.log(delivery.time))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()

#Error caluculation


error4=delivery.time-pred4_time
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

train, test = train_test_split(delivery, test_size = 0.2)

finalmodel = smf.ols('time ~ np.log(sort_time)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
test_pred

# Model Evaluation on Test data
error=test.time-test_pred
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

