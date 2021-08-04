# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 13:07:45 2021

@author: Amarnadh Tadi
"""

import pandas as pd
import numpy as np
emp=pd.read_csv(r"C:\Users\Amarnadh Tadi\Desktop\datascience\assign10\emp_data.csv")
emp.columns
emp.rename(columns={'Salary_hike':'s_hike','Churn_out_rate' :'churn'}, inplace=True)

emp.isnull().sum()

##Exploratory data analysis
emp.describe()
#Graphical representation
import matplotlib.pyplot as plt
%matplotlib inline
plt.bar(height=emp.s_hike,x=np.arange(0,10,1))
plt.hist(emp.s_hike)
plt.boxplot(emp.s_hike)#no outliers
plt.bar(height=emp.churn,x=np.arange(0,10,1))
plt.hist(emp.churn)
plt.boxplot(emp.churn)##no outliers

##Scatter plot
plt.scatter(emp.s_hike,emp.churn)
#Correlation
np.corrcoef(emp.s_hike,emp.churn)
##model building
import statsmodels.formula.api as smf
#simple linear regression
model1=smf.ols('churn ~ s_hike',data=emp).fit()
model1.summary()
pred1=model1.predict(pd.DataFrame(emp['s_hike']))

#Regression line
plt.scatter(emp.s_hike,emp.churn)
plt.plot(emp.s_hike,pred1,"r")
plt.legend(['Predicted line','Observed data'])

##Error caluculation
error1=emp.churn-pred1
error_sq1=error1*error1
##mean error square(mes)
mesq1=np.mean(error_sq1)
##root mean squared error
rmse1=np.sqrt(mesq1)
rmse1

###Model building on Transformed data
##Applying  log transformation
#x=log(s_hike),y=churn
plt.scatter(x=np.log(emp.s_hike),y=emp.churn,color='brown')
np.corrcoef(np.log(emp.s_hike),emp.churn)
model2=smf.ols('churn ~ np.log(s_hike)' ,data=emp).fit()
model2.summary
pred2=model2.predict(pd.DataFrame(emp['s_hike']))
#regression line after log transformation

plt.scatter(np.log(emp.s_hike),emp.churn)
plt.plot(np.log(emp.s_hike),pred2,'r')
plt.legend(['Predicted line','Observed Data'])

##Error calucualtion
error2=emp.churn-pred2
error_sq2=error2*error2
mesq2=np.mean(error_sq2)
rmse2=np.sqrt(mesq2)
rmse2

##Applying Exponential transformation
#x=weight,y=log(calories)
plt.scatter(x=emp.s_hike,y=np.log(emp.churn),color='brown')
np.corrcoef(emp.s_hike,np.log(emp.churn))

model3=smf.ols('np.log(churn) ~ s_hike',data=emp).fit()
pred3=model3.predict(pd.DataFrame(emp['s_hike']))
pred3_time=np.exp(pred3)
##regression line
plt.scatter(emp.s_hike,np.log(emp.churn))
plt.plot(emp.s_hike,pred3,"r")
plt.legend(['Predicted line','Observed Data'])

##Error calculation

error3=emp.churn-pred3_time
error_sq3=error3*error3
mesq3=np.mean(error_sq3)
rmse3=np.sqrt(mesq3)
rmse3

##Applying polynomial transformation
#x=s_hike,x^2=s_hike*s_hike,y=log(churn)

model4=smf.ols('np.log(churn) ~ s_hike+I(s_hike*s_hike)',data=emp).fit()
model4.summary()
pred4=model4.predict(pd.DataFrame(emp['s_hike']))
pred4_churn=np.mean(pred4)

#Regression line

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
X=emp.iloc[:,0:1]
X
X_poly=poly_reg.fit_transform(X)
X_poly
plt.scatter(emp.s_hike,np.log(emp.churn))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()

#Error caluculation


error4=emp.churn-pred4_churn
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

train, test = train_test_split(emp, test_size = 0.2)

finalmodel = smf.ols('np.log(churn) ~ s_hike', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
test_pred

# Model Evaluation on Test data
error=test.churn-test_pred
error_sq=error*error
mesq=np.mean(error_sq)
rmse_final=np.sqrt(mesq)
rmse_final


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_churn = np.exp(train_pred)
pred_train_churn

# Model Evaluation on train data
train_res = train.churn - pred_train_churn
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

`

