# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 15:03:04 2021

@author: Amarnadh Tadi
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 13:07:45 2021

@author: Amarnadh Tadi
"""

import pandas as pd
import numpy as np
emp = pd.read_csv(r"C:\Users\Amarnadh Tadi\Desktop\datascience\linear_regression\Salary_data.csv")
emp.columns
emp.rename(columns={'YearsExperience':'exp','Salary' :'salary'}, inplace=True)

emp.isnull().sum()

##Exploratory data analysis
emp.describe()
#Graphical representation
import matplotlib.pyplot as plt
%matplotlib inline
plt.bar(height=emp.exp,x=np.arange(0,30,1))
plt.hist(emp.exp)
plt.boxplot(emp.exp)#no outliers
plt.bar(height=emp.salary,x=np.arange(0,30,1))
plt.hist(emp.salary)
plt.boxplot(emp.salary)##no outliers

##Scatter plot
plt.scatter(emp.exp,emp.salary)
#Correlation
np.corrcoef(emp.exp,emp.salary)
##model building
import statsmodels.formula.api as smf
#simple linear regression
model1=smf.ols('salary ~ exp',data=emp).fit()
model1.summary()
pred1=model1.predict(pd.DataFrame(emp['exp']))

#Regression line
plt.scatter(emp.exp,emp.salary)
plt.plot(emp.exp,pred1,"r")
plt.legend(['Predicted line','Observed data'])

##Error caluculation
error1=emp.salary-pred1
error_sq1=error1*error1
##mean error square(mes)
mesq1=np.mean(error_sq1)
##root mean squared error
rmse1=np.sqrt(mesq1)
rmse1

###Model building on Transformed data
##Applying  log transformation
#x=log(exp),y=salary
plt.scatter(x=np.log(emp.exp),y=emp.salary,color='brown')
np.corrcoef(np.log(emp.exp),emp.salary)
model2=smf.ols('salary ~ np.log(exp)' ,data=emp).fit()
model2.summary
pred2=model2.predict(pd.DataFrame(emp['exp']))
#regression line after log transformation

plt.scatter(np.log(emp.exp),emp.salary)
plt.plot(np.log(emp.exp),pred2,'r')
plt.legend(['Predicted line','Observed Data'])

##Error calucualtion
error2=emp.salary-pred2
error_sq2=error2*error2
mesq2=np.mean(error_sq2)
rmse2=np.sqrt(mesq2)
rmse2

##Applying Exponential transformation
#x=exp,y=log(salary)
plt.scatter(x=emp.exp,y=np.log(emp.salary),color='brown')
np.corrcoef(emp.exp,np.log(emp.salary))

model3=smf.ols('np.log(salary) ~ exp',data=emp).fit()
pred3=model3.predict(pd.DataFrame(emp['exp']))
pred3_time=np.exp(pred3)
##regression line
plt.scatter(emp.exp,np.log(emp.salary))
plt.plot(emp.exp,pred3,"r")
plt.legend(['Predicted line','Observed Data'])

##Error calculation

error3=emp.salary-pred3_time
error_sq3=error3*error3
mesq3=np.mean(error_sq3)
rmse3=np.sqrt(mesq3)
rmse3

##Applying polynomial transformation
#x=s_hike,x^2=s_hike*s_hike,y=log(churn)

model4=smf.ols('np.log(salary) ~ exp+I(exp*exp)',data=emp).fit()
model4.summary()
pred4=model4.predict(pd.DataFrame(emp['exp']))
pred4_sal=np.mean(pred4)

#Regression line

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
X=emp.iloc[:,0:1]
X
X_poly=poly_reg.fit_transform(X)
X_poly
plt.scatter(emp.exp,np.log(emp.salary))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()

#Error caluculation


error4=emp.salary-pred4_sal
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

finalmodel = smf.ols('np.log(salary) ~ exp', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
test_pred

# Model Evaluation on Test data
error=test.salary-test_pred
error_sq=error*error
mesq=np.mean(error_sq)
rmse_final=np.sqrt(mesq)
rmse_final


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_sal = np.exp(train_pred)
pred_train_sal

# Model Evaluation on train data
train_res = train.salary - pred_train_sal
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

`

