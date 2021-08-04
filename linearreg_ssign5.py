# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 16:11:58 2021

@author: Amarnadh Tadi
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 15:03:04 2021

@author: Amarnadh Tadi
"""




import pandas as pd
import numpy as np
sat = pd.read_csv(r"C:\Users\Amarnadh Tadi\Desktop\datascience\assign10\SAT_GPA.csv")
sat.columns
sat.rename(columns={'SAT_Scores':'score','GPA' :'gpa'}, inplace=True)

sat.isnull().sum()

##Exploratory data analysis
sat.describe()
#Graphical representation
import matplotlib.pyplot as plt
%matplotlib inline
plt.bar(height=sat.score,x=np.arange(0,200,1))
plt.hist(sat.score)
plt.boxplot(sat.score)#no outliers
plt.bar(height=sat.gpa,x=np.arange(0,200,1))
plt.hist(sat.gpa)
plt.boxplot(sat.gpa)##no outliers

##Scatter plot
plt.scatter(sat.gpa,sat.score)
#Correlation
np.corrcoef(sat.gpa,sat.score)
##model building
import statsmodels.formula.api as smf
#simple linear regression
model1=smf.ols('score ~ gpa',data=sat).fit()
model1.summary()
pred1=model1.predict(pd.DataFrame(sat['gpa']))

#Regression line
plt.scatter(sat.gpa,sat.score)
plt.plot(sat.gpa,pred1,"r")
plt.legend(['Predicted line','Observed data'])

##Error caluculation
error1=sat.score-pred1
error_sq1=error1*error1
##mean error square(mes)
mesq1=np.mean(error_sq1)
##root mean squared error
rmse1=np.sqrt(mesq1)
rmse1

###Model building on Transformed data
##Applying  log transformation
#x=log(gpa),y=score
plt.scatter(x=np.log(sat.gpa),y=sat.score,color='brown')
np.corrcoef(np.log(sat.gpa),sat.score)
model2=smf.ols('score ~ np.log(gpa)' ,data=sat).fit()
model2.summary
pred2=model2.predict(pd.DataFrame(sat['gpa']))
#regression line after log transformation

plt.scatter(np.log(sat.gpa),sat.score)
plt.plot(np.log(sat.gpa),pred2,'r')
plt.legend(['Predicted line','Observed Data'])

##Error calucualtion
error2=sat.score-pred2
error_sq2=error2*error2
mesq2=np.mean(error_sq2)
rmse2=np.sqrt(mesq2)
rmse2

##Applying Exponential transformation
#x=gpa,y=log(salary)
plt.scatter(x=sat.gpa,y=np.log(sat.score),color='brown')
np.corrcoef(sat.gpa,np.log(sat.score))

model3=smf.ols('np.log(score) ~ gpa',data=sat).fit()
pred3=model3.predict(pd.DataFrame(sat['gpa']))
pred3_score=np.exp(pred3)
##regression line
plt.scatter(sat.gpa,np.log(sat.score))
plt.plot(sat.gpa,pred3,"r")
plt.legend(['Predicted line','Observed Data'])

##Error calculation

error3=sat.score-pred3_score
error_sq3=error3*error3
mesq3=np.mean(error_sq3)
rmse3=np.sqrt(mesq3)
rmse3

##Applying polynomial transformation
#x=gpa,x^2=gpa*gpa,y=log(score)

model4=smf.ols('np.log(score) ~ gpa+I(gpa*gpa)',data=sat).fit()
model4.summary()
pred4=model4.predict(pd.DataFrame(sat['gpa']))
pred4_score=np.mean(pred4)

#Regression line

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
X=sat.iloc[:,[1]]
X
X_poly=poly_reg.fit_transform(X)
X_poly
plt.scatter(sat.gpa,np.log(sat.score))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()

#Error caluculation


error4=sat.score-pred4_score
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

train, test = train_test_split(sat, test_size = 0.2)

finalmodel = smf.ols('np.log(score) ~ gpa', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
test_pred

# Model Evaluation on Test data
error=test.score-test_pred
error_sq=error*error
mesq=np.mean(error_sq)
rmse_final=np.sqrt(mesq)
rmse_final


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_score = np.exp(train_pred)
pred_train_score

# Model Evaluation on train data
train_res = train.score - pred_train_score
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

`

