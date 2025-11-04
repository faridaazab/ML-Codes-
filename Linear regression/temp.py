# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd 
import numpy 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
dataset = pd.read_csv("Experience-Salary-Dataset.csv") # csv file is formating of the dataset "Comma Separated Vector"
display(dataset.head(10)) #To display the data in the dataset 10 --> refers to the number of records you need to display
x = dataset.iloc[:,[0]].values #iloc --> to divided the data "integer location based index" --> [] --> means matrix
#print(x.shape)
y = dataset.iloc[:,1].values # y --> is vector
#print(y.shape)
xTrain  , xTest, yTrain , yTest = train_test_split(x,y,test_size=0.2) # Train and Test 80 : 20 %
linearRegressor = LinearRegression().fit(xTrain,yTrain) # call the model
# after making fitting to the model we must Test the model
yPredict = linearRegressor.predict(xTest) #predict is built in method tp predict the output then comapre it with yTest
# Evaluation to the model to know what is the accurecy 
MeanSqrError = mean_squared_error(yTest, yPredict)

# Visualization Part
yPredictOfTrain = linearRegressor.predict(xTrain)
plt.scatter(xTrain, yTrain,color="purple")
plt.plot(xTrain, yPredictOfTrain,color="pink")
plt.xlabel('xTrain', color="Black")
plt.ylabel("yTrain", color="green")

plt.scatter(xTest, yTest,color="purple")
plt.plot(xTest, yPredict,color="pink")
plt.xlabel('Experiance', color="Black")
plt.ylabel("Salary", color="green")








