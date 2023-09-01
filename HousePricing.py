# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

'''Import bunch of packages
'''
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn import linear_model
from scipy.stats import norm

'''Read the csv file
'''
df = pd.read_csv('/Users/yikangli/Desktop/python code/train.csv')

'''see what the variables are
'''
print(df.columns)
'''SalePrice is the dependent variable. Let's show its properties.
'''
print(df['SalePrice'].describe())

'''Plot a histgram for SalePrice and fit a normal distribution to it. We see it is not that normal
'''

sns.distplot(df['SalePrice'], fit = norm);
fig = plt.figure()

'''One may find by intuition that GrLivArea is highly related to the SalePrice. Lets to a linear regression.
'''

sale_price = df['SalePrice'].tolist()

living_area = df['GrLivArea'].tolist()

slope, intercept, R, p, stderr = stats.linregress(living_area, sale_price) 

#now plot the best-fitted line
def plotfunction(x):
    return slope*x + intercept

predicted_sale_price = list(map(plotfunction, living_area))

plt.scatter(living_area, sale_price, s = 10, c = 'r')
plt.plot(living_area, predicted_sale_price)
plt.show()
print(R**2)
'''The value R^2 is 0.5021, not too bad. It means that GrLivArea has some explanatory power.
'''

'''From the scatter plot, we see that the variance increases as x increases. It is a characteristic of heteroskedasticity. Let's show this
'''
#create a new dataframe with only GrLivArea and SalePrice
df_LS = df[['GrLivArea', 'SalePrice']]
#re-order the dataframe in an ascending order of GrLivArea
df_sort = df_LS.sort_values(by = ['GrLivArea'])

#divide the data into 73 groups (each with 20 elements). calculate the mean in GrLivArea and the standard deviation in SalePrice in each group.
listx = []
listy = []
i=0
while i < 73:
    df_sorti = df_sort.iloc[20*i : 20*(i+1)]
    meanx = df_sorti['GrLivArea'].mean()
    stdy = df_sorti['SalePrice'].std()
    listx.append(meanx)
    listy.append(stdy)
    i+=1
    
plt.scatter(listx, listy)
plt.show()

'''Find the coefficient matrix(how strong variables are related to each other)
'''
correlation_matrix = df.corr()
#print(correlation_matrix)
f, ax = plt.subplots(figsize = (12, 9))
sns.heatmap(correlation_matrix, square = True)
'''re-order the matrix so we see what are the strongest related variables too SalePrice
'''
sorted_corr = correlation_matrix.abs().sort_values(by = ['SalePrice'], ascending = False)
'''From the correlation diagram, we see TotalBsmtSF is highly related to 1stFlrSF
and GarageArea highly related to GarageCars. So there are the variables that cause multicollinearity.
We remove them from the top ten related variables.
'''
multico_vari = ['1stFlrSF', 'GarageCars', 'SalePrice']

related_index = list(set(sorted_corr.index[:10])-set(multico_vari))
print(related_index)


#create a dataframe with only these related variables
updated_df = df[related_index]

print(updated_df)
'''
null_list=[]
for i in related_index:
    null_num = df[i].isnull().sum()
    null_list.append(null_num)

print(null_list)
'''
'''we can do a multiple linear regression on SalePrice with these related variables.
'''
XX = updated_df

yy = df['SalePrice']

regression = linear_model.LinearRegression()
regression.fit(XX, yy)
print(regression.coef_)

'''The variables are not distributed normally. we may want to take log to the continuous variables
to eliminate the skewness. For variables that contains 0, we replace 0 by 1 to avoid having infinity after taking log
'''

df['TotalBsmtSF'] = df['TotalBsmtSF'].replace(0, 1)
df['GarageArea'] = df['GarageArea'].replace(0, 1)

#plot figures for all independent variables
for i in related_index:
    sns.distplot(df[i], fit = norm);
    fig = plt.figure()

#take log on all continuous variables
df['GrLivArea'] = np.log(df['GrLivArea'])
df['GarageArea'] = np.log(df['GarageArea'])
df['TotalBsmtSF'] = np.log(df['TotalBsmtSF'])
df['SalePrice'] = np.log(df['SalePrice'])

#draw a diagram of GrLivArea to illustrate this point

sns.distplot(df['GrLivArea'], fit = norm);
fig = plt.figure()

'''Now, we do a multiple regression on SalePrice
'''
logXX = df[related_index]

logyy = df['SalePrice']

regression = linear_model.LinearRegression()
regression.fit(logXX, logyy)
print(regression.coef_)

    
    
    
