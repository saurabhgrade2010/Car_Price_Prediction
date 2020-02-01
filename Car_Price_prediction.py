
#import libraries
import numpy as np  #numeric python
import pandas as pd #for making structures
import matplotlib.pyplot as plt  #visualization
import seaborn as sns #visualization
from sklearn.model_selection import train_test_split #split data into train and test
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
 

#import data from csv file
data = pd.read_csv('data.csv')

#check if any null exist
data.isnull().sum()

#for data information
data.info()
data.describe(include = 'all')

#visualizing all data features and response
sns.pairplot(data)

#now here is three hypothesis
#1. Less Driven car have hight selling price
#2. Latest car will have high sellng Price
#3. Automatic Transmission Cars have high selling price

#1st hypothesis
data.head(15)
X = data.iloc[:,4:5].values
Y = data.iloc[:,2:3].values

#using sklearn model visual data for 1st hypothesis
regressor = LinearRegression()
regressor.fit(X,Y)
plt.scatter(X,Y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('visualize data using sklearn')
plt.xlabel('Km_driven')
plt.ylabel('Selling Price')
plt.show()
#using seaborn visual data for 1st hypothesis
fig, ax1 = plt.subplots(figsize = (15,10))
sns.scatterplot(x='Kms_Driven',y='Selling_Price',data = data,ax=ax1)


#2nd Hypothesis
data['Y_S_L'] = 2019 - data.Year
data.head()

#using sklearn 
P = data.iloc[:,9:10].values
Q = data.iloc[:,2:3].values

regressor = LinearRegression()
regressor.fit(P,Q)
plt.scatter(P,Q,color='green')
plt.plot(P,regressor.predict(P),color= 'blue')
plt.title('visualize data using sklearn')
plt.xlabel('Less Driven Car')
plt.ylabel('selling price')
plt.show()


#using seaborn for hypothesis 2nd
fig, ax1 = plt.subplots(figsize = (15,10))
sns.scatterplot(x='Y_S_L',y='Selling_Price',data = data,ax=ax1)

#3rd Hypothesis 

#count manual and autmatic cars
data.Transmission.value_counts()
data.loc[:,['Transmission','Selling_Price']].sort_values(by = ['Selling_Price'],ascending = False)['Transmission'].head(15).value_counts().plot.pie(figsize= (5,5),subplots = True,autopct = '%.1f%%',explode = [0,0.08],shadow = True )

#for Prediction
labelencoder = LabelEncoder()
df = pd.get_dummies(data['Fuel_Type'],prefix = 'FT', drop_first = True)
data['Seller_Type'] = labelencoder.fit_transform(data['Seller_Type'])
data['Transmission'] = labelencoder.fit_transform(data['Transmission'])
data = pd.concat([data,df],axis = 1)
data.drop(['Fuel_Type'],axis=1,inplace =True)
data.head()


def get_model(X,y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=10)
    lr = LinearRegression()
    
    lr.fit(X_train,y_train)
    y_pred = lr.predict(X_test)
    
    coeffecients = pd.DataFrame(lr.coef_,X.columns)
    coeffecients.columns = ['Coeffecient']
    print(f' Coefficients : \n {coeffecients} \n')
    
    mse = mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(mse)
    print(f'Mean Squared Error of Test Set : {mse}')
    print(f'Root Mean Square Error of Test Set : {rmse}')
    
    yt_pred = lr.predict(X_train)
    tmse = mean_squared_error(y_test,y_pred)
    trmse = np.sqrt(mse)
    print(f'Mean Squared Error of Train Set : {tmse}')
    print(f'Root Mean Square Error of Train Set : {trmse}')
    fig,ax1 = plt.subplots(figsize=(15,8))
    fig = sns.scatterplot(y_test,y_pred,ax=ax1)
    plt.xlabel('Y true')
    plt.ylabel('Y predicted')
    plt.title('True vs Predicted')
    plt.show(fig)
    
    fig,ax1 = plt.subplots(figsize=(15,8))
    fig = sns.distplot((y_test-y_pred),ax=ax1);
    plt.title('Residual Distrubution')
    plt.show(fig)

    
     
#model one with all features    
X = data.drop(['Car_Name','Selling_Price'],axis = 1)
y = data['Selling_Price']
get_model(X,y)

#model 2 with some features
X = data.drop(['Car_Name','Selling_Price','Kms_Driven','Seller_Type','Transmission',
               'Owner', 'Y_S_L','FT_Petrol'],axis = 1)
y = data['Selling_Price']
get_model(X,y)










