# My project is on building Linear regression model to predict stock price of Tesla
# Downloaded TSLA.csv from https://finance.yahoo.com/quote/TSLA/history?p=TSLA

import pandas as pd
df=pd.read_csv('TSLA.csv')  # importing the csv file as pandas data frame
print(df)

# trimming the data frame to get date and adj close columns only
df.set_index(pd.DatetimeIndex(df['Date']), inplace=True)
df = df[['Adj Close']]
print(df)

# adding another column of exponential moving average for 10 days
import pandas_ta    # pandas_ta is library for technical analysis
df.ta.ema(close='Adj Close', length=10, append=True)
print(df)

# removing NaN values
df=df.iloc[10:]
print(df)

from sklearn import model_selection
# splitting the data into train test data of size 80:20
X_train,X_test,y_train,y_test=model_selection.train_test_split(df[['Adj Close']], df[['EMA_10']], test_size=0.2)

from sklearn.linear_model import LinearRegression

# Creating Regression Model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Using model to make predictions
y_pred = model.predict(X_test)

c=X_test.copy()
c["Prediction"]=y_pred
c                            # We can see that our prediction is quite close to Adj Close

# checking the accuracy of prediction
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
print("Model Coefficients:", model.coef_)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Coefficient of Determination:", r2_score(y_test, y_pred))
