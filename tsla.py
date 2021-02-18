# import libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np

# get data
tickersymbol = 'TSLA'
tickerdata = yf.Ticker(tickersymbol)
stockdata = tickerdata.history(period='1d', start='2020-05-20')
sentiment = pd.read_csv('tsla_help.csv')

# merge data
sentiment['Date'] = pd.to_datetime(sentiment['Date'])
tsladata = pd.merge(stockdata, sentiment, on='Date')

# edit data
tsladata = tsladata.drop(['Dividends', 'Stock Splits'], axis=1)
tsladata['Open'] = tsladata['Open'].shift(-1)
lastrow = tsladata[-1:]
lastrow = lastrow.drop(['Open'], axis=1)
lastrow = np.float32(lastrow[['Close', 'High', 'Low', 'Volume', 'Sentiment']])
tsladata.drop(tsladata.tail(1).index, inplace=True)

# split data into y and x
y = tsladata['Open']
x = tsladata[['Close', 'High', 'Low', 'Volume', 'Sentiment']]

# split data into train and test
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.7, random_state=7)

# build model
RFR = RandomForestRegressor(random_state=7)
RFR.fit(x, y)

# predict close price for the next day
prediction = RFR.predict(lastrow)
prediction = float(prediction)
prediction = round(prediction, 2)
print('Predicted open price for TSLA is')
print('$', prediction)

# validation
RFR.fit(xtrain, ytrain)
accuracy = RFR.score(xtest, ytest)
print('Accuracy is')
print(accuracy)

# plot the data
plt.figure(figsize=(10, 5))
plt.plot(stockdata['Open'], label='TSLA stock open price')
plt.plot(stockdata['High'], 'c--', label='TSLA stock high price')
plt.plot(stockdata['Low'], 'm--', label='TSLA stock low price')
plt.plot(stockdata['Close'], 'y--', label='TSLA stock close price')
plt.xlabel('Date')
plt.ylabel('Open Price')
plt.title('TSLA Stock Price (2020-present)')
plt.legend()
plt.show()
