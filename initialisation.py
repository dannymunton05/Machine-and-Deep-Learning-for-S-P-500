import torch
import torch.nn as nn
import yfinance as yahoofinance
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #ensures T4-GPU runtime is being utilised for Google Colab

#download finanical data
sp500 = yahoofinance.download("^GSPC", start="1990-01-01", end="2019-01-01", progress = False)
nasdaq = yahoofinance.download("^IXIC", start="1990-01-01", end="2019-01-01", progress = False)
sp500.head()
nasdaqPrices = nasdaq["Close"].dropna()
volatilityIndex = yahoofinance.download("^VIX", start="1990-01-01", end="2019-01-01", progress = False)["Close"]
prices = sp500["Close"].dropna()
#define the dataframe for the feature variables
df = pd.DataFrame(index = sp500.index)
df['returns'] = np.log(prices/prices.shift(1))
df['nasdaqReturns'] = np.log(nasdaqPrices/nasdaqPrices.shift(1))
df['VIX'] = volatilityIndex.loc[sp500.index]
df['movingAverage'] = prices.rolling(5).mean()
df['RSI'] = ta.rsi(sp500['Close'].squeeze(), length=14) #14 is standard length for RSI, squeeze() makes sure data is an array
df['MACD'] = (sp500['Close'].ewm(span=12, adjust = False).mean()) - (sp500['Close'].ewm(span=26, adjust = False).mean()) #ema short - ema long (exponential moving average)
df['EMAratio'] = (sp500['Close'].ewm(span=10, adjust = False).mean()) / (sp500['Close'].ewm(span=30, adjust = False).mean())
#for Bollinger Band Width based on  ((Upper Band - Lower Band) / Middle Band) * 100
df['middleBand'] = sp500["Close"].rolling(20).mean()
df['rollingStd'] = sp500["Close"].rolling(20).std()
df['upperBand'] = df['middleBand'] + (df['rollingStd'] * 2)
df['lowerBand'] = df['middleBand'] - (df['rollingStd'] * 2)
df['BBW'] = ((df['upperBand'] - df['lowerBand'])/ df['middleBand']) * 100 #Bollinger Band Width
#above code adapted from: https://www.marketcalls.in/python/introduction-to-pandas-dataframe-python-tutorial-for-traders-part-1.html#:~:text=Creating%20a%20DataFrame%20from%20yfinance,download()%20function.&text=This%20will%20download%20the%20stock,Finance%20and%20create%20a%20DataFrame.
dfClean = df.dropna()

#create classes for the outputs
rawReturns = dfClean['returns'].shift(-1) #used for class creation
dfClean = dfClean.iloc[:-1] #data shift prevents model from looking ahead and cheating
dfClean = dfClean.dropna()
rawReturns = rawReturns.iloc[:-1]
rawReturns1 = rawReturns[19:].reset_index(drop=True)
 #this categorises returns into 3 classes, if significant return then buy class (2), if significant loss then sell class (0), else hold.
factor = 0.35
rollingVolatility = rawReturns.rolling(20).std()
rollingVolatility = rollingVolatility.dropna()
dfClean = dfClean.iloc[19:]
rawReturns = rawReturns.iloc[19:]
upper = factor * rollingVolatility #upper and lower bounds for threshold
lower = -factor * rollingVolatility

yClasses = pd.Series(1, index=rawReturns.index)
print(len(rawReturns), len(rollingVolatility), len(upper))
yClasses[rawReturns > upper] = 2
yClasses[rawReturns < lower] = 0
yClasses = yClasses[:-1]
ySplit = int(0.8 * len(yClasses))
yClasses_training = yClasses[:ySplit]
yClasses_testing = yClasses[ySplit:]

#scale the values down so VIX isnt too large in comparison to returns etc.
features = dfClean[['returns','VIX','nasdaqReturns','movingAverage','RSI','MACD','BBW', 'ADX', 'STOCHk', 'STOCHd', 'ATR']]
featuresClean = features.dropna()
featuresClean = featuresClean.iloc[:-1]
# Split the aligned features and yClasses into training and testing sets
totalSamplesAligned = len(featuresClean)
splitIndex = int(0.8 * totalSamplesAligned)
features_trainingData = featuresClean.iloc[:splitIndex]
features_testingData = featuresClean.iloc[splitIndex:]
validationSplitIndex = int(0.8 * len(features_trainingData))#split training set further into a validation set
features_train = features_trainingData.iloc[:validationSplitIndex]
features_validationData = features_trainingData.iloc[validationSplitIndex:]

#scale the values down so VIX isnt too large in comparison to returns etc.
scaler = StandardScaler()
scaledFeatures_train = pd.DataFrame(scaler.fit_transform(features_train), index=features_train.index, columns=features_train.columns)
scaledFeatures_testingData = pd.DataFrame(scaler.transform(features_testingData), index=features_testingData.index,columns=features_testingData.columns)
scaledFeatures_validationData = pd.DataFrame(scaler.transform(features_validationData),index=features_validationData.index, columns=features_validationData.columns)

yClasses_training = yClasses.iloc[:splitIndex]
yClasses_testing = yClasses.iloc[splitIndex:]
yClasses_train = yClasses_training.iloc[:validationSplitIndex]
yClasses_validationData = yClasses_training.iloc[validationSplitIndex:]
#define a 30 day window for the data
#code adapted from: https://www.geeksforgeeks.org/machine-learning/introduction-to-recurrent-neural-network/
windowSize = 60
def windowData(data, yClass, window):
  X = []
  y = []
  indexList = []

  for x in range(window, len(data)):
    X.append(data.iloc[x- window:x].values)
    y.append(yClass.iloc[x]) #X is 30 day of features, y is the 31st day
    indexList.append(yClass.index[x])
  return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.int64), indexList #returns a 3D tensor containing [batch, days, features]

#call the function to create the sequences and test the shape of X and y:
X_trainingData, y_trainingData, trainIndex = windowData(scaledFeatures_train, yClasses_train, windowSize)
X_validationData, y_validationData, validIndex = windowData(scaledFeatures_validationData, yClasses_validationData, windowSize)
X_testingData, y_testingData, testIndex = windowData(scaledFeatures_testingData, yClasses_testing, windowSize)

X_trainingData = X_trainingData.to(device)
y_trainingData = y_trainingData.to(device)
X_validationData = X_validationData.to(device)
y_validationData = y_validationData.to(device)
X_testingData = X_testingData.to(device)
y_testingData = y_testingData.to(device)
