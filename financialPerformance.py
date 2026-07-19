import matplotlib.pyplot as plt
#code adapted from: #https://www.geeksforgeeks.org/python/automated-trading-using-python/
rnnPredictions, yTest = predictions(rnnTrained[0], X_testingData, y_testingData,rnnLinearLayer)
gruPredictions, _ = predictions(gruTrained[0], X_testingData, y_testingData, gruLinearLayer)
rawReturnsOriginal = rawReturns.copy()
rawReturnsAlign =  rawReturnsOriginal.iloc[windowSize:]

data = pd.DataFrame(index = testIndex)
data['GRUSignal'] = gruPredictions
data['RNNSignal'] = rnnPredictions
data['Returns'] = rawReturns.loc[testIndex]
data['GRUPosition'] = data['GRUSignal'].map({0:-1, 1:0, 2:1}) #map buy hold sell to sell now -1, buy 1
data['GRUPosition'] = data['GRUPosition'].shift(1).fillna(0) # to counter look-ahead bias
data['RNNPosition'] = data['RNNSignal'].map({0:-1, 1:0, 2:1}) #map buy hold sell to sell now -1, buy 1
data['RNNPosition'] = data['RNNPosition'].shift(1).fillna(0)

data['GRUStrategyReturn'] = data['GRUPosition'] * data['Returns']
data['RNNStrategyReturn'] = data['RNNPosition'] * data['Returns']
# cumulative returns
data['CumulativeMarket'] = (1 + data['Returns']).cumprod()
data['GRUCumulativeStrategy'] = (1 + data['GRUStrategyReturn']).cumprod()
data['RNNCumulativeStrategy'] = (1 + data['RNNStrategyReturn']).cumprod()
plt.figure(figsize=(14, 7))
plt.plot(data['CumulativeMarket'], label='Market Return', alpha=0.75)
plt.plot(data['GRUCumulativeStrategy'], label='GRU Strategy Return', alpha=0.75)
plt.plot(data['RNNCumulativeStrategy'], label='RNN Strategy Return', alpha=0.75)
plt.title("Cumulative Returns")
plt.legend()
plt.show()

GRUReturn = data['GRUCumulativeStrategy'].iloc[-1] - 1
RNNReturn = data['RNNCumulativeStrategy'].iloc[-1] - 1
total_market_return = data['CumulativeMarket'].iloc[-1] - 1

print(f"GRU Strategy Return: {GRUReturn:.2%}")
print(f"RNN Strategy Return: {RNNReturn:.2%}")
print(f"Total Market Return: {total_market_return:.2%}")
#https://www.geeksforgeeks.org/python/automated-trading-using-python/
