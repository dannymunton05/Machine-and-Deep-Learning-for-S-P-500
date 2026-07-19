#define the networks
#the following code is an alternative adaptation without the class approach: https://www.kaggle.com/code/fanbyprinciple/learning-pytorch-3-coding-an-rnn-gru-lstm
inputSize = X_trainingData.shape[2]
hiddenSize = 64
numLayers = 2
outputSize = 3 #output is either buy, hold, sell
rnn = nn.RNN(inputSize, hiddenSize, numLayers, batch_first=True).to(device)
gru = nn.GRU(inputSize, hiddenSize, numLayers, dropout = 0.2, batch_first=True).to(device)
rnnLinearLayer = nn.Linear(hiddenSize, outputSize).to(device)
gruLinearLayer = nn.Linear(hiddenSize, outputSize).to(device)

rnnParameters = list(rnn.parameters()) + list(rnnLinearLayer.parameters())
gruParameters = list(gru.parameters()) + list(gruLinearLayer.parameters())

lossFunction = nn.CrossEntropyLoss() #cross entropy for categorical
rnnOptimizer = torch.optim.Adam(rnnParameters, lr=0.005)
gruOptimizer = torch.optim.Adam(gruParameters, lr=0.005)

def training(model):
  trainingAccuracyList = []
  trainingLossList = []
  validationAccuracyList = []
  validationLossList = []
  if model == rnn:
    optimizer = rnnOptimizer
    linearLayer = rnnLinearLayer
  elif model == gru:
    optimizer = gruOptimizer
    linearLayer = gruLinearLayer

  for epoch in range (30):
    model.train()
    linearLayer.train()
    #forget previous errors for start of each loop
    optimizer.zero_grad()
    x, hidden = model(X_trainingData) #hidden is throwaway variable in this context
    output = x[:,-1,:] #this makes the output the final day of the 30 day window
    yi = linearLayer(output) #yi is a logit
    loss = lossFunction(yi, y_trainingData.long()) #view is used to ensure correct shape
    loss.backward() # for backpropagation
    optimizer.step()

    trainingAccuracy = (torch.argmax(yi, dim=1) ==  y_trainingData.long()).float().mean().item() #store loss and accuracy
    trainingAccuracyList.append(trainingAccuracy)
    trainingLossList.append(loss.item())

    #evaluation
    model.eval()
    with torch.no_grad():
      valX, hidden = model(X_validationData)
      valOutput = valX[:,-1,:]
      valYi = linearLayer(valOutput)

      valLoss = lossFunction(valYi, y_validationData.long())
      validationAccuracy = (torch.argmax(valYi, dim=1) ==  y_validationData.long()).float().mean().item()
    validationAccuracyList.append(validationAccuracy)
    validationLossList.append(valLoss.item())

  return model, trainingAccuracyList, trainingLossList, validationAccuracyList, validationLossList

rnnTrained = training(rnn) #updated models
gruTrained = training(gru)

def predictions(model,X_testingData,y_testingData,linearLayer): #used to get predictions for financial metrics
  model.eval()
  with torch.no_grad():
    x, hidden = model(X_testingData) #hidden is throwaway variable in this context
    output = x[:,-1,:]
    yi = linearLayer(output)
    prediction = torch.argmax(yi, dim=1)
  return prediction.cpu().numpy(), y_testingData.cpu().numpy()


