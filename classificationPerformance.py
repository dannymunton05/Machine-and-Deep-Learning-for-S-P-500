import matplotlib.pyplot as plt

gruModel,gruTrainingAccuracy, gruTrainingLoss, gruValidationAccuracy, gruValidationLoss= rnnTrained
rnnModel,rnnTrainingAccuracy, rnnTrainingLoss, rnnValidationAccuracy, rnnValidationLoss = gruTrained

plt.figure(figsize=(8,5))
plt.plot(range(len(gruTrainingAccuracy)), gruTrainingAccuracy, label='Training Accuracy')
plt.plot(range(len(gruValidationAccuracy)), gruValidationAccuracy, label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Model Accuracy")
plt.title("GRU")
plt.legend()
plt.show()

plt.figure(figsize=(8,5))
plt.plot(range(len(rnnTrainingAccuracy)), rnnTrainingAccuracy, label='Training Accuracy')
plt.plot(range(len(rnnValidationAccuracy)), rnnValidationAccuracy, label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Model Accuracy")
plt.title("RNN")
plt.legend()
plt.show()

plt.figure(figsize=(8,5))
plt.plot(range(len(gruTrainingLoss)), gruTrainingLoss, label='Training Loss')
plt.plot(range(len(gruValidationLoss)), gruValidationLoss, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Model Loss")
plt.title("GRU")
plt.legend()
plt.show()

plt.figure(figsize=(8,5))
plt.plot(range(len(rnnTrainingLoss)), rnnTrainingLoss, label='Training Loss')
plt.plot(range(len(rnnValidationLoss)), rnnValidationLoss, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Model Loss")
plt.title("RNN")
plt.legend()
plt.show()
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import seaborn as sns
#code adapted from: https://www.geeksforgeeks.org/machine-learning/confusion-matrix-machine-learning/
rnnPrediction, yActual = predictions(rnnTrained[0], X_testingData, y_testingData, rnnLinearLayer)
gruPrediction, _ = predictions(gruTrained[0], X_testingData, y_testingData, gruLinearLayer)

print("RNN Macro F1:", f1_score(yActual, rnnPrediction, average='macro'))
print("GRU Macro F1:", f1_score(yActual, gruPrediction, average='macro'))

print("Classification Report for RNN:")
print(classification_report(yActual, rnnPrediction, target_names = ["Sell","Hold","Buy"]))

print("Classification Report for GRU:")
print(classification_report(yActual, gruPrediction, target_names = ["Sell","Hold","Buy"]))

confusionMatrixRNN = confusion_matrix(yActual, rnnPrediction)
confusionMatrixGRU = confusion_matrix(yActual, gruPrediction)

sns.heatmap(confusionMatrixRNN, annot = True, fmt = "d", xticklabels = ["Sell","Hold","Buy"], yticklabels = ["Sell","Hold","Buy"])
plt.title("RNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

sns.heatmap(confusionMatrixGRU, annot = True, fmt = "d", xticklabels = ["Sell","Hold","Buy"], yticklabels = ["Sell","Hold","Buy"])
plt.title("GRU Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
