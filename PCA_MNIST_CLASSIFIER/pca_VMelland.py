import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
# PCA - Linear dimensionality reduction using Singular Value Decomposition 
# of the data to project it to a lower dimensional space.
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import accuracy_score
#I think I'm going to use naive bayes for the classifier model
#X is images, Y is labels. Match images to labels

(trainX, trainy), (testX, testy) = mnist.load_data()
print('Train: X = %s, y = %s' % (trainX.shape, trainy.shape))
print('Test: X = %s, y = %s' % (testX.shape, testy.shape))
#returns the right shape and sizes of the data 

#plot the first 9 images to check if importing data worked
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(trainX[i], cmap = plt.get_cmap('gray'))
plt.show()    
#import worked, first 9 images plotted

# I want the 28*28 columns to be reshaped to one 784 size dimension
xtrain = np.reshape(trainX, (60000,784))
xtest = np.reshape(testX, (10000,784))

#classifier model class
class NaiveBayes(object):
    def fit(self, Xtrain, Ytrain, smoothing = 1e-2):
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Ytrain)
        for c in labels:
            current_x = Xtrain[Ytrain == c]
            self.gaussians[c] = {
                'mean': np.mean(current_x, axis=0),
                'var': np.var(current_x, axis=0) + smoothing,
            }
            self.priors[c] = float(len(Ytrain[Ytrain == c])) / len(Ytrain)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K))
        for c, g in dict.items(self.gaussians):
            mean, var = g['mean'], g['var']
            P[:,c] = stats.multivariate_normal.logpdf(X, mean=mean, cov=var) + np.log(self.priors[c])
        return np.argmax(P, axis=1)
#end Naive Bayes class
    
model = NaiveBayes()
model.fit(xtrain, trainy)

print("Train accuracy:", model.score(xtrain, trainy))
print("Test accuracy:", model.score(xtest, testy))
# model.plots()
ypred = model.predict(xtest)
#train and test accuracy before using PCA was .6146 and .6129
ypred
print(accuracy_score(testy, ypred), ": is the accuracy score")

matrix = confusion_matrix(testy, ypred)
print(matrix)
sns.heatmap(matrix, annot=True)
plt.xlabel('True class value')
plt.ylabel('Predicted class value')
plt.title('Confusion Matrix for Dimension = 784')
plt.show()

#use PCA to reduce the dimension of the input 150,100 or 50 and train/test
#the new model that recieves the result of the PCA as input and ouputs
#the label/class of the image

#Standardize the data (not really needed here but that's ok)
scaler = StandardScaler()
scaler.fit(xtrain)
x_trainpca = scaler.transform(xtrain)
x_testpca = scaler.transform(xtest)

print(x_trainpca.shape)
print(x_testpca.shape)
#Apply PCA reduction to 150 
pca = PCA(n_components=150)
pca.fit(x_trainpca)
x_trainpca = pca.transform(x_trainpca)
x_testpca = pca.transform(x_testpca)
print(x_trainpca.shape)
print(x_testpca.shape)

#trying new data on model
model.fit(x_trainpca, trainy)

print("Train accuracy:", model.score(x_trainpca, trainy))
print("Test accuracy:", model.score(x_testpca, testy))

ypred2 = model.predict(x_testpca)
print(ypred2)
print(accuracy_score(testy, ypred2), ": is the accuracy score")

matrix2 = confusion_matrix(testy, ypred2)
print(matrix2)
sns.heatmap(matrix2, annot=True)
plt.xlabel('True class value')
plt.ylabel('Predicted class value')
plt.title('Confusion Matrix for Dimension = 150')
plt.show()

#Let's try doing a lower dimension, lets try 50. 
print(x_trainpca.shape)
print(x_testpca.shape)
#Apply PCA
pca2 = PCA(n_components=50)
pca2.fit(x_trainpca)
x_trainpca = pca2.transform(x_trainpca)
x_testpca = pca2.transform(x_testpca)
print(x_trainpca.shape)
print(x_testpca.shape)

#trying new data on model
model.fit(x_trainpca, trainy)

print("Train accuracy:", model.score(x_trainpca, trainy))
print("Test accuracy:", model.score(x_testpca, testy))

ypred2 = model.predict(x_testpca)

print(ypred2)
print(accuracy_score(testy, ypred2), ": is the accuracy score")

matrix2 = confusion_matrix(testy, ypred2)
print(matrix2)
sns.heatmap(matrix2, annot=True)
plt.xlabel('True class value')
plt.ylabel('Predicted class value')
plt.title('Confusion Matrix for Dimension = 50')
plt.show()
