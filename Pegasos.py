import random
import numpy as np
from scipy.sparse import dok_matrix
import math
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.cross_validation import StratifiedShuffleSplit

DELIMITER = ','
ITERATIONS = "iterations"
REGULARIZATION_CONST = "regularization_const"
KERNEL_CONST = "kernel"

# Represents a single data point
class DataPoint:
	def __init__(self, x, y):
		self.x = x
		self.y = y

# Represent the binary classifier parameters
class ClassifierParameters:
	def __init__(self, theta, X=None, Y=None):
		self.theta = theta
		self.X = X
		self.Y = Y

# Basic Pegasos algorithm mentioned in the Paper
class PegasosClassifier:
	def __init__(self):
		# we'll be using One-vs-All method for multi-class classification
		# each class will be a key in this dictionary and the value will be the parameters for that class
		self.classifiers = {}
		self.classLabels = []

		# Default parameters for the algorithm, to be used in case the user does not explicitly provide them
		self.DefaultRegularizationConst = 0.03
		self.DefaultIterations = 100000
		self.lmbda = self.DefaultRegularizationConst
		self.T = self.DefaultIterations

		# are we dealing with binary classification problem?
		self._isBinaryClassification = False

	# Parse the parameters dictionary passed by the user
	def _parseParameters(self, parameters):
		self.T = int(parameters[ITERATIONS]) if ITERATIONS in parameters else self.DefaultIterations
		self.lmbda = float(parameters[REGULARIZATION_CONST]) if REGULARIZATION_CONST in parameters else self.DefaultRegularizationConst
		self.classLabels = self._uniqueClassLabels([point.y for point in data])
		self._isBinaryClassification = True if len(self.classLabels)==2 else False

	# Y is the list of output for the given data,
	# This function returns the unique class labels
	def _uniqueClassLabels(self, Y):
		return list(Counter(Y))

	# default classifier parameters
	def _defaultClassifierParameters(self, dataLen, featureVectorLen):
		return ClassifierParameters( np.zeros((featureVectorLen)) )

	# update the classifier for classLabel with ith data point (xi,yi)
	def _updateClassifierWithDataPoint(self, classLabel, i, xi, yi, etaT):
		if ( yi * np.dot(self.classifiers[classLabel].theta, xi)) < 1.0:
			self.classifiers[classLabel].theta = (1.0- etaT*lmbda)*self.classifiers[classLabel].theta + (etaT*yi)*xi
		else:
			self.classifiers[classLabel].theta = (1.0- etaT*lmbda)*self.classifiers[classLabel].theta

	# train the classifier on the given data with given parameters
	def fit(self, data, parameters):
		if (data is None) or (len(data)==0):
			print "Invalid training set"
			return

		self._parseParameters(parameters)

		dataLen = len(data)
		featureVectorLen = len( data[0].x )
		for classLabel in self.classLabels:
			# initialize the classifier theta's to 0 vector
			self.classifiers[classLabel] = self._defaultClassifierParameters(dataLen, featureVectorLen)
		
			# train the learner with single example till T iterations
			for t in xrange(1,self.T+1):
				i = random.randint(0, dataLen-1)
				etaT = 1.0 / (self.lmbda * t)
				xi = data[i].x
				yi = 1 if data[i].y == classLabel else -1
				self._updateClassifierWithDataPoint(classLabel, i, xi, yi, etaT)

			# training complete
			print "Training complete for label : ", classLabel
			#print "Theta : ", self.classifiers[classLabel].theta

			# Break after training one classifier for binary classification
			if self._isBinaryClassification: break

	# return (confidence for class, class Label)
	def _classLabelConfidenceForInput(self, classLabel, x):
		return (np.dot(self.classifiers[classLabel].theta, x), classLabel)

	# predict the class label for given input
	def predict(self, x):
		if len(self.classifiers) == 0:
			print "Not trained yet"
			return

		predictions = []
		if self._isBinaryClassification:
			classOne, classTwo = self.classLabels[0], self.classLabels[1]

			# while training we break the loop after training for the classOne
			confidence, _ = self._classLabelConfidenceForInput(classOne, x)
			predictedClass = classOne if confidence>0 else classTwo
			prediction = (confidence, predictedClass)
		else:
			predictions = []
			for classLabel in self.classifiers:
				predictions.append( self._classLabelConfidenceForInput(classLabel, x) )

			prediction = max(predictions)

		#print "class predicted : ", prediction[1], " with confidence ", prediction[0]
		return prediction[1]

# Linear kernel
def _linear(x1,x2):
	return np.dot(x1,x2)

# Gaussian kernel
def _gaussian(x1,x2):
	const_sigma = 0.1
	differenceNorm = np.linalg.norm(x1-x2)
	return (math.e) ** ( differenceNorm / (2*(const_sigma**2)) )
	
# Pegasos algorithm with kernel tricks
class PegasosClassifierWithKernels(PegasosClassifier):
	def __init__(self):
		PegasosClassifier.__init__(self)

		# override the default iterations
		self.DefaultIterations = 1000

		# kernel function pointer, used to calculate similarity between data points
		self.kernel = _linear

	# Parse the parameters passed by user
	def _parseParameters(self, parameters):
		PegasosClassifier._parseParameters(self, parameters)
		self.kernel = parameters[KERNEL_CONST] if KERNEL_CONST in parameters else _linear

	# default classifier parameters
	def _defaultClassifierParameters(self, dataLen, featureVectorLen):
		return ClassifierParameters( dok_matrix((1,dataLen)), {}, dok_matrix((1,dataLen)) )

	# update the classifier for classLabel with ith data point (xi,yi)
	def _updateClassifierWithDataPoint(self, classLabel, i, xi, yi, etaT):
		comp = 0
		for ((r,c),alpha) in self.classifiers[classLabel].theta.iteritems():
			yj = self.classifiers[classLabel].Y[r,c]
			xj = self.classifiers[classLabel].X[c]
			comp += alpha * yj * self.kernel(xi, xj)

		if (yi*etaT*comp) < 1.0:
			if self.classifiers[classLabel].theta[0,i] == 0.0:
				self.classifiers[classLabel].X[i] = xi
				self.classifiers[classLabel].Y[0,i] = yi

			self.classifiers[classLabel].theta[0,i] += 1

	# return (confidence for class, class Label)
	def _classLabelConfidenceForInput(self, classLabel, x):
		confidence = 0.0
		for ((r,c),alpha) in self.classifiers[classLabel].theta.iteritems():
			yj = self.classifiers[classLabel].Y[r,c]
			xj = self.classifiers[classLabel].X[c]
			confidence += alpha * yj * self.kernel(x, xj)

		return (confidence, classLabel)


# returns data read from the file, treating each line as a data point
# classLabelIndex denotes the zero based index of the column containing the class label
def load_dataset(fileName, classLabelIndex):
	data = []
	with open(fileName) as file:
		for line in file:
			line = line.rstrip('\n')
			
			if line == "" : continue;
			
			features = line.split(DELIMITER)
			x = features[ : classLabelIndex ] + features[ classLabelIndex+1 : ]
			
			x = np.array( [float(u) for u in x] )
			y = features[classLabelIndex]
			data.append(DataPoint(x, y));

	return data

def dataSplit(data, randomShuffle):
	# perform random shuffle of the data, so that we do not have the same class clustered at same point
	if randomShuffle: random.shuffle(data)
	
	dataLen = len(data)
	trainDataSize = int(0.9*dataLen)
	train_data = data[ : trainDataSize]
	test_data = data[trainDataSize : ]
	return train_data, test_data	

def accuracy(predictedValues, actualValues):
	return sum([pv==av for pv,av in zip(predictedValues, actualValues)]) * 100.0 / len(predictedValues)

# perform the hypothesis evaluation
def hypothesisEvaluation(classifier, testData):
	predictedValues = []
	actualValues = []
	for dataPoint in testData:
		actualValues.append(dataPoint.y)
		predictedValues.append(classifier.predict(dataPoint.x))

	print "accuracy : ", accuracy(predictedValues, actualValues)
	print "accuracy_score : ", accuracy_score(actualValues, predictedValues)
	print "f1_score : ", f1_score(actualValues, predictedValues, average="macro")
	print "precision_score : ", precision_score(actualValues, predictedValues, average="macro")
	print "recall_score : ", recall_score(actualValues, predictedValues, average="macro")

data = load_dataset("datasets/iris.data.txt", 4)
trainData, testData = dataSplit(data, True)

#classifier = PegasosClassifier()
classifier = PegasosClassifierWithKernels()
parameters =  {}
classifier.fit(trainData, parameters)

hypothesisEvaluation(classifier, testData)

classifier.predict(np.array([4.4,2.9,1.4,0.2]))
classifier.predict(np.array([5.0,2.3,3.3,1.0]))
classifier.predict(np.array([5.9,3.0,5.1,1.8]))

# TODOs
# Preprocessing
#	- Data normalization
#	- Categorical variables
# Pegasos Learning Algorithm
#	- DONE Basic Pegasos for binary classification
# 	- DONE Multi class classification (one-vs-all)
#	- SKIPPED Optional step for the parameters
#	- DONE Kernels (Fix the accuracy)
#	- SKIPPED introduction of b (Pegasos measurements do not include this in their implementation)
# Evaluation
#	- Test/Train separation
#	- Score: Accuracy, F1 score, Precision/Recall
#	- KFold cross_validation
#	- Publish numbers for standard SVC from scikit-learn, PegasosClassifier, PegasosClassifierWithKernels for Wine & Iris
# Optimizations
#	- DONE Numpy arrays for better performance/memory consumption
#	- DONE Sparse vectors
#	- Sparse dot product
#	- DONE Refactor the code for PegasosClassifier and PegasosClassifierWithKernels using inheritance