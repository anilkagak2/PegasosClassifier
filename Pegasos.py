import random
import numpy as np
import ClassifierUtilities
from collections import Counter

ITERATIONS = "iterations"
REGULARIZATION_CONST = "regularization_const"

# Basic Pegasos algorithm mentioned in the Paper
class PegasosClassifier:
	def __init__(self, parameters={}):
		# we'll be using One-vs-All method for multi-class classification
		# each class will be a key in this dictionary and the value will be the parameters for that class
		self.classifiers = {}
		self.classLabels = []

		# Default parameters for the algorithm, to be used in case the user does not explicitly provide them
		self.DefaultRegularizationConst = 0.003
		self.DefaultIterations = 100000
		self.lmbda = self.DefaultRegularizationConst
		self.T = self.DefaultIterations

		# are we dealing with binary classification problem?
		self._isBinaryClassification = False
		self._parseParameters(parameters)

	# Parse the parameters dictionary passed by the user
	def _parseParameters(self, parameters):
		self.T = int(parameters[ITERATIONS]) if ITERATIONS in parameters else self.DefaultIterations
		self.lmbda = float(parameters[REGULARIZATION_CONST]) if REGULARIZATION_CONST in parameters else self.DefaultRegularizationConst

	# Y is the list of output for the given data,
	# This function returns the unique class labels
	def _uniqueClassLabels(self, Y):
		return list(Counter(Y))

	# default classifier parameters
	def _defaultClassifierParameters(self, dataLen, featureVectorLen):
		return ClassifierUtilities.ClassifierParameters( np.zeros((featureVectorLen)) )

	# update the classifier for classLabel with ith data point (xi,yi)
	def _updateClassifierWithDataPoint(self, classLabel, i, xi, yi, etaT):
		if ( yi * np.dot(self.classifiers[classLabel].theta, xi)) < 1.0:
			self.classifiers[classLabel].theta = (1.0- etaT*self.lmbda)*self.classifiers[classLabel].theta + (etaT*yi)*xi
		else:
			self.classifiers[classLabel].theta = (1.0- etaT*self.lmbda)*self.classifiers[classLabel].theta

	# train the classifier on the given data with given parameters
	def fit(self, X, Y, parameters={}):
		if (X is None) or (len(X)==0) or (Y is None) or (len(X) != len(Y)):
			print "Invalid training set"
			return

		self.classLabels = self._uniqueClassLabels(Y)
		self._isBinaryClassification = True if len(self.classLabels)==2 else False

		dataLen = len(X)
		featureVectorLen = len( X[0] )
		for classLabel in self.classLabels:
			# initialize the classifier theta's to 0 vector
			self.classifiers[classLabel] = self._defaultClassifierParameters(dataLen, featureVectorLen)
		
			# train the learner with single example till T iterations
			for t in xrange(1,self.T+1):
				i = random.randint(0, dataLen-1)
				etaT = 1.0 / (self.lmbda * t)
				xi = X[i]
				yi = 1 if Y[i] == classLabel else -1
				self._updateClassifierWithDataPoint(classLabel, i, xi, yi, etaT)

			# training complete
			#print "Training complete for label : ", classLabel
			#print "Theta : ", self.classifiers[classLabel].theta

			# Break after training one classifier for binary classification
			if self._isBinaryClassification: break

	# return (confidence for class, class Label)
	def _classLabelConfidenceForInput(self, classLabel, x):
		return (np.dot(self.classifiers[classLabel].theta, x), classLabel)

	# predict the class label for given input
	def predict(self, X):
		if len(self.classifiers) == 0:
			print "Not trained yet"
			return

		predictedValues = []
		for x in X:
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
			predictedValues.append( prediction[1] )

		# return the predicted values for all X
		return predictedValues

