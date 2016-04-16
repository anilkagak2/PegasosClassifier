import random
from collections import Counter

DELIMITER = ','
ITERATIONS = "iterations"
REGULARIZATION_CONST = "regularization_const"

# Represents a single data point
class DataPoint:
	def __init__(self, x, y):
		self.x = x
		self.y = y

# Represent the binary classifier parameters
class ClassifierParameters:
	def __init__(self, theta):
		self.theta = theta

class PegasosClassifier:
	def __init__(self):
		# we'll be using One-vs-All method for multi-class classification
		# each class will be a key in this dictionary and the value will be the parameters for that class
		self.classifiers = {}

		# Default parameters for the algorithm, to be used in case the user does not explicitly provide them
		self.DefaultRegularizationConst = 0.03
		self.DefaultIterations = 100000
	
	# dot product between the given vectors
	def _dot(self, a, b):
		if len(a) != len(b): 
			print "Cannot perform dot product of the given vectors : ", a , b 
			return

		return sum(p*q for p,q in zip(a,b))
	
	# addition operator for given two vectors
	def _add(self, a, b):
		if len(a) != len(b): 
			print "Cannot perform addition of the given vectors : ", a , b 
			return
			
		return [p+q for p,q in zip(a,b)]
	
	# returns k*a where a is a vector and k is a scalar
	def _scalarProduct(self, a, k):
		return [k*p for p in a]
	
	# Y is the list of output for the given data,
	# This function returns the unique class labels
	def _uniqueClassLabels(self, Y):
		return list(Counter(Y))

	# train the classifier on the given data with given parameters
	def fit(self, data, parameters):
		if (data is None) or (len(data)==0):
			print "Invalid training set"
			return
	
		T = int(parameters[ITERATIONS]) if ITERATIONS in parameters else self.DefaultIterations
		lmbda = float(parameters[REGULARIZATION_CONST]) if REGULARIZATION_CONST in parameters else self.DefaultRegularizationConst
		
		classes = self._uniqueClassLabels([point.y for point in data])
		dataLen = len(data)
		featureVectorLen = len( data[0].x )
		for classLabel in classes:
			# initialize the classifier theta's to 0 vector
			self.classifiers[classLabel] = ClassifierParameters([0]*featureVectorLen)
		
			# train the learner with single example till T iterations
			for t in xrange(1,T+1):
				i = random.randint(0, dataLen-1)
				etaT = 1.0 / (lmbda * t)

				xi = data[i].x
				yi = 1 if data[i].y == classLabel else -1
			
				if ( yi * self._dot(self.classifiers[classLabel].theta, xi)) < 1.0:
					self.classifiers[classLabel].theta = self._add( \
														self._scalarProduct(self.classifiers[classLabel].theta, 1.0- etaT*lmbda), \
														self._scalarProduct(xi, etaT*yi) )
				else:
					self.classifiers[classLabel].theta = self._scalarProduct(self.classifiers[classLabel].theta, 1.0- etaT*lmbda)
				
			# training complete
			print "Training complete for label : ", classLabel
			print "Theta : ", self.classifiers[classLabel].theta
		
	# predict the class label for given input
	def predict(self, x):
		if len(self.classifiers) == 0:
			print "Not trained yet"
			return

		predictions = [ (self._dot(self.classifiers[classLabel].theta, x), classLabel) for classLabel in self.classifiers ]
			
		prediction = max(predictions)
		print "class predicted : ", prediction[1], " with confidence ", prediction[0]
	
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
			
			x = [float(u) for u in x]
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

data = load_dataset("datasets/iris.data.txt", 4)
train_data, test_data = dataSplit(data, True)

clf = PegasosClassifier()
parameters =  {}
clf.fit(data, parameters)

clf.predict([4.4,2.9,1.4,0.2])
clf.predict([5.0,2.3,3.3,1.0])
clf.predict([5.9,3.0,5.1,1.8])

# TODOs
# Data normalization
# Multi class classification (Done, use only one classifier for binary classification remaining)
# Categorical variables
# Kernels
# Evaluation
# Numpy arrays for better performance/memory consumption