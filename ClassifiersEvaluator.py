import ClassifierUtilities
import numpy as np
import Pegasos
import PegasosWithKernels
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.cross_validation import KFold

DELIMITER = ','
NUM_FOLDS = 10

# Represents a single data point
class DataPoint:
	def __init__(self, x, y):
		self.x = x
		self.y = y
		
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

# perform the hypothesis evaluation
def hypothesisEvaluation(classifier, testX, testY):
	predictedValues = [ classifier.predict([x]) for x in testX ]

	accuracy = accuracy_score(testY, predictedValues)*100
	f1 = f1_score(testY, predictedValues, average="macro")
	precision = precision_score(testY, predictedValues, average="macro")
	recall = recall_score(testY, predictedValues, average="macro")
	#print "accuracy : ", accuracy
	#print "f1_score : ", f1
	#print "precision_score : ", precision
	#print "recall_score : ", recall

	return [accuracy, f1, precision, recall]

# For each problem, read input -> normalize features -> train on classifier -> evaluate
problems = [ ("datasets/iris.data.txt", 4), ("datasets/wine.data.txt", 0)]
for fileName, targetFeatureIndex in problems:
	print "Dataset : ", fileName
	data = load_dataset(fileName, targetFeatureIndex)

	# Split the data into training and testing (random shuffle to remove the order available in the training data)
	# Examples in one class are all together => random shuffle to mix the order
	kf = KFold(len(data), n_folds=NUM_FOLDS, shuffle=True)
	X = np.array( [point.x for point in data] )
	Y = np.array( [point.y for point in data] )
	evaluationScores = {"BasicPegasos" : [], "PegasosWithLinearKernel" : [], "PegasosWithGaussianKernel" : [], "scikit-learn.svm.SVC" : []}
	for train_index, test_index in kf:
		trainX = X[train_index]
		trainY = Y[train_index]
		testX = X[test_index]
		testY = Y[test_index]

		# Normalize the input data
		scaler = preprocessing.StandardScaler().fit(trainX)
		trainX = scaler.transform(trainX)
		testX = scaler.transform(testX)

		classifiers = {}
		classifiers["BasicPegasos"] = Pegasos.PegasosClassifier()
		classifiers["PegasosWithLinearKernel"] = PegasosWithKernels.PegasosClassifierWithKernels()
		classifiers["PegasosWithGaussianKernel"] = PegasosWithKernels.PegasosClassifierWithKernels(parameters={PegasosWithKernels.KERNEL_CONST: ClassifierUtilities._gaussian})
		classifiers["scikit-learn.svm.SVC"] = svm.SVC()

		for classifierName in classifiers:
			#print "Getting stats for : ", classifierName
			classifiers[classifierName].fit(trainX, trainY)
			evaluationScores[classifierName].append( hypothesisEvaluation( classifiers[classifierName], testX, testY ) )

	print "Evaluation results : "
	for classifierName in evaluationScores:
		print "ClassifierName : ", classifierName
		statsArray  = np.array( evaluationScores[classifierName] )
		print "mean of the stats : ", np.mean(statsArray, axis=0)
		print "median of the stats : ", np.median(statsArray, axis=0)

# TODOs
# Preprocessing
#	- DONE Data normalization (uses scikit-learn preprocessing)
#	- SKIPPED Categorical variables (can be done via scikit-learn preprocessing)
# Pegasos Learning Algorithm
#	- DONE Basic Pegasos for binary classification
# 	- DONE Multi class classification (one-vs-all)
#	- SKIPPED Optional step for the parameters
#	- DONE Kernels (Fix the accuracy)
#	- SKIPPED introduction of b (Pegasos measurements do not include this in their implementation)
#	- DONE Make training method 'fit' to take similar arguments as SVC from scikit-learn
# Evaluation
#	- DONE Test/Train separation
#	- DONE Score: Accuracy, F1 score, Precision/Recall
#	- DONE KFold cross_validation
#	- DONE Publish numbers for standard SVC from scikit-learn, PegasosClassifier, PegasosClassifierWithKernels for Wine & Iris
# Optimizations
#	- DONE Numpy arrays for better performance/memory consumption
#	- DONE Sparse vectors
#	- Sparse dot product for PegasosClassifierWithKernels
#	- DONE Refactor the code for PegasosClassifier and PegasosClassifierWithKernels using inheritance