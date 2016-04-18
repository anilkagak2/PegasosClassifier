import ClassifierUtilities
import Pegasos
from scipy.sparse import dok_matrix

KERNEL_CONST = "kernel"

# Pegasos algorithm with kernel tricks
class PegasosClassifierWithKernels(Pegasos.PegasosClassifier):
	def __init__(self, parameters={}):
		Pegasos.PegasosClassifier.__init__(self, parameters)

		# override the default iterations
		self.DefaultIterations = 1000
		self.DefaultRegularizationConst = 0.03

		# kernel function pointer, used to calculate similarity between data points
		self.kernel = ClassifierUtilities._linear
		self._parseParameters(parameters)

	# Parse the parameters passed by user
	def _parseParameters(self, parameters):
		Pegasos.PegasosClassifier._parseParameters(self, parameters)
		self.kernel = parameters[KERNEL_CONST] if KERNEL_CONST in parameters else ClassifierUtilities._linear

	# default classifier parameters
	def _defaultClassifierParameters(self, dataLen, featureVectorLen):
		return ClassifierUtilities.ClassifierParameters( dok_matrix((1,dataLen)), {}, dok_matrix((1,dataLen)) )

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