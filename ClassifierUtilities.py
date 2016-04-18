import numpy as np
import math

# Represent the binary classifier parameters
class ClassifierParameters:
	def __init__(self, theta, X=None, Y=None):
		self.theta = theta
		self.X = X
		self.Y = Y

#
# Kernels
#
# Linear kernel
def _linear(x1,x2):
	return np.dot(x1,x2)

# Gaussian kernel
def _gaussian(x1,x2):
	const_sigma = 2
	differenceNorm = np.linalg.norm(x1-x2)
	return (math.e) ** ( -1.0 * differenceNorm / (2*(const_sigma**2)) )