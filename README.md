# PegasosClassifier
Implementation of the Pegasos Classifier based on http://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf


# Evaluation set-up:
	- For each dataset in (iris.data, wine.data)
		- Divided the dataset via KFold cross_validation routine available in scikit-learn
		- For each test,train split from KFold routine:
			- For each classifier in (scikit-learn.svm.SVC, BasicPegasosClassifier, PegasosClassifierWithKernels)
				- Trained the classifier on the train data
				- Extracted the accuracy, f1_score, precision, recall on the test data
				- Stored in evaluation dictionary for that classifier
		- For each classifier in (scikit-learn.svm.SVC, BasicPegasosClassifier, PegasosClassifierWithKernels)
			- Extracted the median of the stats using numpy.median and printed the stat for the corresponding median

# Evaluation Results:

Dataset :  datasets/iris.data.txt
-
| ClassifierName              | Accuracy | F1 Score | Precision | Recall |
|-----------------------------|----------|----------|-----------|--------|
| scikit-learn.svm.SVC        | 96.67    | 0.97     | 0.98      | 0.98   |
| BasicPegasos                | 86.67    | 0.86     | 0.89      | 0.88   |
| PegasosWithKernels (linear) | 86.67    | 0.87     | 0.89      | 0.87   |
| PegasosWithKernels(gaussian)| 90.00    | 0.90     | 0.92      | 0.90   |
---

Dataset :  datasets/wine.data.txt
-------------------------------------------------------------------------------
| ClassifierName              | Accuracy | F1 Score | Precision | Recall |
|-----------------------------|----------|----------|-----------|--------|
| scikit-learn.svm.SVC        | 100      | 1.0      | 1.0       | 1.0    |
| BasicPegasos                | 100      | 1.0      | 1.0       | 1.0    |
| PegasosWithKernels (linear) | 100      | 1.0      | 1.0       | 1.0    |
| PegasosWithKernels(gaussian)| 100      | 1.0      | 1.0       | 1.0    |
--- 