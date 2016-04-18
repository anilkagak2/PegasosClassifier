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
			- Extracted the mean of the stats using numpy.mean and printed the stat for the corresponding mean

# Evaluation Results:

Dataset :  datasets/iris.data.txt
-
| ClassifierName              | Accuracy | F1 Score | Precision | Recall |
|-----------------------------|----------|----------|-----------|--------|
| scikit-learn.svm.SVC        | 96.67    | 0.97     | 0.97      | 0.97   |
| BasicPegasos                | 85.33    | 0.85     | 0.87      | 0.87   |
| PegasosWithKernels (linear) | 82.00    | 0.82     | 0.85      | 0.84   |
| PegasosWithKernels(gaussian)| 90.00    | 0.89     | 0.90      | 0.89   |
---

Dataset :  datasets/wine.data.txt
-------------------------------------------------------------------------------
| ClassifierName              | Accuracy | F1 Score | Precision | Recall |
|-----------------------------|----------|----------|-----------|--------|
| scikit-learn.svm.SVC        | 98.33    | 0.98     | 0.98      | 0.99   |
| BasicPegasos                | 98.30    | 0.98     | 0.98      | 0.99   |
| PegasosWithKernels (linear) | 97.74    | 0.98     | 0.97      | 0.98   |
| PegasosWithKernels(gaussian)| 98.33    | 0.99     | 0.98      | 0.99   |
--- 