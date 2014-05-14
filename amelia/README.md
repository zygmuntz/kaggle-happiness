Amelia
======

Impute missing values, train many models, predict, bag. See [http://fastml.com/impute-missing-values-with-amelia/](http://fastml.com/impute-missing-values-with-amelia/) for description.

	impute.r
	predict_many.py
		vectorize_and_predict.py
	bag_imputed.py
	
Imputed 152 datasets, got 0.74764 public / 0.77891 private AUC (17th place out of 1686 competitors).

Correct the bug in `vectorize_and_predict.py` (it's marked in the code) to get 0.75229 public / __0.78467__ private with 96 sets.
	