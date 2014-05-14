Predicting happiness from demographics and poll answers
=======================================================

Code for [The Analytics Edge (15.071x) competition](http://www.kaggle.com/c/the-analytics-edge-mit-15-071x). 

	amelia - impute missing values, train many models, predict, bag
	validation.r - split the training set for validation, train and score random forest and naive Bayes,
		plot variable importance from random forest
	vectorize_and_predict_inplace.py - convert categorical to -1/0/1, train, write predictions	
	vectorize_validation.py - convert data to numbers only, train, get validation score
	
Get 0.74568 public / 0.77761 private AUC with `vectorize_and_predict_inplace.py` and even better score with Amelia.
	
For description, see: 

* [http://fastml.com/predicting-happiness-from-demographics-and-poll-answers/](http://fastml.com/predicting-happiness-from-demographics-and-poll-answers/)
* [http://fastml.com/converting-categorical-data-into-numbers-with-pandas-and-scikit-learn/](http://fastml.com/converting-categorical-data-into-numbers-with-pandas-and-scikit-learn/)
* [http://fastml.com/impute-missing-values-with-amelia/](http://fastml.com/impute-missing-values-with-amelia/)
