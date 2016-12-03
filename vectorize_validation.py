"""
vectorize categorical variables
optionally train an SVM and a random forest, get validation AUC

importing from another script:
from vectorize_validation import y_train, x_train, y_test, x_test
"""

import numpy as np
import pandas as pd
import sqlite3

from math import sqrt
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score as AUC

###

data_dir = '/path/to/your/data/dir/'	# needs trailing slash

# validation split, both files with headers and the Happy column
train_file = data_dir + 'train_v.csv'
test_file = data_dir + 'test_v.csv'

###

train = pd.read_csv( train_file )
test = pd.read_csv( test_file )

# set missing YOB to zero

train.YOB[ train.YOB.isnull() ] = 0
train.YOB[train.YOB < 1920] = 0
train.YOB[train.YOB > 2004] = 0

test.YOB[ test.YOB.isnull() ] = 0
test.YOB[test.YOB < 1920] = 0
test.YOB[test.YOB > 2004] = 0

# numeric x

numeric_cols = [ 'YOB', 'votes' ]
x_num_train = train[ numeric_cols ].as_matrix()
x_num_test = test[ numeric_cols ].as_matrix()

# scale to <0,1>

max_train = np.amax( x_num_train, 0 )
max_test = np.amax( x_num_test, 0 )		# not really needed

x_num_train = x_num_train / max_train
x_num_test = x_num_test / max_train		# scale test by max_train

# y

y_train = train.Happy
y_test = test.Happy

# categorical

cat_train = train.drop( numeric_cols + [ 'UserID', 'Happy'], axis = 1 )
cat_test = test.drop( numeric_cols + [ 'UserID', 'Happy'], axis = 1 )

cat_train.fillna( 'NA', inplace = True )
cat_test.fillna( 'NA', inplace = True )

x_cat_train = cat_train.to_dict( orient = 'records' )
x_cat_test = cat_test.to_dict( orient = 'records' )

# vectorize

vectorizer = DV( sparse = False )
vec_x_cat_train = vectorizer.fit_transform( x_cat_train )
vec_x_cat_test = vectorizer.transform( x_cat_test )

# complete x

x_train = np.hstack(( x_num_train, vec_x_cat_train ))
x_test = np.hstack(( x_num_test, vec_x_cat_test ))



if __name__ == "__main__":

	# SVM looks much better in validation

	print "training SVM..."
	
	# although one needs to choose these hyperparams
	C = 173
	gamma = 1.31e-5
	shrinking = True

	probability = True
	verbose = True

	svc = SVC( C = C, gamma = gamma, shrinking = shrinking, probability = probability, verbose = verbose )
	svc.fit( x_train, y_train )
	p = svc.predict_proba( x_test )	
	
	auc = AUC( y_test, p[:,1] )
	print "SVM AUC", auc	
	

	print "training random forest..."

	n_trees = 100
	max_features = int( round( sqrt( x_train.shape[1] ) * 2 ))		# try more features at each split
	max_features = 'auto'
	verbose = 1
	n_jobs = 1

	rf = RF( n_estimators = n_trees, max_features = max_features, verbose = verbose, n_jobs = n_jobs )
	rf.fit( x_train, y_train )

	p = rf.predict_proba( x_test )

	auc = AUC( y_test, p[:,1] )
	print "RF AUC", auc

	# AUC 0.701579086548
	# AUC 0.676126704696

	# max_features * 2
	# AUC 0.710060065732
	# AUC 0.706282346719


