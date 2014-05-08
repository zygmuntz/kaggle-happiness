"vectorize categorical variables on the real train and test set, train an SVM, output predictions"
"inplace refers to encoding categorical variables as -1/0/1"

import numpy as np
import pandas as pd
import sqlite3

from math import sqrt
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.svm import SVC

###

# dirs need trailing slashes
data_dir = '/path/to/your/data/dir/'
predictions_dir = '/path/to/your/predictions/dir/'

train_file = data_dir + 'train.csv'
test_file = data_dir + 'test.csv'
output_file = predictions_dir + 'p_svm.csv'

###

train = pd.read_csv( train_file )
test = pd.read_csv( test_file )

# fix missing YOB

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
max_test = np.amax( x_num_test, 0 )
print "max. numerical train/test", max_train, max_test

x_num_train = x_num_train / max_train
x_num_test = x_num_test / max_train

# y

y_train = train.Happy
ids = test.UserID

# categorical

cat_train = train.drop( numeric_cols + [ 'UserID', 'Happy' ], axis = 1 )
cat_test = test.drop( numeric_cols + [ 'UserID' ], axis = 1 )


cat_train.fillna( 0, inplace = True )
cat_test.fillna( 0, inplace = True )

# Gender to -1,0,1
"""
cat_train.Gender[ cat_train.Gender == 'Male' ] = 1
cat_train.Gender[ cat_train.Gender == 'Female' ] = -1
cat_test.Gender[ cat_test.Gender == 'Male' ] = 1
cat_test.Gender[ cat_test.Gender == 'Female' ] = -1
"""

# WARNING: hacks ahead

yeses = [ 'Public', 'Science', 'Try first', 'Giving', 'Idealist', 'Cool headed', 'Odd hours', 'Happy', 'P.M.', 'Start', 'Circumstances',
	'TMI', 'Talk', 'Technology', 'Demanding', 'PC', 'Cautious', 'Yes!', 'Space', 'In-person', 'Yay people!', 'Own', 'Optimist', 'Mom', 'Nope' ]
	
"""	
# check if we got questions covered	
# possible exercise in refactoring	
for col in cat_train.columns:
	if not col.startswith( 'Q' ):
		continue
	uniq = train[col].unique()

	found = False
	for y in [ 'Yes' ] + yeses:
		if y in uniq:
			found = True
			break
	if found:
		continue
	print col, uniq
"""


# training set

# insert ones	
for col in cat_train.columns:
	if not col.startswith( 'Q' ):
		continue
	for y in yeses + [ 'Yes' ]:
		i = cat_train[col] == y
		cat_train[col][i] = 1	
		
# insert -1		
for col in cat_train.columns:
	if not col.startswith( 'Q' ):
		continue
	not_y = ~ cat_train[col].isin(( 0, 1 ))
	cat_train[col][not_y] = -1		

"""
# inspect
for col in cat_train.columns:
	if not col.startswith( 'Q' ):
		continue
	print cat_train[col].unique()
"""	

# testing set

# insert ones	
for col in cat_test.columns:
	if not col.startswith( 'Q' ):
		continue
	for y in yeses + [ 'Yes' ]:
		i = cat_test[col] == y
		cat_test[col][i] = 1	
		
# insert -1		
for col in cat_test.columns:
	if not col.startswith( 'Q' ):
		continue
	not_y = ~ cat_test[col].isin(( 0, 1 ))
	cat_test[col][not_y] = -1		

"""
# inspect
for col in cat_test.columns:
	if not col.startswith( 'Q' ):
		continue
	print cat_test[col].unique()
"""

x_cat_train = cat_train.T.to_dict().values()
x_cat_test = cat_test.T.to_dict().values()

# vectorize

vectorizer = DV( sparse = False )
vec_x_cat_train = vectorizer.fit_transform( x_cat_train )
vec_x_cat_test = vectorizer.transform( x_cat_test )

# complete x

x_train = np.hstack(( x_num_train, vec_x_cat_train ))
x_test = np.hstack(( x_num_test, vec_x_cat_test ))

x_train = np.dot( x_train, a )
x_test = np.dot( x_test, a )

if __name__ == "__main__":

	print "training..."

	C = 0.19
	gamma = 0.0028
	shrinking = True
	#auto_class_weights = False

	probability = True
	verbose = True

	svc = SVC( C = C, gamma = gamma, shrinking = shrinking, probability = probability, verbose = verbose )
	svc.fit( x_train, y_train )
	p = svc.predict_proba( x_test )

	p = p[:,1] 

	# make sure both y and p are of shape (n,1) and not (n,)
	ids_and_p = np.hstack(( ids.reshape(( -1, 1 )), p.reshape(( -1, 1 ))))
	np.savetxt( output_file, ids_and_p, fmt = [ '%d', '%.10f' ], delimiter = ',', header = 'UserID,Probability1', comments = '' )
	