"vectorize imputed set, predict"
"can take params from command line - used by predict_many.py"

import sys
import numpy as np
import pandas as pd
import sqlite3

from math import sqrt
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.svm import SVC

###

points_in_test = 1980
data_dir = '/path/to/amelia/output/files/'

try:
	data_file = sys.argv[1]
except IndexError:
	data_file = data_dir + 'train_and_test_imp1.csv'
	
try:
	output_file = sys.argv[2]
except IndexError:
	output_file = 'p_imp1.csv'

###

data = pd.read_csv( data_file )

# drop index column added by amelia
data = data.drop(['Unnamed: 0'], axis = 1)

train = data.iloc[:points_in_test,]			# <--- A BUG! Should be [:-points_in_test,]
test = data.iloc[-points_in_test:,]

# numeric x

numeric_cols = [ 'YOB', 'votes' ]
x_num_train = train[ numeric_cols ].as_matrix()
x_num_test = test[ numeric_cols ].as_matrix()

# scale to <0,1>

max_train = np.amax( x_num_train, 0 )
max_test = np.amax( x_num_test, 0 )

x_num_train = x_num_train / max_train
x_num_test = x_num_test / max_train

# y

y_train = train.Happy
ids = test.UserID

# categorical

cat_train = train.drop( numeric_cols + [ 'UserID', 'Happy'], axis = 1 )
cat_test = test.drop( numeric_cols + [ 'UserID', 'Happy'], axis = 1 )

x_cat_train = cat_train.T.to_dict().values()
x_cat_test = cat_test.T.to_dict().values()

# vectorize

vectorizer = DV( sparse = False )
vec_x_cat_train = vectorizer.fit_transform( x_cat_train )
vec_x_cat_test = vectorizer.transform( x_cat_test )

# complete x

x_train = np.hstack(( x_num_train, vec_x_cat_train ))
x_test = np.hstack(( x_num_test, vec_x_cat_test ))

if __name__ == "__main__":

	print "training..."

	C = 18292
	gamma = 1.88e-07
	shrinking = True
	#auto_class_weights = False

	probability = True
	verbose = True

	svc = SVC( C = C, gamma = gamma, shrinking = shrinking, probability = probability, verbose = verbose )
	svc.fit( x_train, y_train )
	p = svc.predict_proba( x_test )

	p = p[:,1] 

	ids_and_p = np.hstack(( ids.reshape(( -1, 1 )), p.reshape(( -1, 1 ))))
	np.savetxt( output_file, ids_and_p, fmt = [ '%d', '%.10f' ], delimiter = ',', header = 'UserID,Probability1', comments = '' )
	

