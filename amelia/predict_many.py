import os

n_imputations = 8

for i in range( 1, n_imputations + 1 ):

	cmd = "python vectorize_and_predict.py imputed/train_and_test_imp{}.csv predictions/imputed/p{}.csv".format( i, i )
	
	print cmd
	os.system( cmd )
	
	