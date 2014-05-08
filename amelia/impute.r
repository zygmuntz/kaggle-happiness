# impute train and test (no y) with Amelia

library( Amelia )

# for parallelism
ncpus = 8

setwd( 'data' )

train = read.csv( 'train.csv' )
test = read.csv( 'test.csv' )

output_file_stem = 'train_and_test_imp'

#

cols = colnames( train )
noms = cols[ -c( which( cols == 'YOB' ), which( cols == 'votes' ), 
	which( cols == 'UserID' ), which( cols == 'Happy' )) ]				# can be done this way, can't it?
print( noms )

# join train and test - we need Happy in test

test['Happy'] = NA
data = rbind( train, test )

data[data == ''] = NA

# amelia

a.out = amelia( data, m = 8, idvars = c( 'UserID', 'Happy' ), noms = noms, parallel = 'multicore', ncpus = ncpus )

write.amelia( a.out, file.stem = output_file_stem )




