# random forest and naive bayes

library( randomForest )
library( e1071 )
library( caTools )

setwd( '/path/to/showofhands/data' )

# we split the train set for validation
train = read.csv( 'train_v.csv' )
test = read.csv( 'test_v.csv' )

cols = colnames( train )

# which columns are not factors?
for ( i in 1:length( cols )) {
    col_class = class( train[,i] )
    if ( col_class != 'factor' ) {
        cat( cols[i], col_class, "\n" )
    }
}

"
UserID integer 
YOB integer 
Happy integer 
votes integer
"

y = as.factor( train$Happy )
y_test = as.factor( test$Happy )

# clean up YOB for random forest
train$YOB[train$YOB < 1930] = 0
test$YOB[test$YOB < 1930] = 0

train$YOB[train$YOB > 2004] = 0
test$YOB[test$YOB > 2004] = 0

train$YOB[is.na( train$YOB )] = 0
test$YOB[is.na( test$YOB )] = 0

drops = c( 'UserID' )
train = train[, !( names( train ) %in% drops )]
test = test[, !( names( test ) %in% drops )]

# random forest

ntree = 1000

rf = randomForest( as.factor( Happy ) ~ ., data = train, ntree = ntree, do.trace = 10 )
p <- predict( rf, test, type = 'prob' )
probs =  p[,2]

auc = colAUC( probs, y_test )
auc = auc[1]
print( "Random forest AUC:", auc )

varImpPlot( rf, n.var = 20 )

# naive bayes

nb = naiveBayes( Happy ~ ., data = train )

# for predicting
drops = c( 'Happy' )
x_test = test[, !( names( test ) %in% drops )]

p = predict( nb, x_test, type = 'raw' )
probs =  p[,2]

auc = colAUC( probs, y_test )
auc = auc[1]

cat( "Naive Bayes AUC:", auc, "\n" )
# auc: 0.707359

