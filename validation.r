# random forest and naive bayes

library( randomForest )
library( e1071 )
library( caTools )

setwd( '/path/to/showofhands/data' )

data = read.csv( 'train.csv' )

# which columns are not factors?

cols = colnames( data )

for ( i in 1:length( cols )) {
    col_class = class( data[,i] )
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

# clean-up

drops = c( 'UserID' )
data = data[, !( names( data ) %in% drops )]

data$Happy = as.factor( data$Happy )

# clean up YOB

data$YOB[data$YOB < 1930] = 0
data$YOB[data$YOB > 2004] = 0
data$YOB[is.na(data$YOB)] = 0

# train / test split

p_train = 0.8
n = nrow( data )
train_len = round( n * p_train )
test_start = train_len + 1

i = sample.int( n )
train_i = i[1:train_len]
test_i = i[test_start:n]

train = data[train_i,]
test = data[test_i,]


# random forest

y_test = as.factor( test$Happy )
ntree = 100

rf = randomForest( as.factor( Happy ) ~ ., data = train, ntree = ntree, do.trace = 10 )
p <- predict( rf, test, type = 'prob' )
probs =  p[,2]

auc = colAUC( probs, y_test )
auc = auc[1]
cat( "Random forest AUC:", auc )

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

cat( "\n\n" )
cat( "Naive Bayes AUC:", auc, "\n" )



