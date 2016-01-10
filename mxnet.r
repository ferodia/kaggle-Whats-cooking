#!/usr/bin/env Rscript

### LOAD LIBRARIES ###
print("Loading libraries ...")
library(jsonlite)
library("NLP")
library("tm")
library("SnowballC") 
library("mxnet")
library("forecast")


### LOAD DATA ###
print("### LOAD DATA ###")

train_raw  <- fromJSON("data/trainclean.json", flatten = TRUE)
test_raw  <- fromJSON("data/testclean.json", flatten = TRUE)

### PREPROCESS DATA ###
print("### PREPROCESS DATA ###")

##lower case
docs <- c(Corpus(VectorSource(train_raw$ingredients)),Corpus(VectorSource(test_raw$ingredients)))
docs <- tm_map(docs,tolower)

##remove dash "-"
docs <- tm_map(docs ,FUN=function(x) gsub("-", "_", x) )
# as.character(docs[[10]]) # inspect the dash 

##only regular characters
docs <- tm_map(docs ,FUN=function(x) gsub("[^a-z_ ]", "", x))
# as.character(docs[[681]]) # inspect 1%

##strip extra whitespace
docs <- tm_map(docs, stripWhitespace)

##strip words starting with whitespace
docs <- tm_map(docs, FUN=function(x) gsub("^ ", "", x))
# as.character(docs[[681]]) # inspect the space


docs <- tm_map(docs, PlainTextDocument) 
tdm <-DocumentTermMatrix(docs,control = list(weighting = function(x) weightTfIdf(x, normalize = FALSE)))

##remove sparse terms
tdms <- removeSparseTerms(tdm, 0.996)

dataset <- as.data.frame(as.matrix(tdms))
n_words <- dim(tdms)[2]

##free memory
rm(tdms,tdm,docs)

print("### END OF PROCESSING ###")
### END OF PROCESSING ###

### BUILD TRAIN AND TEST SET ###
print("### BUILD TRAIN AND TEST SET ###")
dataset$cuisine <- as.factor(c(train_raw$cuisine, rep("italian", nrow(test_raw))))

train  <- dataset[1:nrow(train_raw), ]
test <- dataset[-(1:nrow(train_raw)), ]

## free memory
rm(dataset)

#MODEL mxnet deep learning
print("MODEL mxnet ...")
data <- mx.symbol.Variable("data")
fc2 <- mx.symbol.FullyConnected(data, name="fc2", num_hidden=130)
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="sigmoid")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=60)
act3 <- mx.symbol.Activation(fc3, name="relu3", act_type="sigmoid")
fc4 <- mx.symbol.FullyConnected(act3, name="fc4", num_hidden=20)
softmax <- mx.symbol.Softmax(fc4, name="sm")
devices <- mx.cpu()
mx.set.seed(0)
model <- mx.model.FeedForward.create(softmax, X=data.matrix(cbind(train[,1:n_words],train2$xgboost,train2$mxnet)), y=train$cuisine,
                                     ctx=devices, num.round=50, array.batch.size=80,
                                     learning.rate=0.05, momentum=0.8,  eval.metric=mx.metric.accuracy,
                                     initializer=mx.init.uniform(0.07))

#Predict test
pred <- predict(model,data.matrix(test[,1:n_words]))
pred.label = max.col(t(pred))-1

# prepare submission
print("Writing submission file ...")
submission <- cbind(test_raw$id,pred)
colnames(submission) <- c("id", "cuisine")
submission <- as.data.frame(submission)
write.csv(submission, file = 'submissions/mxnetnet_multiclass.csv', row.names=F, quote=F)