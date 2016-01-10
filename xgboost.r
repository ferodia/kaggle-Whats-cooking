#!/usr/bin/env Rscript
### LOAD LIBRARY ###

library('xgboost')
library(jsonlite)
library("NLP")
library("tm")
library("SnowballC") 
library("glmnet")
library("forecast")
library("mxnet")


### LOAD DATA ###
train_raw  <- fromJSON("data/trainclean.json", flatten = TRUE)
test_raw  <- fromJSON("data/testclean.json", flatten = TRUE)


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
tdms <- removeSparseTerms(tdm, 0.9999)

dataset <- as.data.frame(as.matrix(tdms))
n_words <- dim(tdms)[2]

##free memory
rm(tdms,tdm,docs)

print("### END OF PROCESSING ###")
### END OF PROCESSING ###

# BUILD TRAIN AND TEST SET
print("### BUILD TRAIN AND TEST SET ###")
train  <- dataset[1:nrow(train_raw), ]
test <- dataset[-(1:nrow(train_raw)), ]
cv <- train[39574:39774,]
train <- train[1:39573,]

##run xgboost
#feature index in xgboost starts from 0
print("XGBOOST training ...")
xgbmat     <- xgb.DMatrix(Matrix(data.matrix(train[,1:n_words])), label=as.numeric(train$cuisine)-1)
xgb        <- xgboost(xgbmat, max.depth = 7, eta = 0.2, nround = 600,col_subsample=0.6, objective = "multi:softmax", num_class = 20)

#CV 
print("Cross validation")
xgbcvpred <- predict(xgb,data.matrix(cv[,1:n_words]))
xgbcvpred.cuisine <- levels(cv$cuisine)[xgbcvpred +1]
mean(xgbcvpred.cuisine==cv$cuisine)

# submission
pred      <- predict(xgb, newdata = data.matrix(test))
pred.cuisine <- levels(train$cuisine)[pred+1]

# prepare submission
submission <- cbind(as.data.frame(test_raw$id),as.data.frame(pred.cuisine))
colnames(submission) <- c("id", "cuisine")
submission <- as.data.frame(submission)
write.csv(submission, file = 'submissions/xgboost_cooking.csv', row.names=F, quote=F)