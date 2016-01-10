#!/usr/bin/env Rscript

### LOAD LIBRARIES ###
print("Loading libraries ...")
library(jsonlite)
library("NLP")
library("tm")
library("SnowballC") 
library("glmnet")
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
tdms <- removeSparseTerms(tdm, 0.995)

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

### MODEL glmnet ###
print("Model glmnet is running ...")
model <- glmnet(as.matrix(train[,1:n_words]),train$cuisine, family="multinomial")
pred_train <- predict(model,as.matrix(train[,1:n_words]), s=0.01,type="class")
train_accuracy <- 1 - mean(pred_train!=train$cuisine) # accuracy =1 - error
train_accuracy

##predict test set
pred_test <- predict(model,as.matrix(test[,1:n_words]), s=0.01,type="class")

##prepare submission
print("Writing submission file ...")
submission <- cbind(test_raw$id,pred_test)
colnames(submission) <- c("id", "cuisine")
submission <- as.data.frame(submission)
write.csv(submission, file = 'submissions/glmnet_multiclass.csv', row.names=F, quote=F)