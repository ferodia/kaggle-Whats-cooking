#!/usr/bin/env Rscript
# The below code cleans up the data and save it files to be exploited by models later on

## clean train set by merging words with 1 and 2  as distance, then include the test set as well

### LOAD LIBRARIES ###
print("### LOAD LIBRARIES ###")
library(jsonlite)
library("NLP")
library("tm")
library("SnowballC") 
library("forecast")
library("ttutils")
library("stringdist")

### HELPER FUNCTIONS ###

# check spelling of a word, sp is a data stracture containing all the words in the data files with suggestions
checkspell <- function(x)
{
  return(x %in% sp$Original)
}

# compare two words alphabetically
issmaller <- function(x,y)
{return(as.character(x) < as.character(y))}

# various functions returning distance metrics between two words
soundex <- function(x1,x2){return(stringdist(x1,x2,method='soundex'))}
osa <- function(x1,x2){return(stringdist(x1,x2,method='osa'))}
dl <- function(x1,x2){return(stringdist(x1,x2,method='dl'))}
lcs <- function(x1,x2){return(stringdist(x1,x2,method='lcs'))}
hamming <- function(x1,x2){return(stringdist(x1,x2,method='hamming'))}
qgram <- function(x1,x2){return(stringdist(x1,x2,method='qgram'))}
cosine <- function(x1,x2){return(stringdist(x1,x2,method='cosine'))}
jw <- function(x1,x2){return(stringdist(x1,x2,method='jw'))}
jaccard <- function(x1,x2){return(stringdist(x1,x2,method='jaccard'))}

##algorithms for identifying misspelled words and merging similar words

# 1- first check if two words are phonetically similar ie : 0 which means similar and we need to replace one of them with the other
# 2- if one of the two words is misspelled, take the correct one
# 3- if two words misspelled or not english, take the smallest one alphabetically
# 4- if two words are correct, check if difference is only a space and take the one with space
# 5- if two words are correct, if difference is not a space,and there is more than one word then take the smallest
# 6- for non printable characters, filé to be replaced by file power, and kahlùa by kahlua
# 7- at the end we should have a mtrix with two columns with old word and new word 
replace_misspelled <- function(word,suggestion,phonetic,misspelled_word,misspelled_closest)
{
  if(word == "filé")
  {
    return("file powder")
  }
  else if (word == "kahlúa")
  {
    return("kahlua")
  }
  if(!phonetic) #words are similar
  {
    if(misspelled_word & !misspelled_closest) # word is misspelled
    {
      return(suggestion)
    }
    else if(!misspelled_word & misspelled_closest) # suggeston is misspelled
    {
      return(word)
    }
    else if (misspelled_word & misspelled_closest) # both words misspelled or foreign 
    {
      return(min(word,suggestion))
    }
    else if (!misspelled_word & !misspelled_closest)
    {
      if(nchar(word) != nchar(suggestion))
      {
        # return the longer
        return(max(word,suggestion))
      }
      else # there is one letter change
      {
        if( grepl(" ",word) & grepl(" ",suggestion)) # there are more than one term
        {
          return(min(word,suggestion))
        }
        else
          return(word)
      }
    }
  }
  else{ return(word)}
}

#replace words in a recursive way
replacewords <- function(col1, col2, iterations, file)
{
  if(iterations > 0)
  {
    file <- gsub( col1[iterations], col2[iterations], file)
    replacewords(col1, col2, iterations-1, file)
  }
  return(file) 
}

# replace a word with suggestion based on jaccard and jw distances and fixed thresholds
# thresholds are defined upon generic observation
replace_jaccard_jw <- function(word, closest, jaccard, jw){
  if(jw <= 0.07843137 || (jaccard<=0.08 & jw <= 0.2))
  {
    return(min(word,closest))
  }
  else
    return(word)
}

# replace similar word based on a fixed threshold
replace_similarity <- function(word, closest, similarity){
  if(similarity <= 0.004579812)
  {
    return(min(word,closest))
  }
  else
    return(word)
}

### LOAD DATA ###
print("### LOAD DATA ###")

train_raw  <- fromJSON("data/train.json", flatten = TRUE)
test_raw  <- fromJSON("data/test.json", flatten = TRUE)

# collect data from trainint and test sets
list <- merge(train_raw$ingredients,test_raw$ingredients)
ingredients <- as.factor(unlist(list))

# remove duplicated ingredients
ingredients <- unique(ingredients)

# compute Levenshtein distance between ingredients
# Levenshtein distance: minimum edits to change from oneword to the other
print("Computing Levenshtein distance...")
dist <-  stringdistmatrix(ingredients,ingredients, method = "lv")

# collect words whose distances are either 1, 2 or 3 sine we are looking for very similar words
index_1 <- which(dist==1, arr.ind = T)
index_2 <- which(dist==2, arr.ind = T)
index_3 <- which(dist==3, arr.ind = T)

row_1 <- index_1[,1]
col_1 <- index_1[,2]
row_2 <- index_2[,1]
col_2 <- index_2[,2]
row_3 <- index_3[,1]
col_3 <- index_3[,2]

# build table1 with LV DIST 1 with a word and his most similar word (closest)
table1 <- data.frame(ingredients[row_1])
table1$closest <- ingredients[col_1]
names(table1) <- c("word","closest")

# build table2 with LV DIST 2 with a word and his most similar word (closest)
table2 <- data.frame(ingredients[row_2])
table2$closest <- ingredients[col_2]
names(table2) <- c("word","closest")

# build table3 with LV DIST 3 with a word and his most similar word (closest)
table3 <- data.frame(ingredients[row_3])
table3$closest <- ingredients[col_3]
names(table3) <- c("word","closest")

# use aspell to identify misspelled words
print("Misspellings check ...")
sp <- aspell(files =c("data/train.json", "data/test.json" ) )

## TABLE 1 ##
print("Building table 1 ...")
# check if the words of the table 1 are misspelled 
table1$misspelled_word <- mapply(checkspell,table1$word)
table1$misspelled_closest <- mapply(checkspell,table1$closest)

# compute distance metric soundex to identify phonetic similarity
table1$phonetic <- mapply(soundex, table1$word, table1$closest)
# ! a warning will be displayed because of the two terms "kahlúa" and "filé", which are handled separately


## TABLE 2 ##
print("Building table 2 ...")
# compute jaccard and jw distances
table2$jaccard <- mapply(jaccard, table2$word, table2$closest)
table2$jw <- mapply(jw, table2$word, table2$closest)

## TABLE 3 ##
print("Building table 3 ...")
# compute jw distance, length and misspellings
# compute also relative similary because it depends on the length, the shorter two words are, the less likely they would be similar
table3$jw <-  mapply(jw, table3$word, table3$closest)
table3$length <-  mapply(nchar, as.character(table3$word))
table3$misspelled_word <- mapply(checkspell,table3$word)
table3$misspelled_closest <- mapply(checkspell,table3$closest)
table3$relativesimilarity <- mapply(FUN = function(x,y) return(x/y), table3$jw, table3$length)

# make decision on the new word (either word or closest) using the algorithms of the helper function
table3$newWord <- mapply(replace_similarity, as.character(table3$word),as.character(table3$closest),table3$relativesimilarity)
table2$newWord <- mapply(replace_jaccard_jw, as.character(table2$word),as.character(table2$closest),table2$jaccard,table2$jw)
table1$newWord <- mapply(replace_misspelled, as.character(table1$word),as.character(table1$closest),table1$phonetic,table1$misspelled_word, table1$misspelled_closest)

## after building the table we only concider the cases where we need to make some replacement
## we eliminate the rows that have word = newoWord
replaceTable3 <- subset.data.frame(table3,subset=table3$word != table3$newWord)
replaceTable2 <- subset.data.frame(table2,subset=table2$word != table2$newWord)
replaceTable1 <- subset.data.frame(table1,subset=table1$word != table1$newWord)

### BUILDING CLEAN FILES ####
print("### BUILDING CLEAN FILES ####")
  
## now replace words in the json file
train_file <- readLines("data/train.json")
test_file <- readLines("data/test.json")

# replace the words using the replace tables for train and test sets
print("Replacing duplicated words ...")
train_newfile3 <- replacewords(as.character(replaceTable3$word),as.character(replaceTable3$newWord),dim(replaceTable3)[1],train_file)
test_newfile3 <- replacewords(as.character(replaceTable3$word),as.character(replaceTable3$newWord),dim(replaceTable3)[1],test_file)
train_newfile2 <- replacewords(as.character(replaceTable2$word),as.character(replaceTable2$newWord),dim(replaceTable2)[1],train_newfile3)
test_newfile2 <- replacewords(as.character(replaceTable2$word),as.character(replaceTable2$newWord),dim(replaceTable2)[1],test_newfile3)
train_newfile <- replacewords(as.character(replaceTable1$word),as.character(replaceTable1$newWord),dim(replaceTable1)[1],train_newfile2)
test_newfile <- replacewords(as.character(replaceTable1$word),as.character(replaceTable1$newWord),dim(replaceTable1)[1],test_newfile2)

# create the new files
print("Creating new files  trainclean.json and testclean.json...")
writeLines(train_newfile,"data/trainclean.json")
writeLines(test_newfile,"data/testclean.json")



