library("jiebaR", "jiebaRD")
library("stringr")
library("dplyr")
library("quanteda")
library("quanteda.textstats")
library("quanteda.textmodels")
library("magrittr")
library("readxl")
eudist <- function(x1, x2) 
  sqrt( sum( (as_unit_vec(x1) - as_unit_vec(x2))^2 ) )
cossim <- function(x1, x2)
  sum(x1 * x2) / sqrt( sum(x1^2) * sum(x2^2) )
as_unit_vec <- function(x) x / sqrt(sum(x^2))

getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
# library("reticulate")
# use_python("D:\\Anaconda")
# py_available()

setwd("./data_set")
getwd()

text <- list.files(path = "./", full.names = FALSE)## ÀÉ®×
paragraph_num <- vector("numeric", length = length(text))## number of paragraph
article_len <- vector("numeric", length = length(text))## length of article
sentence_num <- vector("numeric", length = length(text))## number of sentencce

post <- list()
first <- list()
last <- list()
for(i in seq_along(text)){
  post[[i]] <- readLines(text[i], encoding = "UTF-8", warn = FALSE)
  paragraph_num[i] <- length(post[[i]])
  a <- post[[i]][1]
  first[i] <- post[[i]][1]
  last[i] <- post[[i]][length(post[[i]])]
  for(j in 2 : length(post[[i]])){
    a <- paste0(a, post[[i]][j])
  }
  post[[i]] <- a
}

first <- unlist(first)
last <- unlist(last)
post <- unlist(post)## turn to vector
for(i in seq_along(post)){ 
  sentence_num[i] <- length(unlist(strsplit(post[i], split = "[。！？]", fixed = FALSE)))
}

# print(last)
seg <- worker()
content <- vector("character", length(post))
first_break <- vector("character", length(post))
last_break <- vector("character", length(post))
for(i in seq_along(post)){
  segged <- segment(post[i], seg)
  segged_F <- segment(first[i], seg)
  segged_L <- segment(last[i], seg)
  article_len[i] <- nchar(paste(segged, collapse = ""))
  content[i] <- paste(segged, collapse = "\u3000")
  first_break[i] <- paste(segged_F, collapse = "\u3000")
  last_break[i] <- paste(segged_L, collapse = "\u3000")
}
similar <- vector("numeric", length(post))
for(i in seq_along(post)){
  a <- c(first_break[i], last_break[i])
  # print(a)
  quanteda_corpus_a <- corpus(a) %>%
    tokenizers::tokenize_regex(pattern =  "\u3000") %>%
    tokens()
  q_dfm <- dfm(quanteda_corpus_a) %>%
    dfm_remove(pattern =  readLines("../source/stopwords.txt"), valuetype = "fixed") %>%
    dfm_select(pattern = "[\u4E00-\u9FFF]", valuetype = "regex") %>%
    dfm_tfidf()
  similar[i] <- cossim(q_dfm[1, ], q_dfm[2, ])
}
print(similar)

df_break <- tibble::tibble(
  id = text,
  content = content
)

df_unbreak <- tibble::tibble(
  id = text,
  par_num = paragraph_num,
  sentence_num = sentence_num,
  article_len = article_len,
  content = post
)


