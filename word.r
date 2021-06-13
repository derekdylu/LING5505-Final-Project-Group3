library("jiebaR", "jiebaRD")
library("stringr")
library("dplyr")
library("quanteda")
library("quanteda.textstats")
library("quanteda.textmodels")
library("magrittr")
library("readxl")

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
for(i in seq_along(text)){
  post[[i]] <- readLines(text[i], encoding = "UTF-8", warn = FALSE)
  paragraph_num[i] <- length(post[[i]])
  a <- post[[i]][1]
  for(j in 2 : length(post[[i]])){
    a <- paste0(a, post[[i]][j])
  }
  post[[i]] <- a
  
}

post <- unlist(post)## turn to vector
for(i in seq_along(post)){ 
  sentence_num[i] <- length(unlist(strsplit(post[i], split = "[。！？]", fixed = FALSE)))
}


seg <- worker()
content <- vector("character", length(post))
for(i in seq_along(post)){
  segged <- segment(post[i], seg)
  article_len[i] <- nchar(paste(segged, collapse = ""))
  content[i] <- paste(segged, collapse = "\u3000")
}

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

# df_break
# df_unbreak
# 
# df <- read_excel("../frequency_30_word.xlsx")
# test <- df[2]
# test
quanteda_corpus <- corpus(df_break, 
                          docid_field = "id", 
                          text_field = "content") %>%
  tokenizers::tokenize_regex(pattern =  "\u3000") %>%
  tokens()

q_dfm <- dfm(quanteda_corpus) %>% 
  dfm_remove(pattern =  readLines("../stopwords.txt"), valuetype = "fixed") %>%
  dfm_select(pattern = "[\u4E00-\u9FFF]", valuetype = "regex") %>%
  dfm_trim(min_termfreq = 5) %>%
  dfm_tfidf()
q_dfm

doc_sim <- textstat_simil(q_dfm, method = "cosine") %>% as.matrix()
sort(doc_sim["test.TXT", ], decreasing = T)[1:8]

