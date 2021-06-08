library("jiebaR", "jiebaRD")
library("stringr")

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

df_break
df_unbreak