library(jiebaR)
library(tidytext)
library(dplyr)
library(ggplot2)
library(wordcloud2)

setwd("~/Documents/NTU/2_Sophomore/2-2/R/Project/LING5505-Final-Project-Group3")
seg = worker(stop_word = "stop_words.txt")

setwd("~/Documents/NTU/2_Sophomore/2-2/R/Project/LING5505-Final-Project-Group3/data_set")
file_names = list.files()

all_segged = ""

for(i in seq_along(file_names)){
  doc = readLines(file_names[i], warn = FALSE)
  segged = segment(doc, seg)
  all_segged = c(all_segged, segged)
}


all_segged_ = sapply(all_segged, function(x) paste(x, collapse = "\u3000"))

df = data.frame(all_segged_)

tidy_df = df %>% unnest_tokens(output = "words", input = "all_segged_", token = "regex", pattern = "\u3000")
freq_df = tidy_df %>% group_by(words) %>% summarise(n = n()) %>% arrange(desc(n))

print(freq_df)

cloud = wordcloud2(freq_df,backgroundColor='white',size = 1)
print(cloud)