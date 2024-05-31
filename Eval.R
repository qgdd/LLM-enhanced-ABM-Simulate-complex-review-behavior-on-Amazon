df = read.csv('eval.csv')

# contradiction rate
table(df$hallucination)
sum(df$hallucination == "contradiction")/nrow(df)
# exhibit hallucinated outputs
hal_df = df[df$hallucination == "contradiction",]
# average score of confidence
mean(df[df$hallucination == "contradiction",]$score)
mean(df[df$hallucination == "entailment",]$score)
mean(df[df$hallucination == "neutral",]$score)
# hallucination distribution by rating
table(hal_df$rating)
table(hal_df$rating)/table(df$rating)

# sentiment analysis
conf_mat = table(df$review_sentiment, df$summary_sentiment)
# accuracy
sum(diag(conf_mat))/sum(conf_mat)
addmargins(conf_mat)

# semantic similarity
mean(df$semantic_sim)
mean(df[df$hallucination!="contradiction",]$semantic_sim)
mean(df[df$hallucination=="contradiction",]$semantic_sim)
