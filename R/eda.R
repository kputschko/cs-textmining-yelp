
# Sentiment Analysis ------------------------------------------------------

pacman::p_load(tidyverse, tidytext, listviewer, jsonlite, janitor, lubridate,
               ggridges, glmnet, topicmodels)


# Import Raw --------------------------------------------------------------

# path_business <- "data/yelp_business.rds"
# path_review   <- "C:/Users/exp01754/Downloads/yelp-csv/yelp_academic_dataset_review.rds"
#
# yelp_business <- path_business %>% read_rds() %>% clean_names()
# yelp_review   <- path_review %>% read_rds() %>% clean_names()
#
#
# yelp_review %>% glimpse()
# yelp_business %>% glimpse()
# yelp_business %>% count(city)
# yelp_business %>% count(categories) %>% View()
# yelp_business %>% count(categories) %>% filter(str_to_lower(categories) %>% str_detect("travel")) %>% View()
# yelp_business %>% count(categories) %>% filter(str_to_lower(categories) %>% str_detect("airline")) %>% View()

# yelp_business_review <-
#   yelp_business %>%
#   filter(str_to_lower(categories) %>% str_detect("airline")) %>%
#   select(business_id, name, review_count, stars) %>%
#   arrange(-review_count, stars) %>%
#   filter(business_id != "#NAME?")
#
# yelp_airlines <-
#   yelp_review %>%
#   inner_join(yelp_business_review, by = "business_id") %>%
#   select(date, business_name = name, review_id, stars = stars.x, text) %>%
#   mutate(business_name = case_when(
#     business_name == "Delta Air Lines" ~ "Delta Airlines",
#     business_name == "Jet Blue" ~ "JetBlue",
#     business_name == "United Airlines / TED" ~ "United Airlines",
#     TRUE ~ business_name
#   ))
#
# yelp_airlines %>% write_rds("data/yelp_airline.rds")


# Airlines ----------------------------------------------------------------

yelp_airlines <- read_rds("data/yelp_airline.rds")

# yelp_airlines %>%
#   group_by(business_id) %>%
#   summarise(stars = mean(stars)) %>%
#   arrange(-stars) %>%
#   right_join(yelp_business_review, by = "business_id")

# yelp_airlines %>% count(business_id, sort = TRUE)
# yelp_airlines %>% glimpse()


# Summary Stats -----------------------------------------------------------

yelp_summary <-
  yelp_airlines %>%
  summarise(date_min = min(date),
            date_max = max(date),
            review_n = n_distinct(review_id),
            business_n = n_distinct(business_name))


# Simple Word Freq Summary ------------------------------------------------
# Which words are used in more 10 reviews for more than 5 businesses

data_word_summary <-
  yelp_airlines %>%
  unnest_tokens(word, text) %>%
  count(business_name, review_id, stars, word, sort = TRUE) %>%
  group_by(word) %>%
  summarise(businesses = n_distinct(business_name),
            reviews = n(),
            uses = sum(n),
            average_stars = mean(stars)) %>%
  arrange(-uses) %>%
  print()


data_words_impact <-
  data_word_summary %>%
  filter(businesses > 1, reviews >= 8) %>%
  arrange(desc(average_stars)) %>%
  print()


# data_words_impact %>%
#   mutate(round = scales::number(average_stars, accuracy = 0.50)) %>%
#   group_by(round) %>%
#   top_n(5, average_stars) %>%
#   ggplot() +
#   aes(x = average_stars, y = average_stars, label = word) +
#   geom_label(position = position_jitter(0.4, 0.4, seed = 42),
#              size = 3,
#              fill = "lightgray") +
#   theme_minimal() +
#   labs(x = NULL, y = NULL, title = "Yelp Airlines", subtitle = "Average Rating by Word", caption = "Words chosen were used in more than 10 reviews,\nand for at least 5 different businesses")

temp <-
  data_word_summary %>%
  anti_join(stop_words) %>%
  # filter(businesses > 1, reviews > 5) %>%
  mutate(round = scales::number(average_stars, accuracy = 0.50)) %>%
  group_by(round) %>%
  top_n(5, average_stars) %>%
  top_n(5, businesses) %>%
  top_n(5, reviews) %>%
  top_n(5, uses) %>%
  arrange(average_stars)

View(temp)

temp %>%
  ggplot() +
  aes(x = average_stars, y = uses, label = word) +
  geom_label(position = position_jitter(width = 0.4, height = 0.8, seed = 42),
             size = 3,
             fill = "lightgray",
             alpha = 0.50) +
  theme_minimal() +
  scale_y_log10() +
  labs(x = "Average Stars",
       y = "log(Frequency)",
       title = "Yelp Review Word Frequency",
       caption = "Words displayed were")


# Ratings -----------------------------------------------------------------

yelp_airlines %>%
  count(stars) %>%
  ggplot() +
  aes(x = stars, y = n, label = n, fill = stars) +
  geom_col() +
  geom_text(nudge_y = 50) +
  scale_fill_viridis_c(guide = FALSE) +
  theme_minimal() +
  scale_y_continuous(breaks = NULL, labels = NULL) +
  labs(x = NULL, y = NULL, title = "Yelp Airlines", subtitle = "A majority of reviews are 1*")



# Ratings Over Time -------------------------------------------------------
# Daily is too messy
# Monthly is preferable

eda_monthly <-
  yelp_airlines %>%
  mutate(month = lubridate::floor_date(date, unit = "week")) %>%
  group_by(month) %>%
  summarise(businesses = n_distinct(business_name),
            reviews = n_distinct(review_id),
            avg_stars = mean(stars))

eda_monthly %>%
  ggplot() +
  aes(x = month, y = avg_stars, size = reviews, color = avg_stars) +
  geom_smooth(se = FALSE, span = .6, color = "black") +
  geom_vline(xintercept = as_date("2013-01-01"), color = "red", linetype = "dashed") +
  geom_point(alpha = 0.75) +
  scale_color_viridis_c() +
  theme_minimal() +
  labs(x = NULL, y = NULL,
       color = "Star Rating",
       title = "Yelp Airlines",
       subtitle = "Average monthly star rating has been decreasing since 2013",
       caption = "Size: Number of Monthly Reviews") +
  guides(size = "none") +
  scale_y_continuous(breaks = 1:5, minor_breaks = NULL)


# Sentiment of Reviews ----------------------------------------------------
# What do consumers think of us on the internet?

# Hu & Liu Opinion Sentiment
# # This is very similar to bing sentiment

# list(positive = read_lines("data/positive-words.txt", skip = 35),
#      negative = read_lines("data/negative-words.txt", skip = 35)) %>%
#   enframe(name = "sentiment", value = "word") %>%
#   unnest() %>%
#   write_rds("data/sentiment_huliu.rds")


# sen_score <- get_sentiments(lexicon = "afinn")
sen_label <- get_sentiments(lexicon = "bing")
# sen_emote <- get_sentiments(lexicon = "nrc") %>% rename(emotion = sentiment)
# sen_huliu <- read_rds("data/sentiment_huliu.rds")

lexicon <- sen_label

data_sentiment_review <-
  yelp_airlines %>%
  select(date, review_id, text) %>%
  unnest_tokens(word, text) %>%
  left_join(lexicon) %>%
  group_by(date, review_id) %>%
  count(sentiment) %>%
  mutate(percent_review = n / sum(n)) %>%
  # filter(review_id == "WP6pvnTaIp5zO7Tv7Sr85Q") %>%
  # filter(date == "2008-04-22") %>%
  print()


data_sentiment_daily <-
  data_sentiment_review %>%
  group_by(date) %>%
  mutate(percent_review_daily  = n / sum(n)) %>%
  group_by(date, sentiment) %>%
  summarise(percent_daily = sum(percent_review_daily)) %>%
  mutate(percent_daily = if_else(sentiment == "negative", -percent_daily, percent_daily)) %>%
  filter(!is.na(sentiment)) %>%
  print()


plot_sentiment <-
  data_sentiment_daily %>%
  ggplot() +
  aes(x = date, y = percent_daily, fill = str_to_title(sentiment)) +
  geom_col() +
  geom_hline(yintercept = 0, color = "black") +
  theme_minimal() +
  labs(x = NULL, y = NULL,
       title = "Yelp Airlines",
       subtitle = "Negative sentiment becomes more pronounced after 2012",
       fill = "Sentiment") +
  scale_y_continuous(limits = c(-0.50, 0.50))


# Is This Sentiment Trustworthy?
plot_capture <-
  data_sentiment_review %>%
  left_join(yelp_airlines %>% select(review_id, stars)) %>%
  filter(!is.na(sentiment)) %>%
  group_by(date, review_id) %>%
  filter(percent_review == max(percent_review)) %>%
  ggplot() +
  aes(x = stars, fill = str_to_title(sentiment)) +
  geom_density(alpha = 0.50) +
  scale_y_continuous(breaks = NULL) +
  theme_minimal() +
  labs(y = NULL,
       x = "Review Stars",
       fill = "Sentiment",
       title = "Yelp Airlines",
       subtitle = "Negative sentiment appears to capture lower star reviews,\nhowever positive sentiment is spread across all stars uniformly.")


# plot_capture_2 <-
#   data_sentiment_review %>%
#   left_join(yelp_airlines %>% select(review_id, stars)) %>%
#   filter(!is.na(sentiment)) %>%
#   mutate(percent_review = if_else(sentiment == "negative", -percent_review, percent_review)) %>%
#   group_by(date, review_id, stars) %>%
#   summarise(review_score = sum(percent_review),
#             sentiment = if_else(review_score >= 0, "Positive", "Negative")) %>%
#   ggplot() +
#   aes(x = review_score, y = factor(stars)) +
#   stat_density_ridges(geom = "density_ridges_gradient", calc_ecdf = TRUE, quantiles = 0) +
#   geom_vline(xintercept = 0, linetype = "dashed") +
#   scale_fill_viridis_c()



# Prediction of Rating via Text -------------------------------------------
# Create our own sentiment scores
# Predict sentiment and rating of tweets from Twitter

data_dtm <-
  yelp_airlines %>%
  select(review_id, text) %>%
  unnest_tokens(word, text) %>%
  count(review_id, word) %>%
  cast_dtm(review_id, word, n)

# data_dtm %>% dim()
# data_dtm %>% colnames() %>% last()
# data_dtm %>% tidy()
# data_dtm %>% class()


data_matrix <- data_dtm %>% as.matrix()
response <- yelp_airlines %>% arrange(review_id) %>% pull(stars)

# GLMNET
model_lasso <- glmnet(x = data_matrix, y = response, family = "gaussian", alpha = 1)
model_lasso_cv <- cv.glmnet(x = data_matrix, y = response, family = "gaussian", alpha = 1, type.measure = "mse")
model_lasso_cv %>% plot()

model_predict <- predict(model_lasso, data_matrix, s = model_lasso_cv$lambda.min)
model_predict_coef <- predict(model_lasso, data_matrix, s = model_lasso_cv$lambda.min, type = "coefficient")


model_predict_review <-
  model_predict %>%
  as.data.frame() %>%
  rownames_to_column("review_id") %>%
  as_tibble() %>%
  rename(score = `1`)


model_predict_word <-
  model_predict_coef %>%
  as.matrix() %>%
  as.data.frame() %>%
  rownames_to_column("word") %>%
  as_tibble() %>%
  rename(score = `1`) %>%
  mutate(sentiment = case_when(
    score < 0 ~ "Negative",
    score > 0 ~ "Positive",
    TRUE ~ "Neutral"
  )) %>%
  filter(word != "(Intercept)")



# A little better, but a lot of overlap
model_predict_review %>%
  left_join(yelp_airlines %>% select(review_id, stars)) %>%
  mutate(residual = score - stars) %>%
  arrange(residual) %>%
  ggplot() +
  aes(x = score, y = factor(stars), fill = factor(stars)) +
  ggridges::geom_density_ridges() +
  scale_fill_viridis_d(direction = -1)


# Our custom sentiment scores work better!
data_sentiment_review_custom <-
  yelp_airlines %>%
  select(date, review_id, stars, text) %>%
  unnest_tokens(word, text) %>%
  left_join(model_predict_word) %>%
  group_by(date, review_id, stars) %>%
  count(sentiment) %>%
  mutate(percent_review = n / sum(n)) %>%
  print()


data_sentiment_review_custom %>%
  filter(sentiment != "Neutral") %>%
  group_by(date, review_id, stars) %>%
  filter(percent_review == max(percent_review)) %>%
  ggplot() +
  aes(x = stars, fill = sentiment) +
  geom_density(alpha = 0.50) +
  scale_y_continuous(breaks = NULL) +
  theme_minimal() +
  labs(y = NULL,
       x = "Review Stars",
       fill = "Sentiment",
       title = "Yelp Airlines",
       subtitle = "Our custom sentiment scores do a better job distinguishing\nbetween positive and negative reviews.")



# Most Influential Words --------------------------------------------------
# Then, let's see how the words influence the sentiment by human intuition
# Let's look at test cases, how words influence sentiment scores

# Redo glmnet with tf-idf weighting, because "averaged" is most negative word but shows up once!
# But this redo looks to be much less accurate!

model_predict_word %>%
  filter(sentiment != "Neutral") %>%
  group_by(sentiment) %>%
  top_n(10, wt = abs(score)) %>%
  arrange(score) %>%
  ggplot() +
  aes(x = as_factor(word), y = score, fill = sentiment) +
  geom_col() +
  coord_flip() +
  labs(x = NULL, y = NULL, fill = NULL,
       title = "Yelp Airlines",
       subtitle = "Top 20 most influential words") +
  theme_minimal()

# yelp_airlines %>%
#   filter(str_detect(text, "average|nickel")) %>%
#   arrange(stars) %>%
#   View()

# yelp_airlines %>%
#   filter(str_detect(text, "knock|mornings")) %>%
#   arrange(stars) %>%
#   View()

pacman::p_load(ggrepel)

inner_join(x = model_predict_word %>% filter(sentiment != "Neutral"),
           y = yelp_airlines %>% unnest_tokens(word, text) %>% count(word, sort = TRUE),
           by = "word") %>%
  filter(n < 1000) %>%
  top_n(100, n) %>%
  ggplot() +
  aes(x = score, y = n, color = score, label = word, size = n) +
  geom_text_repel(segment.alpha = 0) +
  scale_y_log10() +
  scale_color_viridis_c(option = "A", end = 0.75, guide = "none") +
  scale_size_continuous(range = c(2, 6),
                        guide = FALSE) +
  theme_minimal() +
  labs(x = "Word Rating Score", y = "Word Frequency")







# TF-IDF ------------------------------------------------------------------
# Terms that are important within a document, and distingush the doc from other
# documents in the collection

# THIS IS OK, NOT GREAT

yelp_airlines %>%
  unnest_tokens(word, text) %>%
  group_by(stars, word) %>%
  summarise(n_business = n_distinct(business_name),
            n_reviews  = n_distinct(review_id),
            n_word     = length(word)) %>%
  arrange(-n_word) %>%
  filter(n_business > 1,
         n_word > 5) %>%
  bind_tf_idf(word, stars, n_word) %>%
  arrange(-tf_idf) %>%
  group_by(stars) %>%
  top_n(5, tf_idf) %>%

  ggplot() +
  aes(x = reorder(word, n_word), y = n_word, alpha = tf_idf) +
  geom_col(color = "black") +
  coord_flip() +
  facet_wrap(vars(stars), scales = "free") +
  scale_alpha(range = c(.2, 1)) +
  theme_minimal()



# Word Usage & Rating -----------------------------------------------------
# If yelp allows us to scrub reviews with these 50 words, which 50 should they be?
# What text is contained in some of the most negative reviews

# FILTER STOP WORDS HERE
## "never", "worst",
## "can't", "complain",
## "presto", "former"


yelp_airlines %>%
  select(date, review_id, stars, text) %>%
  unnest_tokens(word, text, "ngrams", n = 2) %>%
  filter(str_detect(word, "can't|never|complain|worst|presto|former")) %>%
  separate(word, c("word_1", "word_2"), sep = " ") %>%
  count(stars, word_1, word_2, sort = TRUE) %>%
  anti_join(get_stopwords(source = "snowball"), by = c("word_2" = "word")) %>%
  filter(word_1 %in% c("never", "worst")) %>%
  top_n(10, n) %>%
  ggplot() +
  aes(x = reorder(word_2, n), y = n) +
  geom_col() +
  coord_flip() +
  scale_fill_viridis_c() +
  facet_wrap(facets = "word_1", scales = "free")



# LDA Groups --------------------------------------------------------------
# do the reviews group together in a pattern similar to star rank?
##  OH! Which words can be used to distinguish each star rating!!!

#  THIS IS NOT INTERESTING

# snowball > no-removal

# yelp_lda <-
#   yelp_airlines %>%
#   unnest_tokens(word, text) %>%
#   count(review_id, word, sort = TRUE) %>%
#   anti_join(get_stopwords(source = "snowball")) %>%
#   cast_dtm(review_id, word, n) %>%
#   LDA(k = 2, control = list(seed = 42))
#
#
# yelp_lda %>%
#   tidy(matrix = "gamma") %>%
#   left_join(yelp_airlines %>% select(document = review_id, stars)) %>%
#   group_by(document) %>%
#   filter(gamma == max(gamma)) %>%
#   ggplot() +
#   aes(x = stars, y = factor(topic)) +
#   geom_density_ridges()
#
#
#
# yelp_lda %>%
#   tidy(matrix = "beta") %>%
#   anti_join(stop_words, by = c("term" = "word")) %>%
#   group_by(topic) %>%
#   top_n(10, beta) %>%
#   ungroup() %>%
#   arrange(topic, -beta) %>%
#   View(title = "top words")



# Co-Occurance ------------------------------------------------------------

pacman::p_load(igraph, ggraph)

a <- grid::arrow(type = "closed", length = unit(.15, "inches"))

yelp_airlines %>%
  select(date, review_id, stars, text) %>%
  unnest_tokens(word, text, "ngrams", n = 2) %>%
  separate(word, c("word_1", "word_2"), sep = " ") %>%
  count(word_1, word_2, sort = TRUE) %>%
  top_n(100, n) %>%
  graph_from_data_frame() %>%
  ggraph(layout = "fr") +
  geom_edge_link(aes(edge_alpha = n), show.legend = FALSE,
                 arrow = a, end_cap = circle(.07, 'inches')) +
  geom_node_point(color = "lightblue", size = 5) +
  geom_node_text(aes(label = name), vjust = 1, hjust = 1) +
  theme_void()


count_bigrams <- function(dataset) {
  dataset %>%
    unnest_tokens(bigram, text, token = "ngrams", n = 2) %>%
    separate(bigram, c("word1", "word2"), sep = " ") %>%
    filter(!word1 %in% stop_words$word,
           !word2 %in% stop_words$word) %>%
    count(word1, word2, sort = TRUE)
}

visualize_bigrams <- function(bigrams) {
  set.seed(2016)
  a <- grid::arrow(type = "closed", length = unit(.15, "inches"))

  bigrams %>%
    top_n(100, n) %>%
    graph_from_data_frame() %>%
    ggraph(layout = "fr") +
    geom_edge_link(aes(edge_alpha = n), show.legend = FALSE, arrow = a) +
    geom_node_point(color = "lightblue", size = 5) +
    geom_node_text(aes(label = name), vjust = 1, hjust = 1) +
    theme_void()
}


yelp_airlines %>% count_bigrams() %>% visualize_bigrams()


# Sentiment Over Time -----------------------------------------------------

# sen_score <- get_sentiments(lexicon = "afinn")
# sen_label <- get_sentiments(lexicon = "bing")
# sen_emote <- get_sentiments(lexicon = "nrc")
#
# stop_full     <- stop_words %>% distinct(word)
# stop_smart    <- get_stopwords(source = "smart")
# stop_snowball <- get_stopwords(source = "snowball")

# afinn miscategorizes 2013-01-19 review as positive!
# snowball has a more positive skew than smart,
# Keep Stop Words

# data_airline <-
#   yelp_airlines %>%
#   # filter(business_id == "bA-Cj6N9TEMlDlOh2aAnUw") %>%
#   select(business_id, review_id, date, stars, text) %>%
#   arrange(date) %>%
#   rowid_to_column() %>%
#   unnest_tokens(word, text)

# data_score <-
#   data_airline   %>%
#   left_join(sen_score) %>%
#   # anti_join(stop_smart) %>%
#   # anti_join(stop_snowball) %>%
#   # anti_join(stop_full) %>%
#   replace_na(replace = list(score = 0)) %>%
#   group_by(date, review_id, stars) %>%
#   summarise(sum_score = sum(score, na.rm = TRUE),
#             mean_score = mean(score, na.rm = TRUE)) %>%
#   print()


# Is this trustworthy?
# It's ok
# data_score %>%
#   ggplot() +
#   aes(x = factor(stars), y = mean_score) +
#   geom_boxplot()


# Average sentiment score
# Dramatic fluctuations between 2012 and 2014
# data_score %>%
#   ggplot(aes(x = date, y = mean_score)) +
#   geom_smooth() +
#   geom_hline(yintercept = 0, color = "red") +
#   theme_minimal()


#
# data_words$businesses %>% summary()
# data_words$reviews %>% summary()



# Twitter -----------------------------------------------------------------

twt <-
  read_rds("data/twitter_airline.rds") %>%
  mutate(date = tweet_created %>% anytime::anytime()) %>%
  distinct(date, text) %>%
  rownames_to_column("tweet_id")

twt %>% glimpse()

twt_words <- twt %>% unnest_tokens(word, text)

twt_summary <-
  twt %>%
  summarise(date_min = min(date),
            date_max = max(date),
            tweets = length(text),
            char_avg = str_length(text) %>% mean()) %>%
  add_column(word_med = twt_words %>% count(tweet_id) %>% pull(n) %>% median()) %>%
  print()


# Test Yelp Scores
# 9,079 words in Yelp Score
# 15,535 words in Twitter
# 5,389 words in both

twt_words %>% summarise(word_distinct = n_distinct(word))

model_prediction$model_predict_word %>% count(sentiment)
twt_words %>% count(word) %>% inner_join(model_prediction$model_predict_word)

model_prediction$model_predict_word %>%
  inner_join(twt_words %>% distinct(word)) %>%
  ggplot() +
  aes(x = sentiment) +
  geom_bar() +
  scale_y_log10()

intercept <- model_prediction$model_predict_word %>% filter(word == "(Intercept)") %>% pull(score)

# Testing prediction function!
# yelp_words %>%
#   left_join(model_predict_word) %>%
#   group_by(review_id) %>%
#   summarise(stars = unique(stars),
#             score_raw = sum(score),
#             score = score_raw + !!intercept)
# model_predict_review

lala %>% left_join(model_prediction$model_predict_word) %>% filter(score != 0) %>% distinct(word)

group_by(tweet_id) %>%
  summarise(score_raw = sum(score, na.rm = TRUE),
            score = score_raw + intercept,
            sentiment = unique(airline_sentiment))


  ggplot() +
  aes(x = score, y = factor(sentiment)) +
  geom_density_ridges()



twt %>% count(tweet_id) %>% filter(n > 1)
twt %>% add_count(tweet_id) %>% filter(n > 1) %>% arrange(tweet_id) %>% distinct(text)
