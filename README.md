# Reddit Engagement Analytics Dashboard 

This project is a Streamlit web application designed to analyze Reddit post data. It provides insights into post engagement, predicts virality and potential controversy, explores topics using LDA, identifies similar posts, and presents key statistics through an interactive dashboard.

##  Features

The dashboard is divided into several key sections accessible via the sidebar:

1.  **Virality & Controversy Predictor:**
    *   **Input:** Enter a potential Reddit post title.
    *   **Virality Prediction:** Predicts the likelihood (%) of the post going viral using a Logistic Regression model trained on features like title text (TF-IDF), average sentiment, average comment count, and average subreddit subscribers. Displays the result using an intuitive gauge.
    *   **Controversy Prediction:** Predicts the likelihood (%) of the post being "controversial" (defined by a high comment-to-score ratio and minimum comment count) using a RandomForestClassifier model trained on features like title text (TF-IDF), sentiment, subreddit subscribers, and dominant topic. Displays the probability score and a progress bar.

2.  **Similar Posts:**
    *   **Input:** Enter a Reddit post title.
    *   **Output:** Finds and displays the top 5 most textually similar posts from the dataset using TF-IDF and Cosine Similarity.
    *   **Details:** Shows the title, similarity score, actual score, subreddit, and ID of the similar posts.

3.  **Topic Trend Predictor:**
    *   **Title Topic Analysis:**
        *   **Input:** Enter a Reddit post title.
        *   **Topic Identification:** Identifies the dominant topic using a pre-trained Latent Dirichlet Allocation (LDA) model.
        *   **Trend Status:** Predicts if the identified topic is currently trending (Emerging or Stable/Declining) based on recent post volume using Exponential Smoothing.
        *   **Best Posting Day:** Suggests the historically best day(s) of the week to post about that specific topic based on past activity patterns.
    *   **Overall Trends:**
        *   Displays weekly trend plots (actual post counts + forecast) for each major topic identified by the LDA model, allowing users to visually track topic popularity over time.

4.  **Key Stats & EDA:**
    *   **Top Metrics:** Shows overall dataset statistics: Total Posts Analyzed, Average Score, and Average Comments.
    *   **Sentiment Analysis:**
        *   Displays the top 5 most positive and top 5 most negative post titles based on NLTK VADER sentiment analysis.
        *   Includes plots (within expanders) showing:
            *   Distribution of sentiment labels (positive, neutral, negative).
            *   Average sentiment score over time (monthly).
            *   Relationship between sentiment and post score (boxplot with log scale).
    *   **Post Distributions:** Provides various visualizations (within expanders) using Matplotlib and Seaborn:
        *   Top 15 Subreddits by post count.
        *   Distribution of posts per Year.
        *   Distribution of posts per Month.
        *   Distribution of posts per Day of the Week.
        *   Distribution of posts per Day of the Month.
    *   **LDA Topic Word Clouds:** Displays word clouds (within an expander) for each topic identified by the LDA model, showing the most important keywords.

## Technologies Used

*   **Language:** Python 3.x
*   **Web Framework:** Streamlit
*   **Data Manipulation:** Pandas, NumPy
*   **Machine Learning:**
    *   Scikit-learn:
        *   `LogisticRegression` (Virality Prediction)
        *   `RandomForestClassifier` (Controversy Prediction)
        *   `TfidfVectorizer` (Virality/Controversy Prediction, Similarity Search)
        *   `CountVectorizer` (LDA Topic Modeling feature extraction)
        *   `OneHotEncoder` (Controversy Prediction - Topic Feature)
        *   `train_test_split`
        *   `cosine_similarity`
        *   `check_is_fitted`, `NotFittedError` (Model/Vectorizer validation)
    *   `joblib`: For loading the pre-trained LDA model (`lda_model.pkl`).
*   **Natural Language Processing (NLP):**
    *   NLTK: `SentimentIntensityAnalyzer` and `vader_lexicon` for sentiment analysis.
    *   WordCloud: For generating topic keyword visualizations.
*   **Time Series Analysis:**
    *   Statsmodels: `ExponentialSmoothing` for topic trend forecasting.
*   **Visualization:** Matplotlib, Seaborn
*   **Utilities:** warnings, calendar, datetime, timedelta, time

