# Reddit Engagement Analytics Dashboard - Version 1

This project is a Streamlit web application designed to analyze Reddit post data. It provides insights into post engagement, predicts virality, explores topics using LDA, identifies similar posts, and presents key statistics through an interactive dashboard.

##  Features

The dashboard is divided into several key sections:

1.  ** Virality Predictor:**
    *   Input a potential Reddit post title.
    *   Predicts the likelihood of the post going viral based on features like title text (TF-IDF), sentiment, average comment count, and average subreddit subscribers.
    *   Displays the prediction using an intuitive gauge visualization.

2.  ** Similar Posts:**
    *   Input a Reddit post title.
    *   Finds and displays the top 5 most textually similar posts from the dataset using TF-IDF and Cosine Similarity.
    *   Shows the title, similarity score, actual score, subreddit, and ID of the similar posts.

3.  ** Topic Trend Predictor:**
    *   **Title Analysis:**
        *   Input a Reddit post title.
        *   Identifies the dominant topic of the title using a pre-trained Latent Dirichlet Allocation (LDA) model.
        *   Predicts if the identified topic is currently trending (Emerging or Stable/Declining) based on recent post volume using Exponential Smoothing.
        *   Suggests the historically best day(s) of the week to post about that specific topic based on past activity patterns.
    *   **Overall Trends:**
        *   Displays weekly trend plots (actual post counts + forecast) for each major topic identified by the LDA model, allowing users to visually track topic popularity over time.

4.  ** Key Stats & EDA:**
    *   **Top Metrics:** Shows overall dataset statistics like Total Posts Analyzed, Average Score, and Average Comments.
    *   **Sentiment Analysis:**
        *   Displays the top 5 most positive and top 5 most negative post titles based on NLTK VADER sentiment analysis.
        *   Includes plots showing the distribution of sentiment labels (positive, neutral, negative), average sentiment score over time (monthly), and the relationship between sentiment and post score (using a boxplot with log scale).
    *   **Post Distributions:** Provides various visualizations using Matplotlib and Seaborn:
        *   Top 15 Subreddits by post count.
        *   Distribution of posts per Year.
        *   Distribution of posts per Month.
        *   Distribution of posts per Day of the Week.
        *   Distribution of posts per Day of the Month.
    *   **LDA Topic Word Clouds:** Displays word clouds for each topic identified by the LDA model, showing the most important keywords associated with each topic.

##  Technologies Used

*   **Language:** Python 3.x
*   **Web Framework:** Streamlit
*   **Data Manipulation:** Pandas, NumPy
*   **Machine Learning:**
    *   Scikit-learn:
        *   `LogisticRegression` (for Virality Prediction)
        *   `TfidfVectorizer` (for Virality Prediction, Similarity Search)
        *   `CountVectorizer` (for LDA Topic Modeling feature extraction)
        *   `train_test_split`
        *   `cosine_similarity`
        *   `check_is_fitted`, `NotFittedError` (for validation)
    *   `joblib`: For loading the pre-trained LDA model (`lda_model.pkl`).
*   **Natural Language Processing (NLP):**
    *   NLTK: `SentimentIntensityAnalyzer` and `vader_lexicon` for sentiment analysis.
    *   WordCloud: For generating topic keyword visualizations.
*   **Time Series Analysis:**
    *   Statsmodels: `ExponentialSmoothing` for topic trend forecasting.
*   **Visualization:** Matplotlib, Seaborn
*   **Utilities:** warnings, calendar, datetime, timedelta


