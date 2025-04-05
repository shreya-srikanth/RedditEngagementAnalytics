import streamlit as st

st.set_page_config(page_title="Reddit Engagement Analytics", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Wedge
import joblib
import warnings
import calendar
from datetime import timedelta
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

# --- Scikit-learn Imports ---
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation
from scipy.sparse import hstack, csr_matrix
from sklearn.utils.validation import check_is_fitted # To check if vectorizer is fitted
from sklearn.exceptions import NotFittedError

# --- Time Series Imports ---
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module='statsmodels')
warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn')
warnings.filterwarnings("ignore", category=UserWarning, message="FixedFormatter should only be used together with FixedLocator")


# ----------------------------
# App Title & Description
# ----------------------------
st.title("📊 Reddit Engagement Analytics Dashboard")
st.markdown("Explore Reddit post data, predict virality, analyze topics, and view key statistics.")
st.divider()

# ----------------------------
# Download NLTK Data (VADER Lexicon)
# ----------------------------
if 'nltk_download_checked' not in st.session_state:
    st.session_state.nltk_download_checked = False

@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
        return True
    except (nltk.downloader.DownloadError, LookupError):
        if not st.session_state.nltk_download_checked:
             # Use st.empty() to create a placeholder
             download_status_placeholder = st.empty()
             download_status_placeholder.info("Checking/Downloading NLTK VADER lexicon...")
             try:
                 nltk.download('vader_lexicon', quiet=True)
                 st.session_state.nltk_download_checked = True
                 try:
                     nltk.data.find('sentiment/vader_lexicon.zip')
                     download_status_placeholder.success("VADER lexicon ready.")
                     return True
                 except Exception:
                      download_status_placeholder.error("Failed to locate VADER lexicon after download.")
                      return False
             except Exception as e:
                 download_status_placeholder.error(f"Failed to download VADER lexicon: {e}")
                 st.session_state.nltk_download_checked = True
                 return False
        else:
             try:
                 nltk.data.find('sentiment/vader_lexicon.zip')
                 return True
             except Exception:
                
                 return False

nltk_ready = download_nltk_data()

# ----------------------------
# LOAD & PREPROCESS DATA (Includes Sentiment Analysis)
# ----------------------------
@st.cache_data
def load_and_process_data():
    progress_text = "Loading and processing data..."
    my_bar = st.progress(0, text=progress_text)
    try:
        df = pd.read_csv("reddit_data_processed.csv")
        my_bar.progress(10, text="Raw data loaded.")

        required_cols = ['title', 'title_cleaned', 'created_timestamp', 'subreddit', 'id',
                         'author', 'score', 'num_comments', 'subreddit_subscribers', 'is_viral']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Error: Missing required columns: {', '.join(missing_cols)}")
            my_bar.progress(100, text="Data loading error.")
            return None
        my_bar.progress(20, text="Column check complete.")

        df['title_cleaned'] = df['title_cleaned'].fillna("")

        my_bar.progress(30, text="Processing dates...")
        if 'created_timestamp' in df.columns:
             df['created_date'] = pd.to_datetime(df['created_timestamp'], unit='s', errors='coerce')
        elif 'created_date' in df.columns:
             df['created_date'] = pd.to_datetime(df['created_date'], errors='coerce')
        else:
            st.error("Error: No 'created_timestamp' or 'created_date' column found.")
            my_bar.progress(100, text="Date processing error.")
            return None

        initial_rows = len(df)
        df.dropna(subset=['created_date'], inplace=True)
        rows_dropped = initial_rows - len(df)

        df['day_of_week'] = df['created_date'].dt.dayofweek
        my_bar.progress(50, text="Date processing complete.")

        my_bar.progress(60, text="Performing Sentiment Analysis...")
        if nltk_ready:
            try:
                sia = SentimentIntensityAnalyzer()
                df['sentiment_compound'] = df['title_cleaned'].apply(lambda x: sia.polarity_scores(x)['compound'])
                df['sentiment_label'] = df['sentiment_compound'].apply(
                    lambda score: 'positive' if score > 0.05 else ('negative' if score < -0.05 else 'neutral')
                )
                my_bar.progress(80, text="Sentiment analysis complete.")
            except Exception as e:
                st.error(f"Error during sentiment analysis: {e}")
                df['sentiment_compound'] = 0.0
                df['sentiment_label'] = 'neutral'
        else:
            st.warning("Skipping sentiment analysis as VADER lexicon is not ready.")
            df['sentiment_compound'] = 0.0
            df['sentiment_label'] = 'neutral'
            my_bar.progress(80, text="Sentiment analysis skipped.")


        numeric_cols_to_fill = ['sentiment_compound', 'num_comments', 'subreddit_subscribers', 'score']
        for col in numeric_cols_to_fill:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        my_bar.progress(100, text="Data preprocessing complete.")
        my_bar.empty()
        return df

    except FileNotFoundError:
        st.error("Error: `reddit_data_processed.csv` not found.")
        my_bar.empty()
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading/processing: {e}")
        my_bar.empty()
        return None

# Load data
data = load_and_process_data()

# ----------------------------
# TRAIN VIRALITY MODEL
# ----------------------------
@st.cache_resource
def train_virality_model(_data):
    if _data is None: return None, None
    required_cols = ['title_cleaned', 'sentiment_compound', 'num_comments', 'subreddit_subscribers', 'is_viral']
    if not all(col in _data.columns for col in required_cols): return None, None
    try:
        with st.spinner("Training Virality Prediction Model..."):
            tfidf_virality = TfidfVectorizer(max_features=1000, stop_words='english')
            X_text = tfidf_virality.fit_transform(_data['title_cleaned'])
            X_numeric = _data[['sentiment_compound', 'num_comments', 'subreddit_subscribers']].fillna(0)
            X_combined = hstack([X_text, csr_matrix(X_numeric.values)])
            y = _data['is_viral']

            if len(y.unique()) > 1 and y.value_counts().min() > 1:
                X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42, stratify=y)
                model = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
            else:
                 X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)
                 model = LogisticRegression(max_iter=1000, solver='liblinear')

            model.fit(X_train, y_train)
        return model, tfidf_virality
    except Exception as e:
        st.error(f"Error training virality model: {e}")
        return None, None

model_virality, tfidf_virality = train_virality_model(data)


# ----------------------------
# LOAD LDA MODEL & PREPARE TOPIC DATA
# ----------------------------
@st.cache_resource
def load_lda_and_prep_topics(_data):
    if _data is None: return None, None, None, _data
    required_cols = ['title_cleaned', 'created_date']
    if not all(col in _data.columns for col in required_cols): return None, None, None, _data

    try:
        with st.spinner("Loading LDA model and preparing topic data..."):
            lda_model = joblib.load("lda_model.pkl")
            num_topics = lda_model.n_components

            vectorizer_lda = CountVectorizer(max_features=1000, stop_words='english')
            doc_term_matrix = vectorizer_lda.fit_transform(_data['title_cleaned'])

            topic_distributions = lda_model.transform(doc_term_matrix)
            _data['dominant_topic'] = topic_distributions.argmax(axis=1)

            _data['week'] = _data['created_date'].dt.to_period('W')
            weekly_topic_counts = _data.groupby(['week', 'dominant_topic']).size().unstack(fill_value=0)

            for topic_idx in range(num_topics):
                if topic_idx not in weekly_topic_counts.columns:
                    weekly_topic_counts[topic_idx] = 0
            weekly_topic_counts = weekly_topic_counts.sort_index(axis=1)[list(range(num_topics))]
        return lda_model, vectorizer_lda, weekly_topic_counts, _data

    except FileNotFoundError:
        st.error("Error: `lda_model.pkl` not found. LDA features unavailable.")
        return None, None, None, _data
    except Exception as e:
        st.error(f"An error occurred during LDA model loading or topic processing: {e}")
        return None, None, None, _data

lda_model, vectorizer_lda, weekly_topic_counts, data = load_lda_and_prep_topics(data)

# Define Topic Labels
topic_labels = {
    0: "Data Science Careers", 1: "AI + PyTorch/TF", 2: "Help & Datasets",
    3: "ML/AI Concepts", 4: "Stats & Advice"
}

# ----------------------------
# HELPER & VISUALIZATION FUNCTIONS
# ----------------------------
def is_vectorizer_fitted(vectorizer):
    try:
        check_is_fitted(vectorizer)
        return True
    except NotFittedError:
        return False

def map_title_to_topic(title, lda_model, vectorizer):
   
    if lda_model is None or vectorizer is None: return None, None
    cleaned = title.lower()
    try:
        if not is_vectorizer_fitted(vectorizer):
             st.warning("Vectorizer not fitted in map_title_to_topic.")
             return None, None
        bow_vector = vectorizer.transform([cleaned])
        topic_dist = lda_model.transform(bow_vector)
        dominant_topic = int(np.argmax(topic_dist))
        return dominant_topic, topic_dist
    except Exception as e:
        st.warning(f"Error mapping title to topic: {e}")
        return None, None

def is_emerging_topic(topic, weekly_df, forecast_steps=4, min_data_points=10):
   
    if weekly_df is None or topic not in weekly_df.columns: return False
    series = weekly_df[topic].astype(float)
    if series.sum() < min_data_points or len(series.dropna()) < min_data_points: return False
    try:
        # Use RangeIndex for statsmodels if PeriodIndex causes issues
        model = ExponentialSmoothing(series.values, trend='add', seasonal=None, damped_trend=False)
        fit = model.fit()
        forecast = fit.forecast(forecast_steps)
        # Compare using iloc on the original series values
        if len(series) >= 2 and len(forecast) >=2:
            trend_now = series.iloc[-2:].mean()
            trend_next = forecast[:2].mean()
            return trend_next > trend_now
        elif len(series) >= 1 and len(forecast) >= 1:
             return forecast[0] > series.iloc[-1]
        else: return False
    except Exception as e:
        return False

def get_best_post_days(topic, historical_data):
    
    if historical_data is None or not all(col in historical_data.columns for col in ['dominant_topic', 'day_of_week']):
        return []
    try:
        topic_data = historical_data[historical_data['dominant_topic'] == topic]
        if topic_data.empty: return ["Data unavailable"]
        day_counts = topic_data['day_of_week'].value_counts()
        if day_counts.empty: return ["Data unavailable"]
        max_count = day_counts.max()
        best_day_indices = day_counts[day_counts == max_count].index.tolist()
        best_days = sorted([calendar.day_name[day_idx] for day_idx in best_day_indices], key=lambda x: list(calendar.day_name).index(x))
        return best_days
    except Exception as e:
        st.warning(f"Error analyzing best posting days for topic {topic}: {e}")
        return []

# (Visualization functions: show_virality_gauge, plot_*, generate_word_clouds - FIXED PLOTS)
def show_virality_gauge(score):
    
    score = max(0, min(1, score))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.1, 1.1); ax.axis('off')
    zones = [(180, 126, '#d9534f'), (126, 72, '#f0ad4e'), (72, 36, '#5cb85c'), (36, 0, '#4CAF50')]
    for start, end, color in zones:
        ax.add_patch(Wedge(center=(0, 0), r=1, theta1=end, theta2=start, facecolor=color, alpha=0.7))
    angle = 180 - (180 * score)
    x = np.cos(np.radians(angle)); y = np.sin(np.radians(angle))
    ax.arrow(0, 0, x * 0.8, y * 0.8, width=0.02, head_width=0.06, head_length=0.1, fc='black', ec='black')
    ax.add_patch(plt.Circle((0, 0), 0.05, color='black'))
    ax.text(0, -0.15, f"{score * 100:.1f}%", ha='center', fontsize=16, fontweight='bold')
    plt.tight_layout(); plt.close(fig)
    return fig

@st.cache_data
def plot_subreddit_dist(_data):
    
    if _data is None or 'subreddit' not in _data.columns: return None
    if _data['subreddit'].nunique() == 0: return None
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        sub_redd = _data["subreddit"].value_counts().head(15)
        sub_redd.plot(kind="bar", ax=ax, color='#FF4500', edgecolor='#FF4500')
        ax.set_title('Top 15 Subreddits by Post Count')
        ax.set_xlabel(''); ax.set_ylabel('Number of Posts')
        plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    except Exception as e:
        ax.text(0.5, 0.5, f'Plotting Error:\n{e}', ha='center', va='center')
    plt.close(fig)
    return fig

@st.cache_data
def plot_posts_by_year(_data):
    
    if _data is None or 'created_date' not in _data.columns: return None
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        year_d = _data.groupby(_data["created_date"].dt.year).size()
        if year_d.empty: return None
        year_d.plot(kind="bar", ax=ax, color='#FF4500', edgecolor='#FF4500')
        ax.set_title('Posts per Year'); ax.set_xlabel('Year'); ax.set_ylabel('Number of Posts')
        plt.xticks(rotation=45); plt.tight_layout()
    except Exception as e:
         ax.text(0.5, 0.5, f'Plotting Error:\n{e}', ha='center', va='center')
    plt.close(fig)
    return fig

@st.cache_data
def plot_posts_by_month(_data):
    
    if _data is None or 'created_date' not in _data.columns: return None
    fig, ax = plt.subplots(figsize=(10, 5))
    try:
        month_d = _data.groupby(_data["created_date"].dt.month).size().sort_index()
        if month_d.empty: return None
        try: month_d.index = [calendar.month_abbr[i] for i in month_d.index]
        except IndexError: pass
        month_d.plot(kind="bar", ax=ax, color='#FF4500', edgecolor='#FF4500')
        ax.set_title('Posts per Month (Across All Years)')
        ax.set_xlabel('Month'); ax.set_ylabel('Number of Posts')
        plt.xticks(rotation=0); plt.tight_layout()
    except Exception as e:
        ax.text(0.5, 0.5, f'Plotting Error:\n{e}', ha='center', va='center')
    plt.close(fig)
    return fig

@st.cache_data
def plot_posts_by_day_of_week(_data):
    # (function remains the same)
    if _data is None or 'day_of_week' not in _data.columns: return None
    fig, ax = plt.subplots(figsize=(10, 5))
    try:
        day_counts = _data['day_of_week'].value_counts().sort_index()
        if day_counts.empty: return None
        day_counts.index = [calendar.day_name[i] for i in day_counts.index]
        day_counts.plot(kind="bar", ax=ax, color='#FF4500', edgecolor='#FF4500')
        ax.set_title('Total Posts per Day of the Week')
        ax.set_xlabel('Day of Week'); ax.set_ylabel('Number of Posts')
        plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    except Exception as e:
        ax.text(0.5, 0.5, f'Plotting Error:\n{e}', ha='center', va='center')
    plt.close(fig)
    return fig

@st.cache_data
def plot_posts_by_day_of_month(_data):
    
    if _data is None or 'created_date' not in _data.columns: return None
    fig, ax = plt.subplots(figsize=(12, 5))
    try:
        day_d = _data.groupby(_data["created_date"].dt.day).size().sort_index()
        if day_d.empty: return None
        day_d.plot(kind="bar", ax=ax, color='#FF4500', edgecolor='#FF4500')
        ax.set_title('Posts per Day of the Month'); ax.set_xlabel('Day of Month'); ax.set_ylabel('Number of Posts')
        plt.xticks(rotation=0); ax.xaxis.set_major_locator(plt.MaxNLocator(15)); plt.tight_layout()
    except Exception as e:
        ax.text(0.5, 0.5, f'Plotting Error:\n{e}', ha='center', va='center')
    plt.close(fig)
    return fig

@st.cache_data
def plot_sentiment_dist(_data):
    if _data is None or 'sentiment_label' not in _data.columns: return None
    if _data['sentiment_label'].nunique() == 0: return None
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        
        sns.countplot(data=_data, x='sentiment_label', ax=ax, palette='coolwarm',
                      order=['positive', 'neutral', 'negative'], hue='sentiment_label', legend=False)
        ax.set_title("Sentiment Distribution"); ax.set_xlabel("Sentiment"); ax.set_ylabel("Count")
        plt.tight_layout()
    except Exception as e:
         ax.text(0.5, 0.5, f'Plotting Error:\n{e}', ha='center', va='center')
    plt.close(fig)
    return fig

@st.cache_data
def plot_sentiment_over_time(_data):
    if _data is None or 'created_date' not in _data.columns or 'sentiment_compound' not in _data.columns: return None
    if _data.empty: return None
    fig, ax = plt.subplots(figsize=(12, 6))
    try:
        _data_time_indexed = _data.set_index('created_date').sort_index()
        if _data_time_indexed.empty: return None
        
        sentiment_over_time = _data_time_indexed['sentiment_compound'].resample('ME').mean()
        if sentiment_over_time.empty: return None
        sentiment_over_time.plot(ax=ax, title="Average Sentiment Over Time (Monthly)")
        ax.set_xlabel("Time"); ax.set_ylabel("Average Sentiment Score")
        plt.tight_layout()
    except Exception as e:
        ax.text(0.5, 0.5, f'Error plotting sentiment over time:\n{e}', ha='center', va='center')
    plt.close(fig)
    return fig


@st.cache_data
def plot_score_by_sentiment(_data):
   
    if _data is None or not all(col in _data.columns for col in ['sentiment_label', 'score']): return None
    if _data.empty: return None
    plot_data = _data.copy()
    plot_data['score_log'] = np.log1p(plot_data['score'])
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        sns.boxplot(data=plot_data, x='sentiment_label', y='score_log', ax=ax,
                    hue='sentiment_label', palette='Set2', order=['positive', 'neutral', 'negative'], legend=False) # FIXED: hue + legend
        ax.set_title("Post Score (Log Scale) by Sentiment"); ax.set_xlabel("Sentiment"); ax.set_ylabel("Score (log(1 + score))")
        plt.tight_layout()
    except Exception as e:
        ax.text(0.5, 0.5, f'Error plotting score by sentiment:\n{e}', ha='center', va='center')
    plt.close(fig)
    return fig

@st.cache_resource
def generate_word_clouds(_lda_model, _vectorizer_lda):
    
    if _lda_model is None or _vectorizer_lda is None: return None
    if not is_vectorizer_fitted(_vectorizer_lda): return None
    word_cloud_figures = {}
    n_top_words = 30
    try:
        feature_names = _vectorizer_lda.get_feature_names_out()
        for idx, topic_weights in enumerate(_lda_model.components_):
            top_indices = topic_weights.argsort()[-n_top_words:]
            valid_indices = [i for i in top_indices if i < len(feature_names)]
            if not valid_indices: continue
            top_words_dict = {feature_names[i]: topic_weights[i] for i in valid_indices}
            if not top_words_dict: continue
            wordcloud = WordCloud(width=600, height=300, background_color='white').generate_from_frequencies(top_words_dict)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(wordcloud, interpolation='bilinear'); ax.axis('off')
            topic_name = topic_labels.get(idx, f"Topic {idx}")
            ax.set_title(f"Word Cloud for: {topic_name}")
            plt.tight_layout(); plt.close(fig)
            word_cloud_figures[idx] = fig
        return word_cloud_figures
    except Exception as e:
        st.warning(f"Could not generate word clouds: {e}")
        return None

# Generate word clouds once
word_cloud_figs = generate_word_clouds(lda_model, vectorizer_lda)



st.sidebar.header("Navigation")


section = st.sidebar.radio(
    "Select Analysis Section",
    [
        "🚀 Virality Predictor",
        "🧬 Similar Posts",
        "📈 Topic Trend Predictor",
        "📌 Key Stats & EDA"
    ],
    key="navigation_select",
    help="Choose the analysis you want to perform."
)


# ----------------------------
# MAIN CONTENT AREA
# ----------------------------

if data is None:
    st.error("Data loading failed. Cannot display analytics. Please check data file and logs.")
    st.stop()

# --- Display warnings if models failed to load ---
if model_virality is None or tfidf_virality is None:
    st.sidebar.warning("Virality model failed to load.")
if lda_model is None or vectorizer_lda is None or weekly_topic_counts is None:
     st.sidebar.warning("LDA model failed to load.")

# ----------------------------
# 🚀 VIRALITY PREDICTOR
# ----------------------------
if section == "🚀 Virality Predictor":
    st.header("🚀 Virality Predictor")
    st.markdown("Enter a potential Reddit post title to predict its likelihood of going viral.")
    if model_virality and tfidf_virality:
        user_input_title = st.text_input("Enter post title:", key="virality_input")
        if user_input_title:
            cleaned_title = user_input_title.lower()
            try:
                if not is_vectorizer_fitted(tfidf_virality):
                     st.error("Virality TF-IDF Vectorizer not fitted!")
                else:
                    X_text_input = tfidf_virality.transform([cleaned_title])
                    avg_sentiment = data['sentiment_compound'].mean() if 'sentiment_compound' in data else 0
                    avg_num_comments = data['num_comments'].mean() if 'num_comments' in data else 0
                    avg_subscribers = data['subreddit_subscribers'].mean() if 'subreddit_subscribers' in data else 0
                    X_numeric_input = csr_matrix([[avg_sentiment, avg_num_comments, avg_subscribers]])
                    X_combined_input = hstack([X_text_input, X_numeric_input])
                    prob = model_virality.predict_proba(X_combined_input)[0][1]
                    st.subheader("Prediction Result")
                    fig = show_virality_gauge(prob)
                    if fig: st.pyplot(fig)
            except Exception as e:
                st.error(f"An error occurred during virality prediction: {e}")
    else:
         st.warning("Virality prediction model not available.")


# ----------------------------
# 🧬 SIMILAR POSTS SECTION
# ----------------------------
elif section == "🧬 Similar Posts":
    st.header("🧬 Find Similar Posts")
    st.markdown("Find textually similar posts based on TF-IDF cosine similarity.")
    if tfidf_virality:
        input_title_similar = st.text_input("Enter post title:", key="similarity_input")
        if input_title_similar:
            try:
                if not is_vectorizer_fitted(tfidf_virality):
                     st.error("Similarity TF-IDF Vectorizer not fitted!")
                else:
                    tfidf_matrix_full = tfidf_virality.transform(data['title_cleaned'])
                    input_vec = tfidf_virality.transform([input_title_similar.lower()])
                    similarity_scores = cosine_similarity(input_vec, tfidf_matrix_full).flatten()
                    num_similar = 5
                    sorted_indices = np.argsort(similarity_scores)[::-1]
                    top_indices = []
                    input_lower = input_title_similar.lower()
                    for idx in sorted_indices:
                        if len(top_indices) >= num_similar: break # Stop if we found enough
                        # Check if the potential match is the input itself (if present in data)
                        is_self_match = data.iloc[idx]['title'].lower() == input_lower
                        if not is_self_match:
                             top_indices.append(idx)

                    if not top_indices:
                         st.info("No similar posts found (excluding potential self-match).")
                    else:
                        st.subheader(f"Top {len(top_indices)} Similar Posts:")
                        similar_df = data.iloc[top_indices][['title', 'score', 'subreddit', 'id']].copy()
                        similar_df['similarity'] = similarity_scores[top_indices]
                        st.dataframe(
                            similar_df[['title', 'similarity', 'score', 'subreddit', 'id']].reset_index(drop=True),
                            use_container_width=True,
                            column_config={"similarity": st.column_config.NumberColumn(format="%.3f")}
                        )
            except Exception as e:
                st.error(f"An error occurred during similarity search: {e}")
    else:
         st.warning("Similarity vectorizer (TF-IDF) not available.")

# ----------------------------
# 📈 TOPIC TREND PREDICTOR
# ----------------------------
elif section == "📈 Topic Trend Predictor":
    st.header("📈 Topic Trend Predictor")
    st.markdown("Identify a post's topic, see if it's trending, find best posting days, and view overall topic trends.")

    if lda_model and vectorizer_lda and weekly_topic_counts is not None:
        st.subheader("Analyze Your Post Title")
        user_input_topic_title = st.text_input("Enter post title:", key="topic_input", value="word document saving issue")

        if user_input_topic_title:
            dominant_topic, _ = map_title_to_topic(user_input_topic_title, lda_model, vectorizer_lda)
            if dominant_topic is not None:
                topic_name = topic_labels.get(dominant_topic, f"Unknown Topic {dominant_topic}")
                st.info(f"**Identified Dominant Topic:** {dominant_topic} - {topic_name}")
                is_emerging = is_emerging_topic(dominant_topic, weekly_topic_counts)
                
                st.metric("Topic Trend Status", "Emerging" if is_emerging else "Stable/Declining")
                best_days = get_best_post_days(dominant_topic, data)
                if best_days and best_days != ["Data unavailable"]:
                    st.success(f"🗓️ **Historically Best Day(s) to Post:** {', '.join(best_days)}")
                else:
                     st.info("Could not determine specific best posting days (insufficient data for this topic).")
            else:
                st.error("Could not determine the topic for the entered title.")
        st.divider()

        st.subheader("Overall Weekly Topic Trends and Forecasts")
        if weekly_topic_counts.empty:
             st.warning("Weekly topic counts data is empty.")
        else:
            valid_topics = [t for t in weekly_topic_counts.columns if t in topic_labels]
            if not valid_topics: st.warning("No valid topics found matching labels.")
            else:
                cols = st.columns(2)
                col_idx = 0
                # Convert index to string for plotting here if needed
                plot_weekly_counts = weekly_topic_counts.copy()
                plot_weekly_counts.index = plot_weekly_counts.index.astype(str)

                for topic_idx in sorted(valid_topics):
                    series = plot_weekly_counts[topic_idx]
                    topic_name = topic_labels[topic_idx]
                    if series.sum() < 10: continue
                    try:
                        # Use series values directly for model fitting
                        model_trend = ExponentialSmoothing(series.astype(float).values, trend='add', seasonal=None, damped_trend=True)
                        fit = model_trend.fit()
                        forecast_periods = 10
                        forecast_values = fit.forecast(forecast_periods)

                        # Create forecast index as strings relative to the last actual index string
                        last_actual_week_str = series.index[-1]
                        # Basic way to increment week string 'YYYY-WW' -> 'YYYY-WW+1' etc.
                        # This is approximate and might drift over year ends, but often sufficient for plotting short forecasts
                        try:
                            yr, wk = map(int, last_actual_week_str.split('-'))
                            forecast_index_str = []
                            for i in range(1, forecast_periods + 1):
                                 current_wk = wk + i
                                 current_yr = yr
                                 if current_wk > 52: # Simple year rollover
                                     current_yr += (current_wk -1) // 52
                                     current_wk = ((current_wk -1) % 52) + 1
                                 forecast_index_str.append(f"{current_yr}-{current_wk:02d}")
                        except ValueError: # Fallback if index format isn't 'YYYY-WW'
                             forecast_index_str = [f"F{i+1}" for i in range(forecast_periods)]


                        fig_topic, ax_topic = plt.subplots(figsize=(8, 4))
                        ax_topic.plot(series.index, series.values, label="Actual", marker='o', markersize=4)
                        ax_topic.plot(forecast_index_str, forecast_values, label="Forecast", linestyle='--', marker='x', markersize=5, color='red') # Use generated string index
                        ax_topic.set_title(f"{topic_idx}: {topic_name}")
                        ax_topic.set_ylabel("Post Count")
                        ax_topic.legend(); ax_topic.grid(True, alpha=0.5)
                        ax_topic.tick_params(axis='x', rotation=30, labelsize=8)
                        num_ticks = min(8, len(series.index) // 2 if len(series.index) > 0 else 1)
                        ax_topic.xaxis.set_major_locator(plt.MaxNLocator(num_ticks if num_ticks > 0 else 1))
                        plt.tight_layout(); plt.close(fig_topic)
                        with cols[col_idx % 2]: st.pyplot(fig_topic)
                        col_idx += 1
                    except Exception as plot_err:
                        # Provide more context in the warning
                        st.warning(f"Plot failed for Topic {topic_idx} ({topic_name}): {plot_err}")
    else:
         st.warning("Topic modeling components unavailable.")


# ----------------------------
# 📌 KEY STATS & EDA SECTION
# ----------------------------
elif section == "📌 Key Stats & EDA":
    st.header("📌 Key Statistics & Exploratory Data Analysis")

    st.subheader("Overall Dataset Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("🔢 Total Posts Analyzed", f"{len(data):,}")
    col2.metric("📈 Average Score", f"{data['score'].mean():.2f}" if 'score' in data else "N/A")
    col3.metric("💬 Average Comments", f"{data['num_comments'].mean():.2f}" if 'num_comments' in data else "N/A")
    st.divider()

   
    st.subheader("Sentiment Analysis Insights")
    st.markdown("**Top 5 Positive Titles**")
    st.dataframe(data.nlargest(5, 'sentiment_compound')[['title', 'sentiment_compound']].reset_index(drop=True),
                 height=200, use_container_width=True,
                 column_config={"sentiment_compound": st.column_config.NumberColumn(format="%.3f")})

    st.markdown("**Top 5 Negative Titles**")
    st.dataframe(data.nsmallest(5, 'sentiment_compound')[['title', 'sentiment_compound']].reset_index(drop=True),
                 height=200, use_container_width=True,
                 column_config={"sentiment_compound": st.column_config.NumberColumn(format="%.3f")})
    

    with st.expander("Show Sentiment vs. Score and Sentiment Over Time Plots", expanded=False):
         senti_plot_col1, senti_plot_col2 = st.columns(2)
         with senti_plot_col1:
              fig = plot_score_by_sentiment(data)
              if fig: st.pyplot(fig) 
              else: st.info("Score vs Sentiment plot unavailable.")
         with senti_plot_col2:
             fig = plot_sentiment_over_time(data)
             if fig: st.pyplot(fig) 
             else: st.info("Sentiment over time plot unavailable.")
    st.divider()


    st.subheader("Post Distributions")
    with st.expander("Show Distribution Plots", expanded=False):
        eda_col1, eda_col2 = st.columns(2)
        with eda_col1:
            fig = plot_subreddit_dist(data)
            if fig: st.pyplot(fig) 
            else: st.info("Subreddit plot unavailable.")

            fig = plot_posts_by_month(data)
            if fig: st.pyplot(fig) 
            else: st.info("Month distribution plot unavailable.")

            fig = plot_posts_by_day_of_week(data)
            if fig: st.pyplot(fig) 
            else: st.info("Day of week distribution plot unavailable.")


        with eda_col2:
            fig = plot_posts_by_year(data)
            if fig: st.pyplot(fig) 
            else: st.info("Year distribution plot unavailable.")

            fig = plot_posts_by_day_of_month(data)
            if fig: st.pyplot(fig) 
            else: st.info("Day of month distribution plot unavailable.")

            fig = plot_sentiment_dist(data)
            if fig: st.pyplot(fig) 
            else: st.info("Sentiment distribution plot unavailable.")

    st.divider()

    st.subheader("LDA Topic Word Clouds")
    with st.expander("Show Topic Word Clouds", expanded=False):
        if word_cloud_figs:
            wc_cols = st.columns(2)
            wc_col_idx = 0
            sorted_topic_indices = sorted(word_cloud_figs.keys())
            for topic_idx in sorted_topic_indices:
                 fig = word_cloud_figs[topic_idx]
                 with wc_cols[wc_col_idx % 2]:
                      if fig: st.pyplot(fig)
                 wc_col_idx += 1
        else:
            st.warning("Word clouds could not be generated.")


