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
import time

# --- Scikit-learn Imports ---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation
from scipy.sparse import hstack, csr_matrix
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# --- Time Series Imports ---
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module='statsmodels')
warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn')
warnings.filterwarnings("ignore", category=UserWarning, message="FixedFormatter should only be used together with FixedLocator")
warnings.filterwarnings("ignore", message="X does not have valid feature names")


# ----------------------------
# App Title & Description
# ----------------------------
st.title("📊 Reddit Engagement Analytics Dashboard")
st.markdown("Explore Reddit post data, predict virality & controversy, analyze topics, and view key statistics.")
st.divider()

# ----------------------------
# Download NLTK Data (VADER Lexicon)
# ----------------------------
if 'nltk_download_checked' not in st.session_state: st.session_state.nltk_download_checked = False
@st.cache_resource
def download_nltk_data():
    try: nltk.data.find('sentiment/vader_lexicon.zip'); return True
    except (nltk.downloader.DownloadError, LookupError):
        if not st.session_state.nltk_download_checked:
             ph = st.empty(); ph.info("Checking/Downloading NLTK VADER lexicon...")
             try:
                 nltk.download('vader_lexicon', quiet=True); st.session_state.nltk_download_checked = True
                 try: nltk.data.find('sentiment/vader_lexicon.zip'); ph.success("VADER lexicon ready."); time.sleep(1.5); ph.empty(); return True
                 except Exception: ph.error("Failed locate VADER post-download."); return False
             except Exception as e: ph.error(f"Failed VADER download: {e}"); st.session_state.nltk_download_checked = True; return False
        else:
             try: nltk.data.find('sentiment/vader_lexicon.zip'); return True
             except Exception: return False
nltk_ready = download_nltk_data()
sia_global = SentimentIntensityAnalyzer() if nltk_ready else None


# ----------------------------
# LOAD & PREPROCESS DATA (Includes Controversy Label Creation)
# ----------------------------
@st.cache_data
def load_and_process_data():
    ph = st.empty(); ph.info("Loading and processing data...")
    try:
        df = pd.read_csv("reddit_data_processed.csv")
        required=['title','title_cleaned','created_timestamp','subreddit','id','author','score','num_comments','subreddit_subscribers','is_viral']
        missing=[c for c in required if c not in df.columns]
        if missing: st.error(f"Missing columns: {missing}"); ph.empty(); return None
        df['title_cleaned'] = df['title_cleaned'].fillna("")
        if 'created_timestamp' in df.columns: df['created_date'] = pd.to_datetime(df['created_timestamp'], unit='s', errors='coerce')
        elif 'created_date' in df.columns: df['created_date'] = pd.to_datetime(df['created_date'], errors='coerce')
        else: st.error("No date column found."); ph.empty(); return None
        df.dropna(subset=['created_date'], inplace=True); df['day_of_week'] = df['created_date'].dt.dayofweek
        if sia_global:
            try:
                df['sentiment_compound'] = df['title_cleaned'].apply(lambda x: sia_global.polarity_scores(x)['compound'])
                df['sentiment_label'] = df['sentiment_compound'].apply(lambda s: 'positive' if s > 0.05 else ('negative' if s < -0.05 else 'neutral'))
            except Exception: df['sentiment_compound'] = 0.0; df['sentiment_label'] = 'neutral'
        else: df['sentiment_compound'] = 0.0; df['sentiment_label'] = 'neutral'
        epsilon = 1e-6
        df['controversy_ratio'] = df['num_comments'].fillna(0) / (abs(df['score'].fillna(0)) + 1 + epsilon)
        valid_ratios = df.loc[df['controversy_ratio'].notna() & np.isfinite(df['controversy_ratio']), 'controversy_ratio']
        threshold_ratio = valid_ratios.quantile(0.90) if not valid_ratios.empty else 3.0
        min_comments = 20; df['is_controversial'] = 0
        df.loc[(df['controversy_ratio'] > threshold_ratio) & (df['num_comments'].fillna(0) >= min_comments),'is_controversial'] = 1
        num_cols = ['sentiment_compound','num_comments','subreddit_subscribers','score','controversy_ratio']
        for col in num_cols:
             if col in df.columns:
                 median_val = df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else 0
                 df[col] = df[col].fillna(median_val)
        if 'dominant_topic' not in df.columns: df['dominant_topic'] = -1
        ph.empty(); return df
    except FileNotFoundError: st.error("`reddit_data_processed.csv` not found."); ph.empty(); return None
    except Exception as e: st.error(f"Data loading error: {e}"); ph.empty(); return None
data = load_and_process_data()


# ----------------------------
# Calculate Global Averages/Medians
# ----------------------------
# Initialize first
avg_sentiment_global = 0.0
avg_num_comments_global = 0.0
avg_subscribers_global = 0.0       # MEAN for virality predictor
median_sub_subscribers_global = 0.0 # MEDIAN for controversy predictor

# Calculate if data is available
if data is not None:
    avg_sentiment_global = data['sentiment_compound'].mean() if 'sentiment_compound' in data else 0.0
    avg_num_comments_global = data['num_comments'].mean() if 'num_comments' in data else 0.0
    # Calculate BOTH mean and median for subscribers
    if 'subreddit_subscribers' in data and pd.api.types.is_numeric_dtype(data['subreddit_subscribers']):
        # Fill potential NaNs before calculating mean/median
        sub_data = data['subreddit_subscribers'].fillna(0) # Use 0 for missing subs for calculation
        avg_subscribers_global = sub_data.mean()
        median_sub_subscribers_global = sub_data.median()
    else:
        # Keep defaults if column is missing or not numeric
        avg_subscribers_global = 0.0
        median_sub_subscribers_global = 0.0


# ----------------------------
# TRAIN / LOAD MODELS
# ----------------------------

# (Virality Model Training)
@st.cache_resource
def train_virality_model(_data):
    if _data is None: return None, None
    required=['title_cleaned','sentiment_compound','num_comments','subreddit_subscribers','is_viral']
    if not all(c in _data.columns for c in required): return None, None
    try:
        with st.spinner("Training Virality Model..."):
            tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
            X_text = tfidf.fit_transform(_data['title_cleaned'].fillna(""))
            X_numeric = _data[['sentiment_compound', 'num_comments', 'subreddit_subscribers']].fillna(0)
            X_combined = hstack([csr_matrix(X_text), csr_matrix(X_numeric.values)])
            y = _data['is_viral']
            X_train, _, y_train, _ = train_test_split(X_combined, y, test_size=0.2, random_state=42)
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
        return model, tfidf
    except Exception as e: st.error(f"Virality training error: {e}"); return None, None
model, tfidf = train_virality_model(data)
tfidf_virality = tfidf 


# (LDA Model Loading)
@st.cache_resource
def load_lda_and_prep_topics(_data):
    if _data is None: return None, None, None, _data
    req=['title_cleaned','created_date']
    if not all(c in _data.columns for c in req): return None, None, None, _data
    try:
        with st.spinner("Loading LDA Model & Components..."):
            m_lda=joblib.load("lda_model.pkl"); n_topics=m_lda.n_components
            v_lda=CountVectorizer(max_features=1000,stop_words='english'); dtm=v_lda.fit_transform(_data['title_cleaned'].fillna(""))
            td=m_lda.transform(dtm); _data['dominant_topic']=td.argmax(axis=1)
            _data['week']=_data['created_date'].dt.to_period('W'); wtc=_data.groupby(['week','dominant_topic']).size().unstack(fill_value=0)
            for i in range(n_topics):
                if i not in wtc.columns: wtc[i]=0
            wtc=wtc.sort_index(axis=1)[list(range(n_topics))]
        return m_lda, v_lda, wtc, _data
    except FileNotFoundError: st.error("`lda_model.pkl` not found."); return None, None, None, _data
    except Exception as e: st.error(f"LDA loading error: {e}"); return None, None, None, _data
lda_model, vectorizer_lda, weekly_topic_counts, data = load_lda_and_prep_topics(data)

# (Controversy Model Training)
@st.cache_resource
def train_controversy_classifier_rf(_data, text_col='title_cleaned', target_col='is_controversial'):
    if _data is None: return None, None, None
    numeric=['sentiment_compound','subreddit_subscribers']; cat='dominant_topic'; required=[text_col]+numeric+[cat,target_col]
    if not all(c in _data.columns for c in required): return None, None, None
    df=_data.copy(); df[text_col]=df[text_col].fillna("")
    for c in numeric: df[c]=df[c].fillna(df[c].median())
    df[cat]=df[cat].fillna(-1).astype(int); df[target_col]=df[target_col].astype(int)
    y=df[target_col]
    if y.nunique()<2: return None, None, None
    X_txt=df[text_col]; X_num=df[numeric]; X_cat=df[[cat]]
    try: Xtt,Xtx,Xnt,Xnx,Xct,Xcx,yt,yx=train_test_split(X_txt,X_num,X_cat,y,test_size=0.2,random_state=42,stratify=y)
    except ValueError: Xtt,Xtx,Xnt,Xnx,Xct,Xcx,yt,yx=train_test_split(X_txt,X_num,X_cat,y,test_size=0.2,random_state=42)
    try:
        with st.spinner("Training Controversy Prediction Model..."):
            tfidf_c=TfidfVectorizer(max_features=1500,stop_words='english',min_df=5,ngram_range=(1,2))
            Xtt_tf=tfidf_c.fit_transform(Xtt); Xtx_tf=tfidf_c.transform(Xtx)
            Xnt_sp=csr_matrix(Xnt.values); Xnx_sp=csr_matrix(Xnx.values)
            ohe_c=OneHotEncoder(handle_unknown='ignore',sparse_output=True)
            Xct_ohe=ohe_c.fit_transform(Xct); Xcx_ohe=ohe_c.transform(Xcx)
            Xtr_comb=hstack([Xtt_tf, Xnt_sp, Xct_ohe]); Xts_comb=hstack([Xtx_tf, Xnx_sp, Xcx_ohe])
            model_c=RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=42,n_jobs=-1,max_depth=15,min_samples_leaf=5)
            model_c.fit(Xtr_comb, yt)
        return model_c, tfidf_c, ohe_c
    except Exception as e: st.error(f"Controversy training error: {e}"); return None, None, None
rf_controversy_model, rf_controversy_tfidf, rf_controversy_ohe = train_controversy_classifier_rf(data)


# Define Topic Labels
topic_labels = {0: "Data Science Careers", 1: "AI + PyTorch/TF", 2: "Help & Datasets", 3: "ML/AI Concepts", 4: "Stats & Advice"}

# ----------------------------
# HELPER & VISUALIZATION FUNCTIONS
# ----------------------------

def is_vectorizer_fitted(vectorizer):
    try: check_is_fitted(vectorizer); return True
    except: return False
def map_title_to_topic(title, lda_model_map, vectorizer_lda_map):
    if lda_model_map is None or vectorizer_lda_map is None or not title: return None
    cleaned = title.lower()
    try:
        if not is_vectorizer_fitted(vectorizer_lda_map): return None
        bow = vectorizer_lda_map.transform([cleaned]); td = lda_model_map.transform(bow)
        return int(np.argmax(td))
    except Exception: return None
def is_emerging_topic(topic, weekly_df, forecast_steps=4, min_data_points=10):
    if weekly_df is None or topic not in weekly_df.columns: return False
    s = weekly_df[topic].astype(float);
    if s.sum()<min_data_points or len(s.dropna())<min_data_points: return False
    try:
        m = ExponentialSmoothing(s.values, trend='add', seasonal=None, damped_trend=False).fit()
        f = m.forecast(forecast_steps)
        if len(s)>=2 and len(f)>=2: return f[:2].mean() > s.iloc[-2:].mean()
        elif len(s)>=1 and len(f)>=1: return f[0] > s.iloc[-1]
        else: return False
    except Exception: return False
def get_best_post_days(topic, historical_data):
    if historical_data is None or not all(c in historical_data.columns for c in ['dominant_topic','day_of_week']): return []
    try:
        td = historical_data[historical_data['dominant_topic'] == topic];
        if td.empty: return ["Data unavailable"]
        dc = td['day_of_week'].value_counts();
        if dc.empty: return ["Data unavailable"]
        mc = dc.max(); bdi = dc[dc==mc].index.tolist()
        return sorted([calendar.day_name[i] for i in bdi], key=lambda x: list(calendar.day_name).index(x))
    except: return []
def predict_controversy_probability(input_title, model_controv, tfidf_vec_controv, ohe_topic_enc_controv,
                                   lda_model_map, vectorizer_lda_map, sentiment_analyzer,
                                   default_sub_subscribers, topic_column_name='dominant_topic'):
    req_models = [model_controv, tfidf_vec_controv, ohe_topic_enc_controv, lda_model_map, vectorizer_lda_map]
    if not all(req_models): st.warning("Controversy: Missing components."); return None
    if not sentiment_analyzer: st.warning("Controversy: Missing SIA."); return None
    if not isinstance(input_title, str) or not input_title: return None
    if not is_vectorizer_fitted(tfidf_vec_controv) or not is_vectorizer_fitted(vectorizer_lda_map): st.warning("Controversy: Vectorizers not fitted."); return None
    try:
        cleaned = input_title.lower()
        X_text = tfidf_vec_controv.transform([cleaned])
        sentiment = sentiment_analyzer.polarity_scores(cleaned)['compound']
        topic = map_title_to_topic(cleaned, lda_model_map, vectorizer_lda_map)
        topic = topic if topic is not None else -1
        X_cat_df = pd.DataFrame([[topic]], columns=[topic_column_name])
        X_cat = ohe_topic_enc_controv.transform(X_cat_df)
        X_num = csr_matrix([[sentiment, default_sub_subscribers]])
        X_comb = hstack([X_text, X_num, X_cat]) # Order: Text, Numeric, Categorical
        probs = model_controv.predict_proba(X_comb)
        return probs[0][1]
    except Exception as e: st.warning(f"Controversy prediction calc failed: {e}"); return None
def show_virality_gauge(score):
    score=max(0,min(1,score));fig,ax=plt.subplots(figsize=(6,4));ax.set_xlim(-1.1,1.1);ax.set_ylim(-0.1,1.1);ax.axis('off')
    zones=[(180,126,'#d9534f'),(126,72,'#f0ad4e'),(72,36,'#5cb85c'),(36,0,'#4CAF50')]
    for s,e,c in zones:ax.add_patch(Wedge(center=(0,0),r=1,theta1=e,theta2=s,facecolor=c,alpha=0.7))
    a=180-(180*score);x=np.cos(np.radians(a));y=np.sin(np.radians(a));ax.arrow(0,0,x*0.8,y*0.8,width=0.02,head_width=0.06,head_length=0.1,fc='black',ec='black')
    ax.add_patch(plt.Circle((0,0),0.05,color='black'));ax.text(0,-0.15,f"{score*100:.1f}%",ha='center',fontsize=16,fontweight='bold');plt.tight_layout();plt.close(fig);return fig

@st.cache_data
def plot_subreddit_dist(_data):
    if _data is None or 'subreddit' not in _data.columns or _data['subreddit'].nunique()==0:return None
    fig,ax=plt.subplots(figsize=(10,6))
    try:
        sr=_data["subreddit"].value_counts().head(15)
        sr.plot(kind="bar",ax=ax,color='#FF4500')
        ax.set_title('Top 15 Subreddits')
        ax.set_xlabel('')
        ax.set_ylabel('Posts')
        plt.xticks(rotation=45,ha='right')
        plt.tight_layout()
    except Exception as e:
        ax.text(0.5,0.5,f'Plot Err:\n{e}',ha='center',va='center')
    plt.close(fig)
    return fig

@st.cache_data
def plot_posts_by_year(_data):
    if _data is None or 'created_date' not in _data.columns: return None
    fig,ax=plt.subplots(figsize=(8,5)) 
    try:
        yd=_data.groupby(_data["created_date"].dt.year).size()
        if yd.empty: return None
        yd.plot(kind="bar",ax=ax,color='#FF4500')
        ax.set_title('Posts per Year')
        ax.set_xlabel('Year')
        ax.set_ylabel('Posts')
        plt.xticks(rotation=45)
        plt.tight_layout()
    except Exception as e:
        ax.text(0.5,0.5,f'Plot Err:\n{e}',ha='center',va='center')
    plt.close(fig)
    return fig

@st.cache_data
def plot_posts_by_month(_data):
    if _data is None or 'created_date' not in _data.columns: return None
    fig,ax=plt.subplots(figsize=(10,5)) 
    try:
        md=_data.groupby(_data["created_date"].dt.month).size().sort_index()
        if md.empty: return None
        try: md.index=[calendar.month_abbr[i] for i in md.index]
        except IndexError: pass
        md.plot(kind="bar",ax=ax,color='#FF4500')
        ax.set_title('Posts per Month')
        ax.set_xlabel('Month')
        ax.set_ylabel('Posts')
        plt.xticks(rotation=0)
        plt.tight_layout()
    except Exception as e:
        ax.text(0.5,0.5,f'Plot Err:\n{e}',ha='center',va='center')
    plt.close(fig)
    return fig

@st.cache_data
def plot_posts_by_day_of_week(_data):
    if _data is None or 'day_of_week' not in _data.columns: return None
    fig,ax=plt.subplots(figsize=(10,5)) 
    try:
        dc=_data['day_of_week'].value_counts().sort_index()
        if dc.empty: return None
        dc.index=[calendar.day_name[i] for i in dc.index]
        dc.plot(kind="bar",ax=ax,color='#FF4500')
        ax.set_title('Total Posts per Day of Week')
        ax.set_xlabel('Day')
        ax.set_ylabel('Posts')
        plt.xticks(rotation=45,ha='right')
        plt.tight_layout()
    except Exception as e:
        ax.text(0.5,0.5,f'Plot Err:\n{e}',ha='center',va='center')
    plt.close(fig)
    return fig

@st.cache_data
def plot_posts_by_day_of_month(_data):
    if _data is None or 'created_date' not in _data.columns: return None
    fig,ax=plt.subplots(figsize=(12,5)) 
    try:
        dd=_data.groupby(_data["created_date"].dt.day).size().sort_index()
        if dd.empty: return None
        dd.plot(kind="bar",ax=ax,color='#FF4500')
        ax.set_title('Posts per Day of Month')
        ax.set_xlabel('Day')
        ax.set_ylabel('Posts')
        plt.xticks(rotation=0)
        ax.xaxis.set_major_locator(plt.MaxNLocator(15))
        plt.tight_layout()
    except Exception as e:
        ax.text(0.5,0.5,f'Plot Err:\n{e}',ha='center',va='center')
    plt.close(fig)
    return fig

@st.cache_data
def plot_sentiment_dist(_data):
    if _data is None or 'sentiment_label' not in _data.columns or _data['sentiment_label'].nunique()==0: return None
    fig,ax=plt.subplots(figsize=(8,5)) 
    try:
        sns.countplot(data=_data,x='sentiment_label',ax=ax,palette='coolwarm',order=['positive','neutral','negative'],hue='sentiment_label',legend=False)
        ax.set_title("Sentiment Distribution")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        plt.tight_layout()
    except Exception as e:
        ax.text(0.5,0.5,f'Plot Err:\n{e}',ha='center',va='center')
    plt.close(fig)
    return fig

@st.cache_data
def plot_sentiment_over_time(_data):
    if _data is None or 'created_date' not in _data.columns or 'sentiment_compound' not in _data.columns or _data.empty: return None
    fig,ax=plt.subplots(figsize=(12,6)) 
    try:
        dti=_data.set_index('created_date').sort_index()
        if dti.empty: return None
        sot=dti['sentiment_compound'].resample('ME').mean() # Use 'ME'
        if sot.empty: return None
        sot.plot(ax=ax,title="Avg Sentiment Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Avg Sentiment")
        plt.tight_layout()
    except Exception as e:
        ax.text(0.5,0.5,f'Plot Err:\n{e}',ha='center',va='center')
    plt.close(fig)
    return fig

@st.cache_data
def plot_score_by_sentiment(_data):
    if _data is None or not all(c in _data.columns for c in ['sentiment_label','score']) or _data.empty: return None
    pld=_data.copy()
    pld['score_log']=np.log1p(pld['score'])
    fig,ax=plt.subplots(figsize=(10,6)) 
    try:
        sns.boxplot(data=pld,x='sentiment_label',y='score_log',ax=ax,hue='sentiment_label',palette='Set2',order=['positive','neutral','negative'],legend=False)
        ax.set_title("Score (Log) by Sentiment")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Score (log(1+score))")
        plt.tight_layout()
    except Exception as e:
        ax.text(0.5,0.5,f'Plot Err:\n{e}',ha='center',va='center')
    plt.close(fig)
    return fig
@st.cache_resource
def generate_word_clouds(_lda_model, _vectorizer_lda):
    if _lda_model is None or _vectorizer_lda is None or not is_vectorizer_fitted(_vectorizer_lda): return None
    wcf={}; ntw=30
    try:
        fn=_vectorizer_lda.get_feature_names_out()
        for idx,tw in enumerate(_lda_model.components_):
            vi=[i for i in tw.argsort()[-ntw:] if i<len(fn)];twd={fn[i]:tw[i] for i in vi}
            if not twd: continue
            wc=WordCloud(width=600,height=300,background_color='white').generate_from_frequencies(twd)
            fig,ax=plt.subplots(figsize=(8,4));ax.imshow(wc,interpolation='bilinear');ax.axis('off');tn=topic_labels.get(idx,f"Topic {idx}");ax.set_title(f"Word Cloud: {tn}");plt.tight_layout();plt.close(fig);wcf[idx]=fig
        return wcf
    except Exception: return None
word_cloud_figs = generate_word_clouds(lda_model, vectorizer_lda)


# ----------------------------
# SIDEBAR NAVIGATION
# ----------------------------
st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Select Analysis Section",
    ["🚀 Virality Predictor", "🧬 Similar Posts", "📈 Topic Trend Predictor", "📌 Key Stats & EDA"],
    key="navigation_select", help="Choose analysis."
)
# Sidebar warnings
if model is None or tfidf is None: st.sidebar.warning("Virality model N/A.") # Use 'model', 'tfidf'
if lda_model is None or vectorizer_lda is None: st.sidebar.warning("LDA components N/A.")
if rf_controversy_model is None or rf_controversy_tfidf is None or rf_controversy_ohe is None: st.sidebar.warning("Controversy model N/A.")


# ----------------------------
# MAIN CONTENT AREA
# ----------------------------

if data is None:
    st.error("Data loading failed. Cannot display analytics."); st.stop()

# ----------------------------
# 🚀 VIRALITY PREDICTOR (+ Controversy)
# ----------------------------
if section == "🚀 Virality Predictor":
    st.header("🚀 Virality & Controversy Predictor")
    st.markdown("Enter a post title to predict its likelihood of going viral and generating controversy.")

    user_input_title = st.text_input("Enter post title:", key="predictor_input")

    if user_input_title:
        # Check readiness using the correct variable names 'model' and 'tfidf'
        virality_ready = model and tfidf and is_vectorizer_fitted(tfidf)
        controversy_ready = all([
            rf_controversy_model, rf_controversy_tfidf, rf_controversy_ohe,
            is_vectorizer_fitted(rf_controversy_tfidf),
            lda_model, vectorizer_lda, is_vectorizer_fitted(vectorizer_lda),
            sia_global
        ])

        pred_col1, pred_col2 = st.columns(2)

        # --- Virality Prediction---
        with pred_col1:
            st.subheader("📈 Virality")
            if virality_ready:
                try:
                    cleaned_title_v = user_input_title.lower()
                    # Use 'tfidf' to transform
                    X_text_input = tfidf.transform([cleaned_title_v])

                    # Use globally calculated MEANS
                    sentiment = avg_sentiment_global
                    num_comments = avg_num_comments_global
                    subscribers = avg_subscribers_global # Use MEAN

                    # Create numeric features as csr_matrix
                    X_numeric = csr_matrix([[sentiment, num_comments, subscribers]])

                    # Combine features and wrap text vector in csr_matrix again
                    X_input = hstack([csr_matrix(X_text_input), X_numeric])

                    # Predict probability using 'model'
                    prob_v = model.predict_proba(X_input)[0][1]

                    # Display
                    fig_v = show_virality_gauge(prob_v)
                    if fig_v: st.pyplot(fig_v)

                except Exception as e: st.error(f"Virality prediction error: {e}")
            else:
                 if not model: st.info("Virality model not loaded.")
                 elif not tfidf: st.info("Virality TFIDF not loaded.")
                 elif not is_vectorizer_fitted(tfidf): st.info("Virality TFIDF not fitted.")
                 else: st.info("Virality components unavailable.")


        # --- Controversy Prediction ---
        with pred_col2:
            st.subheader("🔥 Controversy")
            if controversy_ready:
                 prob_c = predict_controversy_probability(
                     input_title=user_input_title, model_controv=rf_controversy_model,
                     tfidf_vec_controv=rf_controversy_tfidf, ohe_topic_enc_controv=rf_controversy_ohe,
                     lda_model_map=lda_model, vectorizer_lda_map=vectorizer_lda,
                     sentiment_analyzer=sia_global,
                     # Use MEDIAN for controversy as trained
                     default_sub_subscribers=median_sub_subscribers_global
                 )
                 if prob_c is not None:
                     st.metric(label="Predicted Likelihood", value=f"{prob_c*100:.1f}%")
                     st.progress(prob_c)
                     if prob_c > 0.65: st.warning("Signal suggests higher likelihood of significant debate.")
                     elif prob_c < 0.3: st.success("Signal suggests lower likelihood of significant debate.")
                     else: st.info("Signal does not strongly indicate high/low controversy.")

            else: st.info("Controversy model or dependencies not available/ready.")


# ----------------------------
# 🧬 SIMILAR POSTS SECTION
# ----------------------------
# (Uses 'tfidf')
elif section == "🧬 Similar Posts":
    st.header("🧬 Find Similar Posts"); st.markdown("Find textually similar posts based on TF-IDF.")
    if tfidf and is_vectorizer_fitted(tfidf): # Use 'tfidf'
        input_title_similar = st.text_input("Enter post title:", key="similarity_input")
        if input_title_similar:
            try:
                tfidf_matrix_full = tfidf.transform(data['title_cleaned']); input_vec = tfidf.transform([input_title_similar.lower()])
                similarity_scores = cosine_similarity(input_vec, tfidf_matrix_full).flatten(); num_similar = 5
                sorted_indices = np.argsort(similarity_scores)[::-1]; top_indices = []
                input_lower = input_title_similar.lower()
                for idx in sorted_indices:
                    if len(top_indices) >= num_similar: break
                    if idx < len(data):
                       is_self = data.iloc[idx]['title'].lower() == input_lower
                       if not is_self: top_indices.append(idx)
                if not top_indices: st.info("No similar posts found.")
                else:
                    st.subheader(f"Top {len(top_indices)} Similar Posts:")
                    similar_df = data.iloc[top_indices][['title', 'score', 'subreddit', 'id']].copy()
                    similar_df['similarity'] = similarity_scores[top_indices]
                    st.dataframe(similar_df[['title', 'similarity', 'score', 'subreddit', 'id']].reset_index(drop=True),
                        use_container_width=True, column_config={"similarity": st.column_config.NumberColumn(format="%.3f")})
            except Exception as e: st.error(f"Similarity search error: {e}")
    else: st.warning("Similarity model component (TF-IDF) not available/fitted.")

# ----------------------------
# 📈 TOPIC TREND PREDICTOR
# ----------------------------
elif section == "📈 Topic Trend Predictor":
    st.header("📈 Topic Trend Predictor"); st.markdown("Identify topic, trend, best posting days, and view overall trends.")
    if lda_model and vectorizer_lda and weekly_topic_counts is not None:
        st.subheader("Analyze Your Post Title")
        user_input_topic_title = st.text_input("Enter post title:", key="topic_input", value="word document saving issue")
        if user_input_topic_title:
            dominant_topic = map_title_to_topic(user_input_topic_title, lda_model, vectorizer_lda) # Correct call
            if dominant_topic is not None:
                topic_name = topic_labels.get(dominant_topic, f"Topic {dominant_topic}")
                st.info(f"**Identified Dominant Topic:** {dominant_topic} - {topic_name}")
                is_emerging = is_emerging_topic(dominant_topic, weekly_topic_counts); st.metric("Topic Trend Status", "Emerging" if is_emerging else "Stable/Declining")
                best_days = get_best_post_days(dominant_topic, data)
                if best_days and best_days != ["Data unavailable"]: st.success(f"🗓️ **Best Day(s) to Post:** {', '.join(best_days)}")
                else: st.info("Could not determine specific best posting days.")
            else: st.error("Could not determine the topic.")
        st.divider()
        st.subheader("Overall Weekly Topic Trends and Forecasts")
        if weekly_topic_counts.empty: st.warning("Weekly topic counts data is empty.")
        else:
            valid_topics = [t for t in weekly_topic_counts.columns if t in topic_labels]
            if not valid_topics: st.warning("No valid topics found.")
            else:
                cols = st.columns(2); col_idx = 0
                plot_weekly_counts = weekly_topic_counts.copy()
                if isinstance(plot_weekly_counts.index, pd.PeriodIndex): plot_weekly_counts.index = plot_weekly_counts.index.to_timestamp()
                else: plot_weekly_counts.index = plot_weekly_counts.index.astype(str)
                for topic_idx in sorted(valid_topics):
                    series = plot_weekly_counts[topic_idx]; topic_name = topic_labels[topic_idx]
                    if series.sum() < 10: continue
                    try:
                        model_trend = ExponentialSmoothing(series.astype(float).values, trend='add', seasonal=None, damped_trend=True).fit(); fp = 10; fv = model_trend.forecast(fp)
                        lai = series.index[-1]
                        if pd.api.types.is_datetime64_any_dtype(series.index): fi = pd.date_range(start=lai + pd.Timedelta(weeks=1), periods=fp, freq='W')
                        else: fi = [f"{lai}_F{i+1}" for i in range(fp)]
                        fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(series.index, series.values, label="Actual", marker='o', markersize=4); ax.plot(fi, fv, label="Forecast", linestyle='--', marker='x', markersize=5, color='red')
                        ax.set_title(f"{topic_idx}: {topic_name}"); ax.set_ylabel("Post Count"); ax.legend(); ax.grid(True, alpha=0.5); ax.tick_params(axis='x', rotation=30, labelsize=8)
                        nt = min(8, len(series.index)//2 if len(series.index)>0 else 1); ax.xaxis.set_major_locator(plt.MaxNLocator(nt if nt>0 else 1)); plt.tight_layout(); plt.close(fig)
                        with cols[col_idx % 2]: st.pyplot(fig)
                        col_idx += 1
                    except Exception as pe: st.warning(f"Plot failed: Topic {topic_idx}: {pe}")
    else: st.warning("Topic modeling components unavailable.")


# ----------------------------
# 📌 KEY STATS & EDA SECTION
# ----------------------------
elif section == "📌 Key Stats & EDA":
    st.header("📌 Key Statistics & Exploratory Data Analysis")
    st.subheader("Overall Dataset Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("🔢 Total Posts", f"{len(data):,}")
    col2.metric("📈 Avg Score", f"{data['score'].mean():.2f}" if 'score' in data else "N/A")
    col3.metric("💬 Avg Comments", f"{data['num_comments'].mean():.2f}" if 'num_comments' in data else "N/A")
    st.divider()
    st.subheader("Sentiment Analysis Insights")
    st.markdown("**Top 5 Positive Titles**")
    if 'sentiment_compound' in data and 'title' in data:
         st.dataframe(data.nlargest(5, 'sentiment_compound')[['title', 'sentiment_compound']].reset_index(drop=True), height=200, use_container_width=True, column_config={"sentiment_compound": st.column_config.NumberColumn(format="%.3f")})
         st.markdown("**Top 5 Negative Titles**")
         st.dataframe(data.nsmallest(5, 'sentiment_compound')[['title', 'sentiment_compound']].reset_index(drop=True), height=200, use_container_width=True, column_config={"sentiment_compound": st.column_config.NumberColumn(format="%.3f")})
    else: st.warning("Cannot display top/bottom sentiment posts.")
    with st.expander("Show Sentiment vs. Score/Time Plots", expanded=False):
         senti_plot_col1, senti_plot_col2 = st.columns(2)
         with senti_plot_col1:
              fig = plot_score_by_sentiment(data)
              if fig: st.pyplot(fig) 
              else: st.info("Plot unavailable.")
         with senti_plot_col2:
             fig = plot_sentiment_over_time(data)
             if fig: st.pyplot(fig) 
             else: st.info("Plot unavailable.")
    st.divider()
    st.subheader("Post Distributions")
    with st.expander("Show Distribution Plots", expanded=False):
        eda_col1, eda_col2 = st.columns(2)
        with eda_col1:
            fig = plot_subreddit_dist(data)
            if fig: st.pyplot(fig) 
            else: st.info("Plot unavailable.")
            fig = plot_posts_by_month(data)
            if fig: st.pyplot(fig) 
            else: st.info("Plot unavailable.")
            fig = plot_posts_by_day_of_week(data)
            if fig: st.pyplot(fig) 
            else: st.info("Plot unavailable.")
        with eda_col2:
            fig = plot_posts_by_year(data)
            if fig: st.pyplot(fig)  
            else: st.info("Plot unavailable.")
            fig = plot_posts_by_day_of_month(data)
            if fig: st.pyplot(fig)  
            else: st.info("Plot unavailable.")
            fig = plot_sentiment_dist(data)
            if fig: st.pyplot(fig) 
            else: st.info("Plot unavailable.")
    st.divider()
    st.subheader("LDA Topic Word Clouds")
    with st.expander("Show Topic Word Clouds", expanded=False):
        if word_cloud_figs:
            wc_cols = st.columns(2); wc_col_idx = 0
            for topic_idx in sorted(word_cloud_figs.keys()):
                 fig = word_cloud_figs[topic_idx]
                 with wc_cols[wc_col_idx % 2]:
                      if fig: st.pyplot(fig)
                 wc_col_idx += 1
        else: st.warning("Word clouds unavailable.")

