import pandas as pd
import numpy as np
import recommender as r

# Nlp
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer

# Prepare data
from prepare_data import clean_pipeline

df = pd.read_csv('user-item-interactions.csv')
df_content = pd.read_csv('articles_community.csv')

# Cleanning Pipeline
df, df_content = clean_pipeline(df_reviews=df, df_items=df_content)