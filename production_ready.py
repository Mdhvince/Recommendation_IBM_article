import pandas as pd
import numpy as np
import recommender as r

# Nlp
from sklearn.feature_extraction.text import TfidfVectorizer

# Prepare data
from prepare_data import clean_pipeline, tokenize, create_user_item

df = pd.read_csv('user-item-interactions.csv')
df_content = pd.read_csv('articles_community.csv')

# Cleanning Pipeline
df, df_content = clean_pipeline(df_reviews=df, df_items=df_content)

# Prepare tfidf matrix
tfidf = TfidfVectorizer(tokenizer=tokenize, ngram_range=(1, 2))
df_content['doc_description'] = df_content['doc_description'].fillna('')
tfidf_matrix = tfidf.fit_transform(df_content['doc_description'])

# Create User-Item df
user_item_df = create_user_item(df=df)