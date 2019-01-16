import pandas as pd 
import numpy as np

import re
import nltk
from nltk.corpus import stopwords
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.tokenize import word_tokenize

def clean_pipeline(df_reviews, df_items):
    """
    Inputs:
    - df_reviews: dataframe with interaction of the user on an item
    - df_items: dataframe all unique item

    Outputs:
    - df_reviews clean
    - df_items clean
    """
    
    df = df_reviews.copy()
    df_content = df_items.copy()
    
    if 'Unnamed: 0' in df.columns:
        del df['Unnamed: 0']
        
    if 'Unnamed: 0' in df_content.columns:
        del df_content['Unnamed: 0']

    # make sure all articles from df are in df_content
    df = pd.merge(df, df_content[['article_id']], on='article_id', how='inner')

    df_content = df_content.drop(labels='doc_status', axis=1)
    df_content = df_content.drop_duplicates()

    #df = df.replace('no_email', np.nan)

    def email_mapper():
        coded_dict = dict()
        cter = 1
        email_encoded = []

        for val in df['email']:
            if val not in coded_dict:
                coded_dict[val] = cter
                cter+=1

            email_encoded.append(coded_dict[val])
        return email_encoded

    email_encoded = email_mapper()
    del df['email']
    df['user_id'] = email_encoded

    # Rename my article name field to be the same in both dfs
    df_content['title'] = df_content['doc_full_name']
    df_content = df_content.drop(labels='doc_full_name', axis=1)

    df.article_id = df.article_id.astype('int64')
    df_content.article_id = df_content.article_id.astype('int64')

    # number of times the user seen the article
    pre_data = dict(df.groupby(['user_id', 'article_id'])['article_id'].count())

    list_user = []
    list_article_id = []
    list_nb_interactions = []

    for key, val in pre_data.items():
        list_user.append(key[0])
        list_article_id.append(key[1])
        list_nb_interactions.append(val)

    zipped_list = list(zip(list_user, list_article_id, list_nb_interactions))
    interaction_user = pd.DataFrame(zipped_list, columns=['user_id','article_id','nb_interactions_user_article'])

    df = pd.merge(df, interaction_user, on=['user_id','article_id'])

    # The nb interactions was shown implicitly with the number of rows
    # Now I have information about the number of interactions by creating this column
    # I can drop duplicated rows
    df = df.drop_duplicates(keep='first')
    df = df.reset_index().drop(labels='index', axis=1)

    df['date'] = 0
    
    return df, df_content





def tokenize(text):
    """
    This function do the following steps:
    - Normalize, remove ponctuations
    - Tokenize the input (str)
    - Remove english stopwords
    - Lemmatize but keep the gramatical sense of the word
    - Clean White spaces

    Input:
    - text: a (str) value

    Output:
    - token of clean text
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words]
    words_clean = [word.strip() for word in words]
    return words_clean



def create_user_item():
    """
    Output:
    - dataframe of user by item
    """
    user_item = df[['user_id',
                    'article_id',
                    'nb_interactions_user_article']]
    
    user_item_df=(
        user_item.groupby(['user_id',
                           'article_id'])['nb_interactions_user_article'].max().unstack()
    )
    
    return user_item_df