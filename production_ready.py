import pandas as pd
import numpy as np
import recommender as r
from sklearn.externals import joblib
from prep_data_display import display_recommendations

# Load all necessay datas
df = pd.read_csv('saved_data/df_clean.csv').iloc[:,1:]
df_content = pd.read_csv('saved_data/df_content_clean.csv').iloc[:,1:]
tfidf_matrix = np.load('saved_data/tfidf_matrix.npy')
dot_product_matrix_user = np.load('saved_data/dot_product_matrix_user.npy')

# Load rec object
rec = joblib.load("saved_data/rec_sys.dat")


# make recommendations
rec_ids, rec_names, message, rec_ids_users, rec_user_articles = (
    rec.make_recommendations(_id=3,
                             dot_prod_user= dot_product_matrix_user,
                             tfidf_matrix=tfidf_matrix,
                             _id_type='user',
                             rec_num=5)
    )

# display recommendations
display_recommendations(rec_ids, rec_names, message, rec_ids_users,
                        rec_user_articles)