import pandas as pd
import numpy as np
import recommender as r

# Load all necessay datas
df = pd.read_csv('df_clean.csv').iloc[:,1:]
df_content = pd.read_csv('df_content_clean.csv').iloc[:,1:]
tfidf_matrix = np.load('tfidf_matrix.npy')

# Load rec object
from sklearn.externals import joblib
rec = joblib.load("rec_sys.dat")


df_user_similarity = rec.user_item_df.reset_index().replace(np.nan, 0)
def prep_get_similar_user():
    user_content = np.array(df_user_similarity.iloc[:,1:])
    user_content_transpose = np.transpose(user_content)
    dot_prod = user_content.dot(user_content_transpose)
    return dot_prod

dot_product_matrix_user = prep_get_similar_user()


def display_recommendations(rec_ids, rec_names, message, rec_ids_users,
                            rec_user_articles):
    
    if type(rec_ids) == type(None):
        print(f"{message}")
    
    else:
        dict_id_name = dict(zip(rec_ids, rec_names))
        
        if type(rec_ids_users) != type(None):
            print('Matrix Factorisation SVD:')
            print(f"\t{message}")
            
            for key, val  in dict_id_name.items():
                print(f"\t- ID items: {key}")
                print(f"\tName: {val}\n")

            print('CF User Based:')
            print('\tUser that are similar to you also seen:\n')
            for i in rec_user_articles[:5]:
                print(f"\t- {i}")
        else:
            print(f"\t{message}")
            dict_id_name = dict(zip(rec_ids, rec_names))
            for key, val  in dict_id_name.items():
                print(f"\t- ID items: {key}")
                print(f"\tName: {val}\n")


rec_ids, rec_names, message, rec_ids_users, rec_user_articles = (
    rec.make_recommendations(_id=3,
                             dot_prod_user= dot_product_matrix_user,
                             tfidf_matrix=tfidf_matrix,
                             _id_type='user',
                             rec_num=5)
    )

display_recommendations(rec_ids, rec_names, message, rec_ids_users,
                        rec_user_articles)