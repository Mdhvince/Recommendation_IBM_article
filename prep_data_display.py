import pandas as pd
import numpy as np


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