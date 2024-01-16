import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from typing import List



def load_data(file_path,client_id):
    df = pd.read_csv(file_path)
    sex_map = {'M' : 0, 'F' : 1}
    cancer_map = {'YES':1,'NO':0}
    df['GENDER'] = df['GENDER'].apply(lambda X: sex_map[X])
    df['LUNG_CANCER'] = df['LUNG_CANCER'].apply(lambda X: cancer_map[X])

    feature_map = [
                        'GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
                        'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING',
                        'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
                        'SWALLOWING DIFFICULTY', 'CHEST PAIN'
                    ]
    label_map = ['LUNG_CANCER']

    

    features = df.iloc[:, :-1]
    labels = df.iloc[:, -1]
    
    X_train, X_test,Y_train,Y_test = train_test_split(features,labels,test_size=0.2,random_state=client_id)

    return X_train,X_test,Y_train,Y_test


def get_params(model: RandomForestClassifier) -> List[np.ndarray]:
    params = [
        model.n_estimators,
        model.max_depth,
        model.min_samples_split,
        model.min_samples_leaf,
    ]
    return params


# Set the parameters in the RandomForestClassifier
def set_params(model: RandomForestClassifier, params: List[np.ndarray]) -> RandomForestClassifier:
    model.n_estimators = int(params[0])
    model.max_depth = int(params[1])
    model.min_samples_split = int(params[2])
    model.min_samples_leaf = int(params[3])
    return model