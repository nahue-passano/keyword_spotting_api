import json
import numpy as np
from sklearn.model_selection import train_test_split

def load_dataset(data_path):
    
    with open(data_path, 'r') as fp:
        data = json.load(fp)
    
    # extract inputs and targets
    X = np.array(data['MFCCs'])
    y = np.array(data['labels'])

    return X,y

def get_data_splits(data_path, test_size = 0.2, validation_size = 0.2):

    # load dataset
    X, y = load_dataset(data_path)

    # create train/validation/test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_test, y_test, test_size = validation_size)

    # convert inputs from 2d to 3d arrays
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test