from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

def preprocess_data(images,labels):
    label_encoder = LabelEncoder()
    labels_encoded=label_encoder.fit_transform(labels)
    X_train_full,X_test,y_train_full,y_test=train_test_split(images,labels_encoded,test_size=0.2,random_state=42)
    X_train,X_val,y_train,y_val=train_test_split(X_train_full,y_train_full,test_size=0.2,random_state=42)
    return X_train,X_val,X_test,y_train,y_val,y_test, label_encoder
