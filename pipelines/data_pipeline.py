from prefect import task, Flow
from src.data.load_data import load_metadata, load_images
from src.data.preprocess import preprocess_data
import os
import numpy as np

@task
def load_and_preprocess_data(metadat_path,image_folder):
    metadata=load_metadata(metadata_path)
    images,labels=load_images(metadata,image_folder)
    return preprocess_data(images,labels)
@task
def save_data(X_train,X_val,X_test, y_train, y_val,y_test):
    np.save('data/processed/X_train.npy', X_train)
    np.save('data/processed/X_val.npy', X_val)
    np.save('data/processed/X_test.npy', X_test)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/y_val.npy', y_val)
    np.save('data/processed/y_test.npy', y_test)

with Flow("Data Pipeline") as flow:
    metadata_path="data/raw/ham1000_metadata.csv"
    image_folder="data/raw/ISIC-images"
    X_train,X_val,X_test,y_train,y_val,y_test,label_encoder=load_and_preprocess_data(metadata_path,image_folder)
    save_data(X_train,X_val,X_test, y_train, y_val,y_test)

if __name__ == "__main__":
    flow.run()
    flow.visualize()