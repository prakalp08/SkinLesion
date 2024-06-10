import pandas as pd
import numpy as np
import os
import cv2
def load_metadata(file_path):
    return pd.read_csv(file_path)

def load_images(metadata,image_folder):
    images = []
    labels = [] 

    for index,row in metadata.iterrows():
        image_path=os.path.join(image_folder,row["isic_id"]+ ".JPG")
        if os.path.exists(image_path):
            image=cv2.imread(image_path)
            image=cv2.resize(image,(128,128))
            images.append(image)
            labels.append(row["diagnosis"])
    return np.array(images),np.array(labels)
