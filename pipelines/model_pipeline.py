from prefect import task,flow, Parameter
from src.models.train import train_model
import numpy as np

@task
def train_model_task(X_train, y_train, X_val, y_val, input_shape, num_classes, epochs):
    model, history = train_model(X_train, y_train, X_val, y_val, input_shape, num_classes, epochs)
    return model, history

@task
def save_model_task(model,model_path):
    model.save(model_path)

with Flow("Model Pipeline") as flow:
    X_train=Parameter("X_train")
    y_train=Parameter("y_train")
    X_val=Parameter("X_val")
    y_val=Parameter("y_val")
    input_shape=Parameter("input_shape",default=(128,128,3))
    num_classes=Parameter("num_classes")
    epochs=Parameter("epochs",default=50)
    model_path=Parameter("model_path", default="models/skin_cancer_model.h5")

    model,history=train_model_task(X_train,y_train,X_val,y_val,input_shape,num_classes, epochs)
    save_model_task(model,model_path)
                                   
if __name__=="__main__":
    flow.run(parameters={
        "X_train": np.load("data/processed/X_train.npy"),
        "y_train": np.load("data/processed/y_train.npy"),
        "X_val": np.load("data/processed/X_val.npy"),
        "y_val": np.load("data/processed/y_val.npy"),
        "num_classes": 7  # Assuming 7 classes in the dataset
    })
    flow.visualize()