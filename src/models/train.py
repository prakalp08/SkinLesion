from tensorflow.keras.callbacks import EarlyStopping
from .model import build_model

def train_model(X_train, y_train, X_val, y_val, input_shape,num_classes,epochs=50):
    model=build_model(input_shape,num_classes)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stopping=EarlyStopping(monitor='val_loss', patience=3, restore_best_weight=True)
    history=model.fit(X_train,y_train,epochs=epochs,validation_data=(X_val,y_val),callbacks=[early_stopping])
    return model, history