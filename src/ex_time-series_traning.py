from turtle import pd
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def compact_cnn(input_shape, nb_class):
    ip = keras.layers.Input(shape=input_shape)
    
    conv = keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu', padding='same')(ip)
    conv = keras.layers.BatchNormalization()(conv)
    
    conv = keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', padding='same')(conv)
    conv = keras.layers.BatchNormalization()(conv)
    
    flat = keras.layers.GlobalAveragePooling1D()(conv)
    
    out = keras.layers.Dense(32, activation='relu')(flat)
    out = keras.layers.Dropout(0.3)(out)
    out = keras.layers.Dense(nb_class, activation='softmax')(out)


    model = keras.models.Model([ip], [out])
    model.summary()
    
    return model

'''
Loading the dataset and initializing the model
'''
dataset2_detail_new = pd.read_csv('dataset2_detail_new.csv')
feture_dataset = np.load('feture_dataset.npy')

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
num_classes = len(dataset2_detail_new['forest_type'].value_counts())
model_init = compact_cnn(input_shape=feture_dataset.shape[1:], nb_class=num_classes)
model_init.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

'''
Loading the dataset and initializing the model
'''
X_train, y_train = feture_dataset, dataset2_detail_new['forest_type'].values

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

model_init.fit(X_train, y_train, epochs=500, batch_size=64, validation_split=0.2, callbacks=[early_stopping], class_weight=class_weights_dict, verbose=0)

'''
Predicting the training set
'''
y_pred = model_init.predict(X_train)
y_pred_classes = np.argmax(y_pred, axis=1)