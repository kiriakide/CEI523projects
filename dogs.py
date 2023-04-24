#1.Κατεβαζω τις απαραιτητες βιβλιοθηκες
import tensorflow as tf
print(tf.__version__)

!pip install livelossplot

import numpy as np
import pandas as pd #
import matplotlib.pyplot as plt
from tensorflow import keras

import cv2
import PIL
import os
from IPython.display import Image, display

import plotly.graph_objs as go
import plotly.graph_objects as go
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.applications import xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten,BatchNormalization,Activation
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import gc
import skimage.io
from livelossplot import PlotLossesKeras

#2.Συνδεω τα δεδομενα μου
train_dir = "drive/MyDrive/Master/Semester1/Datascience/Assignment/dtata/train"
test_dir = "drive/MyDrive/Master/Semester1/Datascience/Assignment/dtata/test"
train_labels = pd.read_csv('drive/MyDrive/Master/Semester1/Datascience/Assignment/dtata/labels.csv')
submission=pd.read_csv("drive/MyDrive/Master/Semester1/Datascience/Assignment/dtata/sample_submission.csv")

#3.για να διαβαζει καλα τις φωτογραφιες
def jpg(id):
    return id+".jpg"
train_labels["id"] = train_labels["id"].apply(jpg)
submission["id"] = submission["id"].apply(jpg)

#4.ΚΑΤΑΝΟΩΩ τα δεδομενα μου
train_size = len(os.listdir(train_dir))
test_size = len(os.listdir(test_dir))

print(train_size,test_size)
print(train_labels.shape)
print(submission.shape)

target, dog_breeds = pd.factorize(train_labels['breed'], sort = True)
train_labels['target'] = target

display(train_labels.head())
display(submission.head())

train_labels['breed'].value_counts()

plt.figure(figsize=(13, 6))
train_labels['breed'].value_counts().plot(kind='bar')
plt.show()


#5. Διαχωριζω σε value & train set
labels=[]
data=[]
for i in range(train_labels.shape[0]):
    data.append(train_dir + train_labels['id'].iloc[i]+'.jpg')
    labels.append(train_labels['target'].iloc[i])
df=pd.DataFrame(data)
df.columns=['images']
df['target']=labels
print(df.shape)
display(df.head())

del labels
del data

test_data=[]
for i in range(submission.shape[0]):
    test_data.append(test_dir + submission['id'].iloc[i]+'.jpg')
df_test=pd.DataFrame(test_data)
df_test.columns=['images']
print(df_test.shape)
display(df_test.head())

del test_data

X_train, X_val, y_train, y_val = train_test_split(df['images'],df['target'], stratify = df['target'], test_size=0.2, random_state=1234)

train=pd.DataFrame(X_train)
train.columns=['images']
train['target']=y_train

validation=pd.DataFrame(X_val)
validation.columns=['images']
validation['target']=y_val

print(train.shape)
display(train.head())
print(validation.shape)
display(validation.head())

del X_train, X_val, y_train, y_val

#6.variables που βοηθουν στο classification
N_EPOCHS = 50
BATCH_SIZE = 32
IMG_HEIGHT = 299
IMG_WIDTH = 299

#7.ΚΑΝΩ dataaumentation για τις φωτο στο TRAIN KAI VAL SET
train_datagen = ImageDataGenerator(rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,horizontal_flip=True, zoom_range=0.2)

val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_dataframe(
    train,
    x_col='images',
    y_col='target',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode='raw')

validation_generator = val_datagen.flow_from_dataframe(
    validation,
    x_col='images',
    y_col='target',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=False,
    batch_size=BATCH_SIZE,
    class_mode='raw')

#8.εφαρμογη μοντελου xception απο documenation KERAS
base_model = xception.Xception(weights='imagenet', include_top=False, input_shape=(299,299,3))
# display(base_model.summary())

# train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional Xception layers
base_model.trainable = False

inputs = Input(shape=(299, 299, 3))
x = xception.preprocess_input(inputs)

# The base model contains batchnorm layers. We want to keep them in inference mode
# when we unfreeze the base model for fine-tuning, so we make sure that the
# base_model is running in inference mode here by passing `training=False`.
x = base_model(x, training=False)

x = GlobalAveragePooling2D()(x)

#     adding extra dense layer
#     x = Dense(1024, activation='relu')(x)
#     x = Dropout(.7)(x)
#     x = Dense(512, activation='relu')(x)

x = Dropout(.5)(x)
outputs = Dense(120, activation='softmax')(x)
model = Model(inputs, outputs)

display(model.summary())

optimizer = Adam(learning_rate=0.001)
model.compile(loss="sparse_categorical_crossentropy", metrics=['accuracy'], optimizer=optimizer)

n_train_steps = train.shape[0]//BATCH_SIZE
n_val_steps=validation.shape[0]//BATCH_SIZE
print("Number of training and validation steps: {} and {}".format(n_train_steps, n_val_steps))

EarlyStop_callback = EarlyStopping(min_delta=0.001, patience=10, restore_best_weights=True)

#9. ΚΑΝΩ fit το montelo για να το ελεγξω
history= model.fit(
    train_generator,
    epochs=N_EPOCHS,
    validation_data=validation_generator,
    callbacks=[EarlyStop_callback],
    )

del train_generator, validation_generator

#10.εφαρμοζω το ιδιο me to TEST SET
test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_dataframe(
    df_test,
    x_col='images',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=False,
    batch_size=1,
    class_mode=None
)test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_dataframe(
    df_test,
    x_col='images',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=False,
    batch_size=1,
    class_mode=None
)


predictions = model.predict(
    test_generator,
    verbose=1
)

print(predictions.shape)
print(predictions)

#11. κανω vizulize LOSS KAI ACCURACY
loss = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, 'bo', label='loss_train')
plt.plot(epochs, loss_val, 'b', label='loss_val')
plt.title('value of the loss function')
plt.xlabel('epochs')
plt.ylabel('value of the loss function')
plt.legend()
plt.grid()
plt.show()

acc = history.history['acc']
acc_val = history.history['val_acc']
epochs = range(1, len(loss)+1)
plt.plot(epochs, acc, 'bo', label='accuracy_train')
plt.plot(epochs, acc_val, 'b', label='accuracy_val')
plt.title('accuracy')
plt.xlabel('epochs')
plt.ylabel('value of accuracy')
plt.legend()
plt.grid()
plt.show()