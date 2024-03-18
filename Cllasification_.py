import os
import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils, plot_model
from keras import layers, activations, optimizers
from keras.models import Model 
from keras.layers.wrappers import TimeDistributed

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

data_dir = "data/"
img_height , img_width = 32, 32

classes = glob.glob("data/*")
ch_dir=[]
all_v=[]
for ch in classes:
    ch_dir.append ('./' + ch + '/videos/')
for i in range(13):
    all_v.append (os.listdir(ch_dir[i]))
    
i=-1
X = []
Y=[]
for ch in ch_dir:
    i=i+1
    for d in all_v[i]:
        img_array = []
        j=0
        for filename in glob.glob(ch+d+"*/*/*.jpg"):
            while(j<100):
                img = cv2.imread(filename)
                img = cv2.resize(img, (img_height, img_width))
                img_array.append(img)
                if (j==99):
                    X.append(img_array)
                    img_array = []
                    Y.append(i)
                j=j+1

X = np.asarray(X)
Y = np.asarray(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, shuffle=True, random_state=0)
train_labels_vectors = np_utils.to_categorical(y_train)
test_labels_vectors = np_utils.to_categorical(y_test)
X=[]

video = layers.Input(shape=(100, img_height, img_width, 3))
input_layer = layers.Input(shape=(img_height, img_width,3))
conv1 = layers.Conv2D(96,(3,3),activation=activations.relu,padding='same')(input_layer)
batch1= layers.BatchNormalization()(conv1)
conv2 = layers.Conv2D(96,(3,3),activation=activations.relu,padding='same')(input_layer)
batch2= layers.BatchNormalization()(conv2)
add1 = layers.Add()([batch1, batch2])
drop1 = layers.Dropout(0.5)(add1)
conv3 = layers.Conv2D(96,(3,3),activation=activations.relu,padding='same',strides=2)(drop1)
batch3= layers.BatchNormalization()(conv3)
drop2 = layers.Dropout(0.5)(batch3)
conv4 = layers.Conv2D(192,(3,3),activation=activations.relu,padding='same')(drop2)
batch4= layers.BatchNormalization()(conv4)
conv5 = layers.Conv2D(192,(3,3),activation=activations.relu,padding='same')(drop2)
batch5= layers.BatchNormalization()(conv5)
add2 = layers.Add()([batch4, batch5])
drop3 = layers.Dropout(0.5)(add2)
conv6 = layers.Conv2D(192,(3,3),activation=activations.relu,padding='same',strides=2)(drop3)
batch6= layers.BatchNormalization()(conv6)
drop4 = layers.Dropout(0.5)(batch6)
conv7 = layers.Conv2D(192,(3,3),activation=activations.relu,padding='same')(drop4)
batch7= layers.BatchNormalization()(conv7)
conv8 = layers.Conv2D(192,(1,1),activation=activations.relu,padding='same')(batch7)
batch8= layers.BatchNormalization()(conv8)
conv9 = layers.Conv2D(13,(1,1),activation=activations.relu,padding='same')(batch8)
flat1 = layers.Flatten()(conv9)
cnn = Model(input_layer, flat1)

encoded_frames = TimeDistributed(cnn)(video)
encoded_sequence = layers.LSTM(256)(encoded_frames)
hidden_layer = layers.Dense(output_dim=1024, activation="relu")(encoded_sequence)
outputs = layers.Dense(output_dim=13, activation="softmax")(hidden_layer)
model = Model([video], outputs)
model.compile(loss="categorical_crossentropy",
              optimizer=optimizers.Adam(),
              metrics=["accuracy"]) 
model.summary()
plot_model(model,to_file='Classifier.png',show_shapes=True,show_layer_names=True)
plot_model(cnn,to_file='CNN.png',show_shapes=True,show_layer_names=True)

history = model.fit(x = X_train, y = train_labels_vectors, epochs=30, batch_size = 16 , shuffle=True, validation_split=0.2)

test_loss, test_acc = model.evaluate(X_test,test_labels_vectors)
test_labels_predicted = model.predict(X_test)
test_labels_predicted = np.argmax(test_labels_predicted,axis=1)


