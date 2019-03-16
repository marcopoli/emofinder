import joblib as joblib
import pandas as pd
import keras

from keras.models import Sequential
from keras.layers import Conv1D , Embedding
from keras.layers import Dropout
from keras.layers import MaxPool1D
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import GlobalMaxPool1D
from keras.layers import Bidirectional
from keras.layers import Input
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import TimeDistributed
import functools
import category_encoders as ce
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import keras.backend as K
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


#Link for downloading contents: https://mega.nz/#F!0kYGRYwK!tGEZ8c5pPdfJe8OwpGrDyg


#Load Training Matrix
training3D_matrix = joblib.load('../data_processing/matrix3D_google_general_all_train')
#Load Training Matrix
test3D_matrix = joblib.load('../data_processing/matrix3D_google_general_all_dev')

#Load Dataset
dataset = pd.read_csv( 'train.txt',delimiter='\t')
training_classes = dataset['label']
le =  ce.OneHotEncoder(return_df=False, impute_missing=False, handle_unknown="ignore")

dataset_t = pd.read_csv( 'dev.txt',delimiter='\t')
test_classes = dataset_t['label']

training_classes = le.fit_transform(training_classes.tolist())

#Gold Encode
test_classes = le.transform(test_classes.tolist())
print(training_classes)
print(test_classes)
print(le.category_mapping)


#Model
model = Sequential()
model.add(Conv1D(200,3, activation ='relu', input_shape=(100,600)))
model.add(MaxPool1D(4))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(200, activation ='relu', return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(200))
model.add(Dropout(0.3))
model.add(Dense(4, activation='softmax'))

model.summary(line_length=100)
from keras.utils import plot_model
#plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')


#Custom callback
from keras.callbacks import EarlyStopping
class MyCallBack(EarlyStopping):
    def __init__(self, threshold, **kwargs):
        super(MyCallBack, self).__init__(**kwargs)
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):

        result = self.model.predict_classes(test3D_matrix)
        test = [0]* 2755
        i= 0
        for cl in result:
            test[i] = str(cl)
            i +=1

        test_lab = [0]* 2755
        i= 0
        for cl in test_classes:
            test_lab[ i ] = str(np.argmax ( cl ))
            i += 1

        acc = accuracy_score(test, test_lab)
        print(" Acc:", acc)

        pre = precision_score(test, test_lab, average='micro', labels=['1','2','3'])
        print(" Pre:", pre)

        re = recall_score(test, test_lab, average='micro', labels=['1','2','3'])
        print(" Rec:", re)

        f1 = f1_score(test, test_lab, average='micro', labels=['1','2','3'])
        print(" F1:", f1)

        if f1 >= self.threshold:
            self.stopped_epoch = epoch
            self.model.stop_training = True

callbacks_list = [
    MyCallBack(threshold=0.76,verbose=1)
]


model.compile ( loss='categorical_crossentropy' , optimizer='adam' , metrics=['accuracy'] )
history = model.fit(training3D_matrix,training_classes,64,100,
                      callbacks=callbacks_list,
                      validation_data=(test3D_matrix, test_classes) ,
                      verbose=1)

joblib.dump(model, "trained_model.dump")
