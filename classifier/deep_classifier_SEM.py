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
from keras.layers import Add
from keras.layers import Flatten
from keras.callbacks import ModelCheckpoint
from keras_self_attention import SeqSelfAttention
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
from sklearn.model_selection import train_test_split
os.environ['KMP_DUPLICATE_LIB_OK']='True'


#Link for downloading contents: https://mega.nz/#F!0kYGRYwK!tGEZ8c5pPdfJe8OwpGrDyg


#Load Training Matrix
training3D_matrix = joblib.load('../data_processing/matrix3D_SEM2018_train_fasttext')
#Load Training Matrix
test3D_matrix = joblib.load('../data_processing/matrix3D_SEM2018_test_fasttext')

dataset = pd.read_csv( '../data_processing/semeval2018_train.csv',delimiter='\t')
t1 = dataset['label']


training_classes = t1


dataset = pd.read_csv( '../data_processing/semeval2018_test.csv',delimiter='\t')
test_classes = dataset['label']

le =  ce.OneHotEncoder(return_df=False, impute_missing=False, handle_unknown="ignore")

#dataset_t = pd.read_csv( 'dev.txt',delimiter='\t')
#test_classes = dataset_t['label']

training_classes = le.fit_transform(training_classes.tolist())
test_classes = le.transform(test_classes.tolist())

#Gold Encode
#test_classes = le.transform(test_classes.tolist())
print(training_classes)
#print(test_classes)
print(le.category_mapping)

#X_train, X_test, Y_train, Y_test = train_test_split(training3D_matrix, training_classes, test_size = 0.4, stratify = training_classes)


#Model
input = Input(shape=(80,300))
#model = Sequential() (input)
bi = Bidirectional(LSTM(200, activation ='tanh', return_sequences = True, dropout=0.3, input_shape=(80,300))) (input)
aa = SeqSelfAttention(attention_activation='tanh') (bi)
aa = Conv1D(400,5, activation ='relu' ) (aa)
aa = MaxPool1D(2) (aa)
aa = Dropout(0.2) (aa)

added = keras.layers.Concatenate(axis=1)([aa,bi])

ff = GlobalMaxPool1D() (added)
ff = Dense(100)(ff)
ff = Dropout(0.3) (ff)
ff =Dense(4, activation='softmax') (ff)


#print(X_train.shape)
m = keras.models.Model(inputs=[input], outputs=[ff])

m.summary(line_length=100)
from keras.utils import plot_model
#plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')


#Custom callback
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
class MyCallBack(EarlyStopping):
        def __init__ ( self , training_data , validation_data, verbose ):
            self.x = training_data
            self.y_val = validation_data

        def on_train_begin ( self , logs={} ):
            return

        def on_train_end ( self , logs={} ):
            return

        def on_epoch_begin ( self , epoch , logs={} ):
            return

        def on_epoch_end ( self , epoch , logs={} ):
            y_pred = self.model.predict ( self.x )

            roc = roc_auc_score ( self.y_val , y_pred )

            roc_val = roc_auc_score ( self.y_val , y_pred )
            print ( '\rroc-auc: %s - roc-auc_val: %s' % (str ( round ( roc , 4 ) ) , str ( round ( roc_val , 4 ) )) ,
                    end=100 * ' ' + '\n' )

            test = [ 0 ] * 1580
            i = 0
            for cl in y_pred:
                test[ i ] = str ( np.argmax ( cl ) )
                i += 1

            test_lab = [ 0 ] * 1580
            i = 0
            for cl in self.y_val:
                test_lab[ i ] = str ( np.argmax ( cl ) )
                i += 1

            print(classification_report(test , test_lab ))

            acc = accuracy_score ( test , test_lab )
            print ( " Acc:" , acc )

            pre = precision_score ( test , test_lab , average='micro')
            print ( " Pre:" , pre )

            re = recall_score ( test , test_lab , average='micro')
            print ( " Rec:" , re )

            f1 = f1_score ( test , test_lab , average='micro')
            print ( " F1:" , f1 )

            return

        def on_batch_begin ( self , batch , logs={} ):
            return

        def on_batch_end ( self , batch , logs={} ):
            return

filepath="weights.{epoch:02d}-{val_loss:.2f}_fasttext_semeval2018.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacks_list = [
checkpoint,
    MyCallBack(test3D_matrix,test_classes,verbose=1)
]



m.compile ( loss='categorical_crossentropy' , optimizer='adam' , metrics=['accuracy'] )
history = m.fit(training3D_matrix,training_classes,64,100,
                     callbacks=callbacks_list,
                     validation_data = [test3D_matrix,test_classes],
                      verbose=1)

joblib.dump(m, "trained_model.dump")
