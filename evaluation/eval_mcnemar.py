from statsmodels.sandbox.stats.runs import mcnemar

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


#Load Training Matrix
test3D_matrix = joblib.load('../data_processing/matrix3D_EMOCONT_test_glove')
test3D_matrix1 = joblib.load('../data_processing/matrix3D_EMOCONT_test_fasttext')

dataset = pd.read_csv( '../data_processing/train_emocontext.txt',delimiter='\t')
t1 = dataset['label']


training_classes = t1


dataset = pd.read_csv( '../data_processing/test_emocontext.txt',delimiter='\t')
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


model1 = keras.models.load_model("../classifier/EMOC_weights.03-0.40_google.hdf5", custom_objects={'SeqSelfAttention':SeqSelfAttention})
model2 = keras.models.load_model("../classifier/EMOC_weights.05-0.29_fasttext.hdf5", custom_objects={'SeqSelfAttention':SeqSelfAttention})


res1 = model1.predict(test3D_matrix)
res2 = model2.predict(test3D_matrix1)

test1 = [ 0 ] * 5509
i = 0
for cl in res1:
    if  np.argmax ( cl ) == np.argmax(test_classes[i]):
        test1[ i ] = 1
    else:
        test1[ i ] = 0
    i += 1

test2 = [ 0 ] * 5509
i = 0
for cl in res2:
    if  np.argmax ( cl ) == np.argmax(test_classes[i]):
        test2[ i ] = 1
    else:
        test2[ i ] = 0
    i += 1

stts, pvalue = mcnemar(test1,test2)
print(stts,' ',pvalue)