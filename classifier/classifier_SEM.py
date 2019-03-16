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
training3D_matrix = joblib.load('../data_processing/matrix2D_SEM2018_train_fasttext_flatten')
#Load Training Matrix
test3D_matrix = joblib.load('../data_processing/matrix2D_SEM2018_test_fasttext_flatten')

dataset = pd.read_csv( '../data_processing/semeval2018_train.csv',delimiter='\t')
training_classes = dataset['label']


dataset = pd.read_csv( '../data_processing/semeval2018_test.csv',delimiter='\t')
test_classes = dataset['label']


le =  ce.OneHotEncoder(return_df=False, impute_missing=False, handle_unknown="ignore")

#dataset_t = pd.read_csv( 'dev.txt',delimiter='\t')
#test_classes = dataset_t['label']

#training_classes = le.fit_transform(training_classes.tolist())

#Gold Encode
#test_classes = le.transform(test_classes.tolist())
print(training_classes)
#print(test_classes)
#print(le.category_mapping)

#X_train, X_test, Y_train, Y_test = train_test_split(training3D_matrix, training_classes, test_size = 0.4, stratify = training_classes)



from keras.utils import plot_model
#plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
gnb =  RandomForestClassifier(n_estimators=500)#SVC(kernel='rbf',gamma=0.2, C=1.5)# RandomForestClassifier(n_estimators=500)# SVC(kernel='rbf',gamma=0.2, C=1.5) #GaussianNB()


gnb.fit(training3D_matrix,training_classes)

y_pred = gnb.predict (test3D_matrix)
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

y_TT = le.fit_transform(test_classes.tolist())
y_predTT = le.transform(y_pred.tolist())

roc = roc_auc_score ( y_TT , y_predTT )
roc_val = roc_auc_score ( y_TT , y_predTT )


print ( '\rroc-auc: %s - roc-auc_val: %s' % (str ( round ( roc , 4 ) ) , str ( round ( roc_val , 4 ) )) ,
                    end=100 * ' ' + '\n' )


print(classification_report(y_pred , test_classes ))

acc = accuracy_score ( y_pred , test_classes )
print ( " Acc:" , acc )

pre = precision_score ( y_pred , test_classes , average='micro' )
print ( " Pre:" , pre )

re = recall_score ( y_pred , test_classes , average='micro')
print ( " Rec:" , re )

f1 = f1_score ( y_pred , test_classes , average='micro' )
print ( " F1:" , f1 )


#joblib.dump(m, "trained_model.dump")
