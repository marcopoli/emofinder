import joblib as joblib
import pandas as pd
import keras
import numpy as np

import pickle
from numpy.core import multiarray
from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import Dropout
from keras.layers import MaxPool1D
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import GlobalMaxPool1D
from keras.utils import to_categorical
import category_encoders as ce
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import keras.backend as K

import tensorflow as tf

#Load Training Matrix
#Link for downloading contents: https://mega.nz/#F!0kYGRYwK!tGEZ8c5pPdfJe8OwpGrDyg

final3D_matrix = joblib.load('matrix3D_test_final_6')
print(final3D_matrix.shape)

#Load Train For Encoding of classes
dataset_t = pd.read_csv( 'train.tsv',delimiter='\t')
training_classes = dataset_t['label']
training_classes_A = dataset_t['label']

#Load test set for columns
dataset = pd.read_csv( 'testwithoutlabels.txt',delimiter='\t')

#Vector Encoding
le =  ce.OneHotEncoder(return_df=False, impute_missing=False, handle_unknown="ignore")
training_classes = le.fit_transform(training_classes.tolist())


#Load Models
model_1 = joblib.load('trained_model_6_0.7714')
model_2 = joblib.load('trained_model_6_0.8078')
model_3 = joblib.load('trained_model_6_0.78163')

from keras.utils import plot_model
model_1.summary(line_length=100)
plot_model(model_1, show_shapes=True, show_layer_names=True, to_file='final_model.png')

#Make predictions
result_1 = model_1.predict(final3D_matrix)
result_2 = model_2.predict(final3D_matrix)
result_3 = model_3.predict(final3D_matrix)



#Majority vote
test = [0]* 5509
i= 0
for cl in result_1:
    a = [ 0 ] * 4
    index_1 = np.argmax ( result_1[ i ] )
    index_2 = np.argmax ( result_2[ i ] )
    index_3 = np.argmax ( result_3[ i ] )

    a[ index_1 ] = a[ index_1 ] + 1
    a[ index_2 ] = a[ index_2 ] + 1
    a[ index_3 ] = a[ index_3 ] + 1

    test[ i ] = str ( np.argmax ( a ) )
    i +=1

#Write result file
res_file = open('test.txt', 'w')

test_ids = dataset['id']
test_turn1 = dataset['turn1']
test_turn2 = dataset['turn2']
test_turn3 = dataset['turn3']
map_label = ['others','angry','sad','happy']

res_file.write('id\tturn1\tturn2\tturn3\tlabel\n')
index = 0
for t in test:
    line = str(test_ids[index])+'\t'+str(test_turn1[index])+'\t'+str(test_turn2[index])+'\t'+str(test_turn3[index])+'\t'+str(map_label[int(t)])+'\n'
    res_file.write(line)
    index +=1

res_file.close()

print("File creation finished")