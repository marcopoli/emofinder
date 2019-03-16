import pandas as pd
import numpy as np
from nltk.tokenize import TweetTokenizer as TweetTokenizer
from preprocessor.preprocess import *
from gensim.models import Word2Vec
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

#Gensim Custom Word2Vec
dataset = pd.read_csv( 'senti140.csv',delimiter=',',header=None, encoding='latin-1')
dataset = dataset[dataset[0]==4]
tweets = dataset[5]
print(tweets)

# 0 = negative, 2 = neutral, 4 = positive
tw_ = tweets

tok = list()

tk = TweetTokenizer()
p = Preprocess()
text_processor = TextPreProcessor (
    # terms that will be normalized
    normalize=[ 'email' , 'percent' , 'money' , 'phone' ,
                'time' , 'url' , 'date' , 'number' ] ,
    fix_html=True ,  # fix HTML tokens
    segmenter="twitter" ,
    corrector="twitter" ,

    unpack_hashtags=True ,  # perform word segmentation on hashtags
    unpack_contractions=True ,  # Unpack contractions (can't -> can not)
    spell_correct_elong=True ,  # spell correction for elongated words
    dicts=[ emoticons ]
)


for i in tw_:
    #print ( i )
    line = p.preprocess_mentions ( line , repl='<mention>' )
    line = p.preprocess<reserved>words ( line , repl='<reserved>' )
    line = text_processor.pre_process_doc(line)
    tok.append ( tk.tokenize ( line ) )

for k in tok[1:100]:
    print(k)

model = Word2Vec( tok, min_count=5, size=300, window=5, sg=1)
model.train(tok, total_examples=len(tok), epochs=100)

model.wv.save_word2vec_format('w2v_positive_300.bin', binary=True)