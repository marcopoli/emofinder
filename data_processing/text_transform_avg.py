import joblib as joblib
import pandas as pd
import nltk
from nltk.tokenize import TweetTokenizer as TweetTokenizer
from nltk.chunk import tree2conlltags
from preprocessor.preprocess import *
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.dicts.emoticons import emoticons
from textblob import TextBlob
import corenlp
import collections
import spacy
import re
import emoji

from gensim.scripts.glove2word2vec import glove2word2vec
import regex
from spacy.tokenizer import Tokenizer
import gensim
import random as rn
import numpy as np
import pprint
from gensim.test.utils import datapath, get_tmpfile
from gensim.models.wrappers import FastText

#Load Dataset
dataset = pd.read_csv( 'test_emocontext.txt',delimiter='\t')
tweets1 = dataset['turn1']+dataset['turn2']+dataset['turn3']

#dataset = pd.read_csv( 'dev_emocontext.txt',delimiter='\t')
#tweets2 = dataset['turn1']+dataset['turn2']+dataset['turn3']
#tw =pd.concat([tweets1,tweets2],ignore_index=True, axis=1)
tweets = tweets1
tweets1


google_300 = gensim.models.KeyedVectors.load_word2vec_format("crawl-300d-2M.vec")
#gensim.models.KeyedVectors.load_word2vec_format("crawl-300d-2M.vec")
#gensim.models.KeyedVectors.load_word2vec_format( "../word2vec/embeddings/google_w2v_300.bin" , binary=True )

#gensim.models.KeyedVectors.load_word2vec_format("crawl-300d-2M.vec")

#gensim.models.KeyedVectors.load_word2vec_format( "../word2vec/embeddings/glove_word2vec.txt" , binary=False )
#senti_pos_500 = gensim.models.KeyedVectors.load_word2vec_format( "../word2vec/embedings/senti140_w2v_pos_100.bin", binary=True , unicode_errors='ignore')
#senti_neg_500 = gensim.models.KeyedVectors.load_word2vec_format( "../word2vec/embedings/senti140_w2v_neg_100.bin" , binary=True , unicode_errors='ignore')
#generics_500 = gensim.models.KeyedVectors.load_word2vec_format( "../word2vec/embeddings/generics_w2v_neutral_300.bin" , binary=True , unicode_errors='ignore')



#classes = dataset['label']
print(tweets)
tw_ = tweets
tok = list()

#Final 3Dvec
#train EmoCOntext 20463
#2761
matrix2D = np.zeros((5509,300))



pp = {
    'CC':1,
    'CD' :2,
    'DT' : 3,
    'EX' : 4,
    'FW' : 5,
    'IN' : 6,
    'JJ' : 7,
    'JJR' : 8,
    'JJS' : 9,
    'LS' : 10,
    'MD' : 11,
    'NN' : 12,
    'NNS' : 13,
    'NNP' : 14,
    'NNPS' : 15,
    'PDT' : 16,
    'POS' : 17,
    'PRP' : 18,
    'PRP$' : 19,
    'RB' : 20,
    'RBR' : 21,
    'RBS' : 22,
    'RP' : 23,
    'TO' : 24,
    'UH' : 25,
    'VB' : 26,
    'VBD' : 27,
    'VBG' : 28,
    'VBN' : 29,
    'VBP' : 30,
    'VBZ' : 31,
    'WDT' : 32,
    'WP' : 33,
    'WP$' : 34,
    'WRB' : 35,
    '.': 36,
    ',' : 37,
    ':' : 38}

iob = {
    'B-NP':1,
    'I-NP':2,
    'O':0
}

ner = {
        'PERSON':1,
        'NORP':2,
        'FAC':3,
        'ORG':4,
        'GPE':5,
        'LOC':6,
        'PRODUCT':7,
        'EVENT':8,
        'WORK_OF_ART':9,
        'LAW':10,
        'LANGUAGE':11,
        'DATE':12,
        'TIME':13,
        'PERCENT':14,
        'MONEY':15,
        'QUANTITY':16,
        'ORDINAL':17,
        'CARDINAL':18,
        '':19
}

def countSent(tokens):
    countVPos = 0
    countPos = 0
    countNeutral = 0
    countNeg = 0
    countVNeg = 0
    for core_tok in tokens:
        # Sentiment
        core_w_sent = core_tok.sentiment
        # print(core_w_sent)
        polw = 0
        if core_w_sent == 'Very negative':
            polw = -2
            countVNeg += 1
        if core_w_sent == 'Negative':
            polw = -1
            countNeg += 1
        if core_w_sent == 'Positive':
            polw = 1
            countPos += 1
        if core_w_sent == 'Very positive':
            polw = 2
            countVPos += 1
        if core_w_sent == 'Neutral':
            polw = 0
            countNeutral += 1
    return countVPos, countPos, countNeutral, countNeg, countVNeg


def countEsclamation(tw):
    esclamation = 0
    for token in tw.split():
        if (token == '!' or token == '!!' or token == '!!!' or token == '!!!!'):
            esclamation += 1
    return esclamation

def countQMark(tw):
    esclamation = 0
    for token in tw.split():
        if (token == '!' or token == '!!' or token == '!!!' or token == '!!!!'):
            esclamation += 1
    return esclamation

def countStopWord(tw):
    f = pd.read_fwf ( 'stopword_en' )
    stopList = f[ : ]
    isStop = 0
    for token in tw.split():
        if token in stopList:
            isStop += 1
    return isStop

def countInDictionary(tw):
    f = pd.read_fwf ( 'dict_basic_en' )
    stopList = f[ : ]
    isStop = 0
    for token in tw.split():
        if token in stopList:
            isStop += 1
    return isStop


prefix_re = re.compile(r'''^[\[\("']''')
suffix_re = re.compile(r'''[\]\)"']$''')
infix_re = re.compile(r'''[-~]''')
simple_url_re = re.compile(r'''^https?://''')
def custom_tokenizer(nlp):
    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                token_match=simple_url_re.match)

def countNouns(tw):
    count = 0
    for tupla in tw:
        if tupla[1] == 'NN' or tupla[1] == 'NNP' or tupla[1] == 'NNS' or tupla[1] == 'NNPS':
            count+=1
    return count

def countVerbs( tw ):
    count = 0
    for tupla in tw:
        if tupla[ 1 ] == 'VB' or tupla[ 1 ] == 'VBD' or tupla[ 1 ] == 'VBG' or tupla[ 1 ] == 'VBN' or tupla[ 1 ] == 'VBN' or tupla[ 1 ] == 'VBP' or tupla[ 1 ] == 'VBZ ':
           count += 1
    return count

def countAdjectivies ( tw ):
    count = 0
    for tupla in tw:
       if tupla[ 1 ] == 'JJ' or tupla[ 1 ] == 'JJR' or tupla[ 1 ] == 'JJS':
          count += 1
    return count

def countPronoun ( tw ):
    count = 0
    for tupla in tw:
        if tupla[ 1 ] == 'PRP' or tupla[ 1 ] == 'PRP$':
            count += 1
    return count

def countAdverb ( tw ):
    count = 0
    for tupla in tw:
        if tupla[ 1 ] == 'RB' or tupla[ 1 ] == 'RBR' or tupla[ 1 ] == 'RBS':
            count += 1
    return count

def countEmoji(text):
    emoji_counter = 0
    data = regex.findall(r'\X', text)
    for word in data:
        if any(char in emoji.UNICODE_EMOJI for char in word):
            emoji_counter += 1
            # Remove from the given text the emojis
            text = text.replace(word, '')

    words_counter = len(text.split())

    return emoji_counter, words_counter

def countHashtags(tw):
    pat = re.compile ( r"#(\w+)" )
    l = pat.findall (tw)
    return len(l)

def numSpecial_meta(tw):
    countMention = 0
    countReserved = 0
    countUrls = 0
    countNumbers= 0
    countPercents = 0
    countEmails = 0
    countMoney = 0
    countPhone = 0
    countTime = 0
    countDate = 0

    for c in tw:
        if c == '<mention>':
            countMention+=1
        if c == '<reserved>':
            countReserved+=1
        if c == '<url>' or c == '<url>':
            countUrls+=1
        if c == '<number>':
            countNumbers+=1
        if c == '<percent>':
            countPercents+=1
        if c == '<email>':
            countEmails+=1
        if c == '<money>':
            countMoney+=1
        if c == '<phone>':
            countPhone+=1
        if c == '<time>':
            countTime+=1
        if c == '<date>':
            countDate=1

    return countMention,countReserved,countUrls,countNumbers,countEmails,countMoney,countPhone,countTime,countDate

def checkRetweet(tw):
    if 'RT ' in tw:
        return 1
    else:
        return 0

def countWhitespaces(tw):
    numberWhite = sum ( 1 for c in tw if c == ' ' or c=='   ' )
    return numberWhite

def pecentUpper(tw):
    numberUpper = sum ( 1 for c in tw if c.isupper ( ) )
    wordLength = sum ( 1 for c in tw)
    upperPercent = numberUpper / wordLength
    return upperPercent

def percRepeatedChars(tw):
    totRepetitions = 0;
    d = collections.defaultdict( int )
    for c in tw:
        d[ c ] += 1
    for c in sorted ( d , key=d.get , reverse=True ):
        if d[ c ] > 1:
            totRepetitions = totRepetitions + d[ c ]
    wordLength = sum ( 1 for c in tw )
    repPercent = totRepetitions / wordLength
    return repPercent

#client = corenlp.CoreNLPClient ( start_server=False , annotators="sentiment".split ( ) )



#Preprocessing and Tokenization
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

#REPLACE with SPECIAL TAGS

for i in tw_:
    #print ( str(i) )
    #line = p.preprocess_urls ( i , repl='<url>' )
    line = p.preprocess_mentions ( str(i) , repl='<mention>' )
    line = p.preprocess_reserved_words ( line , repl='<reserved>' )
    line = text_processor.pre_process_doc(line)
    tok.append ( tk.tokenize ( line ) )


indexs = 0
#Create the token (n-gram) vector
for line in tok:

    original_tw = tweets[indexs]
    reconstructed_tw = ''


    lineLen = len(line)
    for token in line:

        #Take from Embeddings
        g_vec =[]
        is_in_model = False
        #token = token.lower()
        if token == "<mention>":
            g_vec = google_300.wv[ "mention" ]
            is_in_model = True
        elif token == "<url>":
            is_in_model = True
            g_vec = google_300.wv[ "url" ]
        elif token == "<reserved>":
            is_in_model = True
            g_vec = google_300.wv[ "reserved" ]
        elif token == "<number>":
            is_in_model = True
            g_vec = google_300.wv[ "number" ]
        elif token == "<percent>":
            is_in_model = True
            g_vec = google_300.wv[ "percent" ]
        elif token == "<money>":
            is_in_model = True
            g_vec = google_300.wv[ "money" ]
        elif token == "<email>":
            is_in_model = True
            g_vec = google_300.wv[ "email" ]
        elif token == "<phone>":
            is_in_model = True
            g_vec = google_300.wv[ "phone" ]
        elif token == "<time>":
            is_in_model = True
            g_vec = google_300.wv[ "time" ]
        elif token == "<date>":
            is_in_model = True
            g_vec = google_300.wv[ "date" ]

        elif token in google_300.wv.vocab.keys ( ):
             is_in_model = True
             g_vec = google_300.wv[ token ]

        elif not is_in_model:
            max = len ( google_300.wv.vocab.keys ( ) ) - 1
            index = rn.randint ( 0 , max )
            word = google_300.index2word[ index ]
            g_vec = google_300.wv[ word ]

        # VECTOR FOR EACH TOKEN
        tok2D = []
        tok2D = g_vec #np.concatenate((g_vec,gen_vec), axis = 0)
        # tok2D = np.concatenate ( (tok2D , senti_p_vec) , axis=0 )
        # tok2D =  np.concatenate ( (tok2D , senti_n_vec) , axis=0 )


        actual_vec = matrix2D[indexs]
        matrix2D[ indexs ] = [ x + y for x , y in zip ( actual_vec , tok2D ) ]


    #Normalize line
    actual_vec = matrix2D[ indexs ]
    matrix2D[ indexs ] = [ x/lineLen for x in actual_vec ]
    indexs = indexs + 1


    print ( '------------' , indexs )



joblib.dump(matrix2D, 'matrix2D_EMOCONT_test_fasttext_flatten')


