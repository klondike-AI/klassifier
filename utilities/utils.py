#***********************************
# SPDX-FileCopyrightText: 2009-2020 Vtenext S.r.l. <info@vtenext.com> and KLONDIKE S.r.l. <info@klondike.ai>
# SPDX-License-Identifier: AGPL-3.0-only
#***********************************

# coding=utf-8
from sklearn.metrics import classification_report, confusion_matrix
from difflib import SequenceMatcher
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted, column_or_1d
from keras.utils.np_utils import to_categorical
from nltk.corpus import stopwords
from imblearn.under_sampling import RandomUnderSampler #AllKNN, EditedNearestNeighbours,RepeatedEditedNearestNeighbours
from imblearn.over_sampling import RandomOverSampler
import dill as pickle
import copy
import re
import tracemalloc
import os
import pdb
import sys
import numpy as np
import string
from nltk.stem import SnowballStemmer
import pprint
import treetaggerwrapper
import json
from multiprocessing import cpu_count, Pool
import pandas as pd
from django.utils.text import slugify
import traceback
import pycountry
import datetime
from autocorrect import Speller
import math
from collections import Counter
from tabulate import tabulate
#spacy
import spacy
from spacy.lang.en import English
#conformal prediction and semantic CNN
from nonconformist.nc import ClassifierNc,ClassificationErrFunc,NcFactory, InverseProbabilityErrFunc, MarginErrFunc
from nonconformist.icp import IcpClassifier
from nonconformist.base import ClassifierAdapter
import spacy_tags_index #from file spacy_tags_index.py
import gensim
from multiprocessing import cpu_count
CORES = cpu_count()
import psutil
import gc
import torch



from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D, Conv1D, Concatenate, Lambda
from tensorflow.keras.layers import  SpatialDropout1D, GlobalMaxPooling1D, MaxPooling1D, ZeroPadding1D, Convolution1D
from tensorflow.keras.layers import Embedding, Input, BatchNormalization, Flatten, Dense, Dropout, AlphaDropout, ThresholdedReLU, Activation, concatenate
from tensorflow.keras.optimizers import Adam,Nadam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import Sequence
from keras.utils.np_utils import to_categorical

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted, column_or_1d
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelBinarizer 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from pandas.api.types import CategoricalDtype, is_numeric_dtype


import Stemmer

SPACY_MAX_LENGHT = 900000 #number of chars of a single input text to spacy pipeline (for NER element)
PRINTABLE = set(string.printable)

# suppress tensorflow warning and info logs 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import subprocess as sp
import logging


def get_gpu_memory():

    try:
        _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

        COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        print(memory_free_values)
    except:
        memory_free_values = [-1]
        
    return memory_free_values[0]


def setup_tf(disable_eager_execution=True, gpu_memory_limit=10000): #in MB

    tf.get_logger().setLevel(logging.ERROR) #disable Tensorflow standard logs 
    if disable_eager_execution:
        tf.compat.v1.disable_eager_execution()
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_virtual_device_configuration(gpu,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_memory_limit)])
    
            print(gpus)
            print(str(len(gpus)) + " GPU(s) available" if len(gpus) > 0 else "Warning: no GPU available.")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

setup_tf(gpu_memory_limit=get_gpu_memory()*0.9)


# Reset Keras Session
def reset_keras():
    try:
        tf.keras.backend.clear_session()
        gc.collect()
    except:
        pass

# class "override" to ignore unseen labels in test data for CNN 
# unseen labels are labels contained in test data but not in the train dataset 
class TolerantLabelEncoder(LabelEncoder):
    def __init__(self, ignore_unknown=False,
                       unknown_original_value='unknown', 
                       unknown_encoded_value=-1):
        self.ignore_unknown = ignore_unknown
        self.unknown_original_value = unknown_original_value
        self.unknown_encoded_value = unknown_encoded_value

    def transform(self, y):
        check_is_fitted(self, 'classes_')
        y = column_or_1d(y, warn=True)

        indices = np.isin(y, self.classes_)
        if not self.ignore_unknown and not np.all(indices):
            raise ValueError("y contains new labels: %s" 
                                         % str(np.setdiff1d(y, self.classes_)))

        y_transformed = np.searchsorted(self.classes_, y)
        y_transformed[~indices]=self.unknown_encoded_value
        return y_transformed

        y_transformed = np.asarray(self.classes_[y], dtype=object)
        y_transformed[~indices]=self.unknown_original_value
        return y_transformed
        
    def inverse_transform(self, y):
        check_is_fitted(self, 'classes_')

        labels = np.arange(len(self.classes_))
        indices = np.isin(y, labels)
        if not self.ignore_unknown and not np.all(indices):
            raise ValueError("y contains new labels: %s" 
                                         % str(np.setdiff1d(y, self.classes_)))

        y_transformed = np.asarray(self.classes_[y], dtype=object)
        y_transformed[~indices]=self.unknown_original_value
        return y_transformed


class semantic_cnn():
    
    def __init__(self, labels, xval, yval, language, spacy_model, embedding_vocab, embedding_params, preprocessing_params, semantic_features_params, cnn_model_train_params, semantic_features_embeddings):
        #PARAMS
        self.CNN_FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        self.NB_EPOCHS = cnn_model_train_params["NB_EPOCHS"]
        self.BATCH_SIZE = cnn_model_train_params["BATCH_SIZE"]
        self.NB_CONVOLUTION_FILTERS = cnn_model_train_params["NB_CONVOLUTION_FILTERS"]
        self.CONVOLUTION_KERNEL_SIZE = cnn_model_train_params["CONVOLUTION_KERNEL_SIZE"]
        self.LABEL_SMOOTHING = cnn_model_train_params["LABEL_SMOOTHING"]
        self.EARLYSTOPPING_PATIENCE = cnn_model_train_params["EARLYSTOPPING_PATIENCE"]
        self.EARLYSTOPPING_MONITOR_PARAM = cnn_model_train_params["EARLYSTOPPING_MONITOR_PARAM"]
        self.DROPOUT_PROB = cnn_model_train_params["DROPOUT_PROB"]
        self.CNN_ACTIVATION_FUNCTION = "relu"
        self.checkpoint_file = ""#'./Checkpoints/CNN_train_best_weights.h5'
        self.checkpoints = []
        
        #OPTIMIZABLE PARAMS
        self.embedding_params = embedding_params
        self.preprocessing_params = preprocessing_params
        self.semantic_features_params = semantic_features_params
        self.semantic_features_embeddings = semantic_features_embeddings
        
        self.language = language
        self.model = None
        self.results = None
        self.embeddings = embedding_vocab
        self.embedding_vocab_index_set = set(embedding_vocab.wv.index_to_key)
        self.accuracy = None
        
        #spacy stuff
        self.spacy_model = spacy_model
        
        #label encoder (onehotencoding)
        self.labels_count = len(Counter(labels))
        self.le = TolerantLabelEncoder(ignore_unknown=True)
        self.le.fit(labels)
        
        #validation set
        self.x_val = xval
        self.y_val = yval
        
        #conformal model
        self.conformal_model = None
        self.conformal_model_accuracy = None
    
    
    #remove unwanted attributes to be pickled
    """
    def __getstate__(self):
        
        d = dict(self.__dict__)
        if "model" in d.keys(): del d['model']
        if "checkpoints" in d.keys(): del d['checkpoints']
        #spacy_model to save disk space/time 
        if "spacy_model" in d.keys(): del d['spacy_model']
        #gensim embedding vocab to save disk space/time
        if "embeddings" in d.keys(): del d['embeddings']
        if "embedding_vocab_index_set" in d.keys(): del d['embedding_vocab_index_set']
        
        return d
   
    
    def __setstate__(self, d):
        
        self.__dict__.update(d)
    """    
        
    def input_preprocess(self,texts,labels = None,use_batch_generator=True):
        """
        input: cleaned word sequences and labels
        returns: padded encoded input sequences: words w2v, indexes to semantic embedding matrices for each semantic feature and onehotencoded labels
        """
        
        oov_perc = oov_count = np.zeros(len(texts))
        oov_all = []
        texts = filter_by_max_lenght(texts,SPACY_MAX_LENGHT,max_lenght_in_words=self.preprocessing_params['max_length'])
    
        #get values to local var to speedup the loop
        EMBEDDING_DIM = self.embedding_params['embedding_size']
        SEMANTIC_FEATURES = self.preprocessing_params['semantic_features']
        #add an empty value as when searching for the index while building the index vectors in the main loop we need to take into
        #account that the first row of the embedding matrix is the 0-vector that represent the "not found" value
        SFEATS_POS = np.append(np.array([""]),self.semantic_features_embeddings['pos']['feats_labels'])
        SFEATS_DEP = np.append(np.array([""]),self.semantic_features_embeddings['dep']['feats_labels'])
        SFEATS_ENT = np.append(np.array([""]),self.semantic_features_embeddings['ent']['feats_labels'])
        SFEATS_TAG = np.append(np.array([""]),self.semantic_features_embeddings['tag']['feats_labels'])
        POS_ENABLED = self.semantic_features_params['pos']
        DEP_ENABLED = self.semantic_features_params['dep']
        ENT_ENABLED = self.semantic_features_params['ent']
        TAG_ENABLED = self.semantic_features_params['tag']
        DEP_LEV_ENABLED = self.semantic_features_params['dep_lev']
        LEMMING = self.preprocessing_params['lemming']
        STEMMING = self.preprocessing_params['stemming']
        REMOVE_OOV = self.preprocessing_params['remove_oov']
        w2v_vocab = self.embeddings.wv.key_to_index
        
        start_ip = datetime.datetime.now()
        print("Processing input - START: ", start_ip)
        print("--- Processing " + str(len(texts)) + " input texts")
        
        print_current_memory_state(VERBOSE=1)
        
        #initialize the stemmer / add spelling component to spacy pipe
        #stemmer = SnowballStemmer(self.language) 
        stemmer = Stemmer.Stemmer(self.language) # PyStemmer should be faster than Snowball
        all_semantic_feats = {} 
        all_words = []
        for k in self.semantic_features_params.keys():
            all_semantic_feats[k] = []
        
        #for each text in input dataset
        for texts_batch in batch_generator(texts, batch_size=(2000 if use_batch_generator else len(texts) + 1)):
            for j,s in enumerate(self.spacy_model.pipe(texts_batch,n_process=int(CORES*0.8), batch_size=200)):#divmod(len(texts),CORES)[0]):

                #initialize arrays
                oov_list = []
                words_vector = [] #word indexes (to the embedding matrix) for a text
                semantic_features = {} #semantic feats vocab for indexes (to the embedding matrix) for a text
                for k in self.semantic_features_params.keys():
                    semantic_features[k] = []

                #for each word in text
                for token in s:
                    word = token.text

                    if LEMMING:
                        word = token.lemma_
                    if STEMMING:
                        word = stemmer.stemWord(word)
                    if token.pos_ == "PROPN":
                        if token.ent_type_ != "":
                            word = token.ent_type_
                        else: 
                            word = "_ENTITY_"

                    #index to word2vec embedding matrix - 0 if word not in word2vec vocab (0-vector as input) 
                    w2v = self.token2index_fast(w2v_vocab, self.embedding_vocab_index_set, word)

                    if w2v == 0:
                        oov_count[j] += 1
                        oov_list.append(word)

                    #oov removal based on word2vec vocab 
                    if not REMOVE_OOV or (REMOVE_OOV and w2v != 0):

                        #onehotencoding of spacy tags
                        if SEMANTIC_FEATURES: 

                            #indexes to embedding matrices
                            if POS_ENABLED: semantic_features['pos'].append(np.argmax(SFEATS_POS == token.pos_)) #spacy attr is "" if not found --> index 0 of embedding matrix
                            else: semantic_features['pos'].append(0) # 0-vector  

                            if DEP_ENABLED: semantic_features['dep'].append(np.argmax(SFEATS_DEP == token.dep_)) #spacy attr is "" if not found --> index 0 of embedding matrix
                            else: semantic_features['dep'].append(0) # 0-vector  

                            if ENT_ENABLED: semantic_features['ent'].append(np.argmax(SFEATS_ENT == token.ent_type_)) #spacy attr is "" if not found --> index 0 of embedding matrix
                            else: semantic_features['ent'].append(0) # 0-vector  

                            if TAG_ENABLED: semantic_features['tag'].append(np.argmax(SFEATS_TAG == token.tag_)) #spacy attr is "" if not found --> index 0 of embedding matrix
                            else: semantic_features['tag'].append(0) # 0-vector  

                            #integer input NOT index as there is no Embedding layer in the input channel in model for this feature (not categorical)
                            dep_level = abs(list(token.ancestors)[0].i - token.i) if DEP_LEV_ENABLED and token.dep_ != "ROOT" else 0
                            semantic_features['dep_lev'].append(dep_level)

                        else: 
                            semantic_features['pos'].append(0)
                            semantic_features['dep'].append(0)
                            semantic_features['ent'].append(0)
                            semantic_features['tag'].append(0)
                            semantic_features['dep_lev'].append(0)

                        #gensim w2v representation vector
                        words_vector.append(w2v)

                oov_perc[j] = oov_count[j] * 100 / len(s)
                oov_all.append(oov_list)
                #print("oov_count: ", oov_count[j], " / " , len(s), " --> " , oov_perc[j], "%")

                all_words.append(np.array(words_vector))
                all_semantic_feats['pos'].append(np.array(semantic_features['pos']))
                all_semantic_feats['dep'].append(np.array(semantic_features['dep']))
                all_semantic_feats['ent'].append(np.array(semantic_features['ent']))
                all_semantic_feats['tag'].append(np.array(semantic_features['tag']))
                all_semantic_feats['dep_lev'].append(np.array(semantic_features['dep_lev']))

        # pad the sequences 
        all_words = pad_sequences(all_words, maxlen=self.preprocessing_params['max_length'] ,dtype=float)
        all_semantic_feats['pos'] = pad_sequences(all_semantic_feats['pos'], maxlen=self.preprocessing_params['max_length'] ,dtype=float)
        all_semantic_feats['dep'] = pad_sequences(all_semantic_feats['dep'], maxlen=self.preprocessing_params['max_length'] ,dtype=float)
        all_semantic_feats['ent'] = pad_sequences(all_semantic_feats['ent'], maxlen=self.preprocessing_params['max_length'] ,dtype=float)
        all_semantic_feats['tag'] = pad_sequences(all_semantic_feats['tag'], maxlen=self.preprocessing_params['max_length'] ,dtype=float)
        all_semantic_feats['dep_lev'] = pad_sequences(all_semantic_feats['dep_lev'], maxlen=self.preprocessing_params['max_length'] ,dtype=float)
                
        # perform labels one hot encoding 
        labels_one_hot = None
        if labels is not None:
            labels_le = self.le.transform(labels)
            labels_one_hot = to_categorical(labels_le)
        # labels from encoder mapping 
        #TEXT_LABELS = self.le.classes_
        print("Processing input -   END: ",datetime.datetime.now()," - total exec time: ",self.elapsed_time(start_ip,datetime.datetime.now()))
        
        
        print("OOV WORDS AVERAGE PERCENTAGE : ", oov_perc.mean() if len(oov_perc) > 0 else 0 , "%")
        unique_oov = {x for l in oov_all for x in l}
        print("UNIQUE OOV WORDS FOUND: ", len(unique_oov))
        
        #dump_var_to_file(unique_oov, "unique_oov_"+self.language+"_"+str(datetime.datetime.now()).replace("-","").replace(" ","").replace(":",".")[0:16], "logs")
        
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #MUST BE IN THE SAME ORDER AS THE MODEL INPUT LAYERS *CHECK MODEL STRUCTURE*
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #the input data has to be a list of nparrays, one for each model input layer
        inputs = [all_words]
        for k in self.semantic_features_embeddings.keys(): 
            inputs.append(all_semantic_feats[k])
        #inputs.append(all_semantic_feats['dep_lev'])
        
        if labels is not None:
            return inputs, labels_one_hot
        else:
            return inputs
    
       
    def compile(self):

        print("COMPILE CNN MULTIPLE INPUTS with total labels: " + str(self.labels_count))
        
        EMBEDDING_DIM = self.embedding_params['embedding_size']
        MAXLEN = self.preprocessing_params['max_length']

        # embedding matrix contains weights for all the words in gensim word2vec vocab + 1 as the 0-index in 
        # the embedding matrix is used for the 0-vector for OOV words
        w2v_embedding_matrix = np.zeros((len(self.embeddings.wv.key_to_index) + 1, EMBEDDING_DIM))
        for word,wv in self.embeddings.wv.key_to_index.items():
            weights = self.embeddings.wv[word]
            w2v_embedding_matrix[self.token2index_fast(self.embeddings.wv.key_to_index, self.embedding_vocab_index_set, word) + 1] = weights

        convolution_outputs = []
        inputs = []
        #input layers
        w2v_input = Input(shape=(MAXLEN,),sparse=False,name="w2v")
        embedding_layer = Embedding(len(w2v_embedding_matrix),EMBEDDING_DIM, name='emb_w2v')#w2v vectors lenght
        embedding_layer.build((None, MAXLEN))
        embedding_layer.set_weights([w2v_embedding_matrix])
        embedding_layer.trainable = False
        w2v_embedded = embedding_layer(w2v_input)
        w2v_conv = Convolution1D(
                    filters=self.NB_CONVOLUTION_FILTERS,
                    kernel_size=self.CONVOLUTION_KERNEL_SIZE,
                    padding='same',
                    activation=self.CNN_ACTIVATION_FUNCTION,
                    name='conv_w2v')(w2v_embedded)
        convolution_outputs.append(w2v_conv)
        inputs.append(w2v_input)
        
        #generate one input layer for each semantic feature
        for f in self.semantic_features_embeddings.keys(): #pos dep ent tag

            semantic_feats_input = Input(shape=(MAXLEN,), name=self.semantic_features_embeddings[f]['name'])
            semantic_feats_embedded = Embedding(
                    self.semantic_features_embeddings[f]['feats_vocab_len'],
                    self.semantic_features_embeddings[f]['feats_embedding_dim'],#oeh vec lenght 
                    weights=[self.semantic_features_embeddings[f]['feats_embedding_matrix']], 
                    input_length=MAXLEN,
                    trainable=False,
                    name='emb_'+self.semantic_features_embeddings[f]['name'])(semantic_feats_input)
            semantic_feats_conv = Convolution1D(
                    filters=self.NB_CONVOLUTION_FILTERS,
                    kernel_size=self.CONVOLUTION_KERNEL_SIZE,
                    padding='same',
                    activation=self.CNN_ACTIVATION_FUNCTION,
                    name='conv_'+self.semantic_features_embeddings[f]['name'])(semantic_feats_embedded)
            
            convolution_outputs.append(semantic_feats_conv)
            inputs.append(semantic_feats_input)
            
        #dep_level input layer
        #semantic_feats_dep_lev_input = Input(shape=(MAXLEN,), name='dep_lev')
        #semantic_feats_dep_lev_conv = Convolution1D(
        #        filters=self.NB_CONVOLUTION_FILTERS,
        #        kernel_size=self.CONVOLUTION_KERNEL_SIZE,
        #        padding='same',
        #        activation=self.CNN_ACTIVATION_FUNCTION,
        #        name='conv_dep_lev')(semantic_feats_dep_lev_input)
        #inputs.append(semantic_feats_dep_lev_input)
        #convolution_outputs.append(semantic_feats_dep_lev_conv)
        
        #merge convultions outputs from all the input layers into the input for the main convolution
        merged_layers = Concatenate()(convolution_outputs)
        
        output = Convolution1D(
                filters=self.NB_CONVOLUTION_FILTERS,
                kernel_size=self.CONVOLUTION_KERNEL_SIZE,
                padding='same',
                activation=self.CNN_ACTIVATION_FUNCTION)(merged_layers)
        output = BatchNormalization()(output)
        output = MaxPooling1D()(output)
        output = Flatten()(output)
        output = Dropout(self.DROPOUT_PROB)(output)
        output = Dense(100, activation=self.CNN_ACTIVATION_FUNCTION)(output)
        output = Dropout(self.DROPOUT_PROB)(output)
        output = Dense(self.labels_count,activation='softmax')(output)
        
        model = Model( inputs, output, name='CNNClassifier')

        #LABEL SMOOTHING
        loss = CategoricalCrossentropy(label_smoothing=self.LABEL_SMOOTHING)
        checkpoints = []
        
        model.compile(
            optimizer=Nadam(),
            loss=loss, # 'categorical_crossentropy',
            metrics=['accuracy'])
    
        checkpoints.append(
            ModelCheckpoint(self.checkpoint_file, monitor=self.EARLYSTOPPING_MONITOR_PARAM, verbose=0,
               save_best_only=True, save_weights_only=True, mode='auto', period=1))

        checkpoints.append(
            TensorBoard(
                log_dir='./logs',
                histogram_freq=0,
                write_graph=True,
                write_images=False,
                embeddings_freq=0,
                embeddings_layer_names=None,
                embeddings_metadata=None))

        checkpoints.append(EarlyStopping(monitor=self.EARLYSTOPPING_MONITOR_PARAM, 
                                         patience=self.EARLYSTOPPING_PATIENCE, restore_best_weights=True))
        
        self.checkpoints = checkpoints
        self.model = model

       
    def fit(self,x,y, validation = True):
        
        start_fit = datetime.datetime.now()
        print("FIT CNN - START: ", start_fit)
        
        xval = yval = None
        if validation: 
            xval, yval = self.input_preprocess(self.x_val, self.y_val)
        
        self.model.fit( x, y, 
                        batch_size=self.BATCH_SIZE, 
                        epochs=self.NB_EPOCHS, 
                        validation_data=(xval, yval) if validation else None,
                        #steps_per_epoch = len(y), # SET ONLY IF USING BATCH GENERATOR AS INPUT
                        verbose=1, 
                        callbacks=self.checkpoints,
                        workers=CORES, use_multiprocessing=True)
        #load the weights from the checkpoint file, the best weights
        self.model.load_weights(self.checkpoint_file)

        print("--- CNN MODEL SUMMARY: ")
        print(self.model.summary())
        print("FIT CNN -   END: ",datetime.datetime.now()," - total exec time: ",self.elapsed_time(start_fit,datetime.datetime.now()))
        
        return self.model


    def predict(self,X_test):

        #print("PREDICT CNN")
        return self.model.predict(X_test,workers=CORES,use_multiprocessing=True)
    
    
    def predict_proba(self,X_test):
        
        return self.predict(X_test)
        

    def evaluate(self,X_test,Y_test):

        print("EVALUATE CNN")
        results = self.model.evaluate(X_test, Y_test, workers=CORES, use_multiprocessing=True)
        
        self.results = results
        return results
    
    
    def probs_to_labels(self,probs_list):
        
        preds = []
        for i in range(0, len(probs_list)):
            probs = probs_list[i]
            predicted_index = np.argmax(probs)
            preds.append(str(self.le.inverse_transform(predicted_index)))

        return preds


    def token2index_fast(self, vocab, index_set, token):
        
        if token in index_set: 
            return vocab.get(token) #model.vocab[token].index
        else:
            return 0


    def elapsed_time(self, start, end):

        seconds = (end-start).seconds
        days, seconds = divmod(seconds, 86400)
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        if days > 0:
            return '%dd%dh%dm%ds' % (days, hours, minutes, seconds)
        elif hours > 0:
            return '%dh%dm%ds' % (hours, minutes, seconds)
        elif minutes > 0:
            return '%dm%ds' % (minutes, seconds)
        else:
            return '%ds' % (seconds,)
        
        
    def save_model(self, save_path, keras_model_filename, semantic_cnn_object_filename):
    
        #save the internal keras cnn model
        self.model.save(os.path.join(save_path, keras_model_filename))
        
        #delete not picklable objects 
        #keras stuff 
        self.model = None
        self.checkpoints = None
        #spacy_model to save disk space/time 
        self.spacy_model = None
        #gensim embedding vocab to save disk space/time
        self.embeddings = None
        self.embedding_vocab_index_set = None
                
        gc.collect()
        #now save the semantic_cnn class object
        with open(os.path.join(save_path, semantic_cnn_object_filename), "wb") as pklfile:
            pickle.dump(self, pklfile)
        
    
    def load_internals(self, save_path, keras_model_filename, word2vec_vocab, language):
        
        #loads all the internal objects needed
    
        #load the internal keras cnn model
        self.model = load_model(os.path.join(save_path, keras_model_filename))
        
        #spacy_model load
        self.spacy_model = load_spacy(language)
        
        #gensim embedding vocab
        #self.embeddings = gensim.models.Word2Vec.load(word2vec_filename)
        self.embeddings = word2vec_vocab
        self.embedding_vocab_index_set = set(self.embeddings.wv.index_to_key)
            

class lemmer(object):
    def __init__(self, language):
        self.lang = language
    def __call__(self, data):
        return list_word_lemming(data, self.lang)        


class words_remover(object):
    def __init__(self, words):
        self.words = words
    def __call__(self, list_of_strings):
        return remove_words_in_list(list_of_strings, self.words) 


class speller(object):
    def __init__(self, language, parameters):
        self.lang = language
        self.params = parameters
    def __call__(self, list_of_strings):
        return word_spelling_corrector(list_of_strings, self.lang, self.params)   
        

class oov_remover_obj(object):
    def __init__(self, spacy_model, parameters):
        self.nlp = spacy_model
        self.params = parameters
    def __call__(self, list_of_strings):
        return oov_remover(list_of_strings, self.nlp, self.params)   


class MyClassifierAdapter(ClassifierAdapter):
    def __init__(self, model, fit_params=None):
        super(MyClassifierAdapter, self).__init__(model,fit_params)
        self.model = model
        
    def fit(self,x,y):
        return self.model.fit(x,y)
    
    def predict(self,x):
        return self.model.predict_proba(x)
    
    def compile(self):
        return self.model.compile()
    

def report_and_confmat(test_labels, prediction, title="generic", save_data=False):
    """
    Compute the performance of the classification and compute the confusion matrix. It saves them into a file "title.txt".
    :param test_labels: labels of test data
    :param prediction: predicted labels
    :param title: name of the file to save
    :param save_data: bool, wether to save arrays of test and predicted labels
    :return report: text report of metrics 
    :return conf_mat: the confusion matrix
    :return metrics: the metrics dictionary with this struct {'label 1': {'precision':0.5,'recall':1.0,'f1-score':0.67,'support':1}, 'label 2': { ... },...}
    """
    ytest = metrics = report = conf_mat = None
    save_dir = os.path.join(os.getcwd(), "Results")
    try:
        ytest = np.array(test_labels)

        if save_data:
            #save arrays to file
            with open(os.path.join(os.path.join(os.getcwd(), "logs"), title + "_DUMP_predictions.txt"), "w") as file:
                file.write("\nPredictions |~~| True Values\n")
                for j in range(len(prediction)):
                    file.write(str(prediction[j]) + " |~~| " + str(ytest[j]) + "\n")

        #print(title + " started \n")
        report = classification_report(ytest, prediction, output_dict=False, zero_division=0)
        metrics = classification_report(ytest, prediction, output_dict=True, zero_division=0)
        conf_mat = confusion_matrix(ytest, prediction, normalize='true')#values_format= '.0%'
        
    except Exception as e:
        print("Error when making classification report or confusion matrix for: " + title)
        print(e)
        traceback.print_exc()

    try:
        with open(os.path.join(save_dir, title + "_results.txt"), "w") as file:
            t = np.get_printoptions()["threshold"]
            l = np.get_printoptions()["linewidth"]
            float_formatter = "{:.2f}".format
            np.set_printoptions(threshold=sys.maxsize, linewidth=1000000, suppress=True, formatter={'float_kind':float_formatter})
            
            file.write(str(report))
            file.write("\nConfusion matrix:\n")
            file.write(str(conf_mat))
            np.set_printoptions(threshold=t, linewidth=l)
            
    except FileNotFoundError as e:
        print("Error when saving classification report into file")
        print(e)
        traceback.print_exc()
    
    return report, conf_mat, metrics


def similar_sentences(sentence_1, sentece_2):
    """Compute the similarity between two strings
    :returns: float, similarity score
    """

    sim = SequenceMatcher(None, sentence_1, sentece_2).ratio()
    return sim


def create_train_test(x, x_labels, test_size=0.25, json_parameters = None, proportional=True):
    """
    Uses train test split from sk-learn to split data into two sets.
    :param x: data
    :param x_labels: labels
    :param test_size: size of the testing set
    :param proportional: split the dataset keeping the proportions between text and labels
    :return: train data, train labels, test data, test labels
    """
    X_train = X_test = y_train = y_test = None

    try:
        stratify_param = get_service_parameter_col(json_parameters,"stratify_split")
        if stratify_param is not None and stratify_param is False:
            proportional = False
        
        stratify = None
        if proportional:
            stratify = x_labels
        X_train, X_test, y_train, y_test = train_test_split(x, x_labels, test_size=test_size, stratify=stratify)
        
    except ValueError as e:
        print("Error when splitting data into training/testing sets")
        print(e)
        traceback.print_exc()

    return X_train, X_test, y_train, y_test


def correct_dataframe(frame, cat_list):
    """
    Corrects misspelled classes name.
    :param frame: DataFrame
    :param cat_list: list of categories in the dataframe
    :return: Dataframe with corrected classes
    """
    try:
        for cat1 in frame:
            for cat2 in cat_list:
                if cat1 != cat2:
                    score = similar_sentences(cat1, cat2)
                    if score > 0.90 and score != 1:
                        frame.replace(to_replace=cat1, value=cat2)
    except RuntimeError as e:
        print("Error when correcting misspelled categories")
        print(e)
        traceback.print_exc()
    return frame


def parallel_remove_words(list_of_strings, words):

    print("Removing words from dataset texts")
    cores = cpu_count() #Number of CPU cores on your system
    data_split = np.array_split(list_of_strings, cores)
    pool = Pool(cores)
    data = pd.concat(pool.map(words_remover(words), data_split))
    pool.close()
    pool.join()
    
    return data.values


def remove_words_in_list(list_of_strings, unwanted_words):
    output = list()
    for text in list_of_strings:
        output.append(' '.join([word for word in text.split(' ') if word not in unwanted_words])) 
        #output.append(' '.join(list(set(text.split(' ')) - set(unwanted_words))))
            
    return pd.Series(output)


def clean_text_OLD(text, language):
    """
    Cleans text from unwanted words or characters.
    :param text: String, sentence
    :return: cleaned text, String
    """
    import string
    
    # Remove punctuation
    text = text.translate(string.punctuation)
    # Clean the text. Some re may have been applied before in the code, it's just to be sure.
    text = re.sub(r"[^A-Za-z0-9^,!./'+-=]", " ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"/", " ", text)
    text = re.sub(r"!", " ", text)
    text = re.sub(r"\^", " ", text)
    text = re.sub(r"\+", " ", text)
    text = re.sub(r"-", " ", text)
    text = re.sub(r"=", " ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\W*\b\w{1,2}\b", " ", text)
    text = re.sub(r" \d+", " ", text)
    text = re.sub(r"\W*\b\w{26,}\b", " ", text)
    text = re.sub(' +', ' ', text)
    text = re.sub(r";", " ", text)
    
    # Convert words to lower case and split them
    text = text.lower().split()
    # Remove stop words
    if language != '' and language is not None:
        stops = set(stopwords.words(language))
        text = [w for w in text if not w in stops and len(w) >= 3]
    text = " ".join(text)
    
    return text


def clean_text(text,language,deep_clean=True, max_string_char_count=SPACY_MAX_LENGHT):
    """
    Cleans text from unwanted words or characters.
    :param text: String, sentence
    :param max_string_char_count: needed for spacy as NER pipe object manages max 1.000.000 char strings
    :return: cleaned text, String
    """    

    if len(text) > max_string_char_count:
        text = text[0:max_string_char_count]
    
    # Convert words to lower case and split them
    text = text.lower()
    
    #text = re.sub(r"\S+@\S+", "email_address", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\W*\b\w{26,}\b", " ", text)
    text = re.sub(' +', ' ', text)
    text = re.sub(r"/+", " ", text)
    text = re.sub(r"\^+", " ", text)
    text = re.sub(r"\++", " ", text)
    text = re.sub(r"-+", " ", text)
    text = re.sub(r"=+", " ", text)
    text = re.sub(r"e - mail", "mail", text)
    text = re.sub(r"e-mail", "mail", text)
    text = re.sub(r"email", "mail", text)
    text = re.sub(r"^nan ", " ", text)
    text = re.sub(r"^none ", " ", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r" 9 11 ", "911", text)

    if deep_clean is True:
        # Remove punctuation
        text = text.translate(string.punctuation)

        # Clean the text 
        text = re.sub(r"[^A-Za-z0-9^,!./'+-=]", " ", text)
        text = re.sub(r"\++", " ", text)
        text = re.sub(r":+", " ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"\W*\b\w{1,2}\b", " ", text)
        text = re.sub(r";", " ", text)
        
        # Remove stop words
        if language != '' and language is not None:
            stops = set(stopwords.words(language))
            text = " ".join([w for w in text.split() if not w in stops and len(w) >= 3])        
    
    return text


def bert_clean_text(text):
    """
    Basic text preprocessing for BERT models.
    """    
    
    text = ''.join(filter(lambda x: x in PRINTABLE, text))
    
    # Convert words to lower case and split them
    text = text.lower()
    
    #text = re.sub(r"\S+@\S+", "email_address", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\W*\b\w{26,}\b", " ", text)
    text = re.sub(' +', ' ', text)
    
    punct = '"#$%&*+/<=>@[\\]^_`{|}~'
    text = text.translate(punct)
    text = text.replace('\r\n','')
    text = text.replace('\n','')
    text = text.replace('\t','')
    text = text.replace('\r','')
    text = text.replace('#','')
    text = text.replace('=','')
    text = text.replace('>','')
    
    return text


def text_preprocess(data, text_column, parameters, data_colums = None, guessed_language = None):
    """
    Preprocess text for classification. Stopwords removal, lemming, stemming.
    :param data: whole dataset in which the text_column contains the text to be processed
    :return: the whole dataframe preprocessed
    """
    
    try:
        # fetch the text language from db
        language = get_language_parameter(parameters)
        
        if (language is '' or language is None) and guessed_language is not None:
            language = guessed_language

        data[text_column] = ''
        if data_colums is not None:
            cols = data_colums
        else:
            cols = data.columns
        for col in cols:
            if col != text_column:
                if type(data[text_column]) is str:
                    data[text_column] = data[text_column] + ' ' + data[col]
                else:
                    data[text_column] = data[text_column].map(str) + ' ' + data[col].map(str)
        
        if type(data[text_column]) is str:
            data[text_column] = clean_text(data[text_column], language)
        else:
            data[text_column] = data[text_column].map(lambda x: clean_text(x, language))
        
        #spell autocorrection --> applies only if spelling_autocorrect parameter is set to True or "fast"
        #data[text_column] = parallel_speller(data[text_column], language, parameters)
        
        #remove oov words 
        #if type(data[text_column]) is not str:
        #    data[text_column] = remove_oov_words(data[text_column].tolist(), language, parameters)
        
        
        # Lemmatize text 
        if get_lemming_parameter(parameters):
            if type(data[text_column]) is str:
                data[text_column] = word_lemming(data[text_column], language)
            else:
                data[text_column] = parallel_lemming(data[text_column], language)
            
        # Stemmatize text 
        if get_stemming_parameter(parameters):
            #print("Applying STEMMING")
            if type(data[text_column]) is str:
                data[text_column] = word_stemming(data[text_column], language)
            else:
                data[text_column] = data[text_column].map(lambda x: word_stemming(x, language))

    except ValueError as e:
        print("Error when processing the text.")
        print(e)
        traceback.print_exc()

    return data


def text_preprocessing(texts, params):
    
    return texts


def parallel_speller(list_of_strings, language, json_parameters):
    
    spelling_autocorrect = get_service_parameter_col(json_parameters,"spelling_autocorrect")
    
    if language == '' or language is None or spelling_autocorrect is None or spelling_autocorrect == '' or spelling_autocorrect is False:
        return list_of_strings
    
    print("Spelling autocorrect - START: " + str(datetime.datetime.now()))
    cores = cpu_count() #Number of CPU cores on your system
    data_split = np.array_split(list_of_strings, cores)
    pool = Pool(cores)
    data = pd.concat(pool.map(speller(language, json_parameters), data_split))
    pool.close()
    pool.join()
    print("Spelling autocorrect -   END: " + str(datetime.datetime.now()))
    
    return data.values


def word_spelling_corrector(text, language, json_parameters):
    """
    Given a text, it applies the autocorrect speller and returns the corrected text
    To be applied after text cleaning (stopwords removal, punctuation removal, etc..) 
    BUT before STEMMING/LEMMING
    :param text: the stringor list of strings to be corrected
    :return: the corrected string/list of strings
    """
    try:
        language = pycountry.languages.get(name=language).alpha_2.lower()
        fast_param = False
        
        spelling_autocorrect = get_service_parameter_col(json_parameters,"spelling_autocorrect")
        if spelling_autocorrect is None or spelling_autocorrect == '' or spelling_autocorrect is False:
            return text
        
        if spelling_autocorrect == "fast":
            fast_param = True

        spell = Speller(lang = language,fast=fast_param)

        #string case
        if type(text) is str:
            text = spell(text)
        #list of strings case
        else:
            text = text.map(lambda x: spell(x))
        
        return text

    except Exception as e:
        print("=== ERROR === while applying Spelling correction")
        print(e)
        traceback.print_exc()
        return text


def parallel_remove_oov(list_of_strings, language, json_parameters):
    
    oov_removal = get_service_parameter_col(json_parameters,"oov_removal")
    
    if language == '' or language is None or oov_removal is None or oov_removal == '' or oov_removal is False:
        return list_of_strings
    
    print("OOV removal - START: " + str(datetime.datetime.now()))
    spacy_model = load_spacy(language) 

    cores = cpu_count() #Number of CPU cores on your system
    data_split = np.array_split(list_of_strings, cores)
    pool = Pool(cores)
    data = pd.concat(pool.map(oov_remover_obj(spacy_model, json_parameters), data_split))
    pool.close()
    pool.join()
    print("OOV removal -   END: " + str(datetime.datetime.now()))
    
    return data.values


def oov_remover(text, spacy_model, json_parameters):
    """
    Given a text, it applies the spacy oov removal and returns the corrected text
    To be applied after text cleaning (stopwords removal, punctuation removal, etc..) 
    BUT before STEMMING/LEMMING
    :param text: the string or list of strings to be corrected
    :return: the corrected string/list of strings
    """
    try:
        oov_removal = get_service_parameter_col(json_parameters,"oov_removal")
        if oov_removal is None or oov_removal == '' or oov_removal is False:
            return text
        
        #string case
        if type(text) is str:
            text = remove_oov_from_string(text,spacy_model)
        #list of strings case
        else:
            text = text.map(lambda x: remove_oov_from_string(x, spacy_model))
        
        return text

    except Exception as e:
        print("=== ERROR === while applying spacy OOV REMOVAL")
        print(e)
        traceback.print_exc()
        return text


def remove_oov_from_string(text, spacy_model):
    """
    remove oov words using spacy library
    :param text: the input string
    :return: the cleaned string
    """
    tokens = spacy_model(text)
    return " ".join([token.text for token in tokens if token.has_vector])


def remove_oov_words(list_of_strings, language, json_parameters):
    """
    remove oov (out of vocaboulary) words from each string in the input list using spacy
    :param list_of_strings: the input list of strings
    :return: the output list 
    """
    #oov_removal parameter values: 
    #oov_removal = True --> removes oov words only
    #oov_removal = "lem" --> removes oov words + lemmatize text
    try:
        oov_removal = get_service_parameter_col(json_parameters,"oov_removal")
        if language == '' or language is None or oov_removal is None or oov_removal == '' or oov_removal is False:
            return list_of_strings
        
        print("OOV removal - START: " + str(datetime.datetime.now()))
        spacy_model = load_spacy(language)

        output_list = list()
        if oov_removal == "lem": #remove oov words and lematize text
            for text in spacy_model.pipe(list_of_strings):
                #output_list.append(" ".join([token.text for token in text if token.has_vector]))
                #also lemmatize text
                output_list.append(" ".join([token.lemma_ for token in text if token.has_vector]))
        else:
            for text in spacy_model.pipe(list_of_strings):
                output_list.append(" ".join([token.text for token in text if token.has_vector]))
                
        print("OOV removal -   END: " + str(datetime.datetime.now()))
        
        return output_list
        
    except Exception as e:
        print("=== ERROR === while applying REMOVE_OOV_WORDS")
        print("=== WARNING === spacy OOV REMOVAL and LEMMING not applied for language: " + language)
        print(e)
        traceback.print_exc()
        return list_of_strings    
    

def word_stemming(text, language):
    """
    Given a text, it applies the SnowballStemmer and returns the stemmed text
    To be applied after text cleaning (stopwords removal, punctuation removal, etc..)
    :param text: the string to be stemmed
    :return: the stemmed string
    """
    try:
        if language == '' or language is None:
            return text
        stemmer = SnowballStemmer(language)
        stemmed = list()
        words = text.split()
        
        for word in words:
            stemmed_word = stemmer.stem(word)
            stemmed.append(stemmed_word)
            
        return ' '.join([str(i) for i in stemmed])

    except Exception as e:
        print("=== ERROR === while applying STEMMING")
        print(e)
        traceback.print_exc()
        return text


def word_lemming(text, language):
    """
    Given a text, it gets  and returns the lemmed text
    To be applied after text cleaning (stopwords removal, punctuation removal, etc..)
    :param text: the string to be lemmed
    :return: the lemmed string
    """
    try:
        if language == '' or language is None:
            return text
        
        #if language == 'italian' : language = 'it'
        #if language == 'english' : language = 'en'
        language = pycountry.languages.get(name=language).alpha_2
         
        tagger = treetaggerwrapper.TreeTagger(TAGLANG=language,TAGDIR='./treetagger')
        tags = tagger.tag_text(text)
        tags2 = treetaggerwrapper.make_tags(tags)
        #pprint.pprint(tags2)
        lemmed = list()
        
        for tag in tags2:
            lemmed.append(tag.lemma)
            
        return ' '.join([str(i) for i in lemmed])
        
    except Exception as e:
        print("=== ERROR === while applying LEMMING")
        print(e)
        traceback.print_exc()
        return text
    
    
def list_word_lemming(data, language):
    """
    Given a text array, it gets  and returns the lemmed text array
    To be applied after text cleaning (stopwords removal, punctuation removal, etc..)
    :param text: the string array to be lemmed
    :return: the lemmed array
    """
    try:
   
        if language == '' or language is None:
            return data
        
        #if language == 'italian' : language = 'it'
        #if language == 'english' : language = 'en'
        language = pycountry.languages.get(name=language).alpha_2
            
        tagger = treetaggerwrapper.TreeTagger(TAGLANG=language,TAGDIR='./treetagger')

        for index,text in data.iteritems():
            tags = treetaggerwrapper.make_tags(tagger.tag_text(text))
            data[index] = ' '.join([t.lemma for t in tags])
    
    except Exception as e:
        print("=== ERROR === while applying LEMMING")
        print(e)
        traceback.print_exc()
        
    return data


def parallel_lemming(data, language):
    """
    Applies the list_word_lemming function in parallel (multithread) splitting the input by the number of CPU cores
    To be applied after text cleaning (stopwords removal, punctuation removal, etc..)
    :param data: the string array to be lemmed
    :return: the lemmed array
    """
    print("Applying LEMMING")
    cores = cpu_count() #Number of CPU cores on your system
    data_split = np.array_split(data, cores)
    pool = Pool(cores)
    data = pd.concat(pool.map(lemmer(language), data_split))
    pool.close()
    pool.join()
    
    return data.values


def check_for_folders():
    """
    Check that folders Results and Models exist. If not it creates them.
    """
    cwd = os.getcwd()
    res_path = os.path.join(cwd, "Results")
    if not os.path.exists(res_path):
        os.mkdir(res_path)

    mod_path = os.path.join(cwd, "Models")
    if not os.path.exists(mod_path):
        os.mkdir(mod_path)


def convert_data_to_index(list_of_sentences, wv):
    """
    Given a word2vec model compute the embeddings relative to the input
    :param list_of_sentences: sentences to change into vectors
    :param wv: voabulary of the word2vec model (created with Gensim)
    :return: indexed data corresponding to the sentences
    """
    index_data = []
    for sentence in list_of_sentences:
        index_sentence = []
        for word in sentence:
            if word in wv.vocab:
                index_sentence.append(wv.vocab[word].index)
        index_data.append(index_sentence)

    return index_data


def get_service_parameter_col(json_parameters, parameter_to_check):
    """
    Gets the value of parameter_to_check in the JSON string in the "parameters" column of the services table
    If the parameter_to_check is not set in the JSON string it returns None
    :param json_parameters: the json string from the services table
    :param parameter_to_check: the parameter to get 
    :return: the value if the parameter, None if not set
    """
    try:
        if json_parameters is None or json_parameters == "":
            return None
        
        parameters = json.loads(json_parameters)
        
        if parameter_to_check in parameters:
            return parameters[parameter_to_check]
        return None
    
    except Exception as e:
        print("=== ERROR === while retrieving the parameter" + parameter_to_check + " from parameters dictionary - check parameters column on services table")
        print(e)
        traceback.print_exc()
        return None
    

def get_language_string(lang_code):
    '''
    Returns the full language string value from the lang code in ISO 639-1 Code format
    using lib pycountry
    '''
    try:
        if lang_code == "DK":
            lang_code = "DA" #fix for danish
        #if lang_code == "NO":
        #    lang_code = "NB" #fix for norwegian
            
        lang = pycountry.languages.get(alpha_2=lang_code.strip()).name.lower()
    
    except Exception as e:
        print("--- ERROR --- Error when getting language ISO string for language code: " + str(lang_code))
        print(e)
        traceback.print_exc()
        lang = None
    
    return lang
    

def get_language_parameter(json_parameters):
    """
    Fetches the language parameter from the 'parameters' json column in services table
    :param json_parameters: the json string from the services table
    :return: the language string value
    """
    language = get_service_parameter_col(json_parameters,"language")
    
    if language is None:
        language = ''
    
    return language.strip().lower()


def get_language_target_col_parameter(json_parameters):
    """
    Fetches the language_target_col parameter from the 'parameters' json column in services table
    :param json_parameters: the json string from the services table
    :return: the language_target_col string value
    """
    language_target_col = get_service_parameter_col(json_parameters,"language_target_col")
    
    if language_target_col is not None:
        language_target_col = language_target_col.strip()
    
    return language_target_col


def get_remove_text_noise_parameter(json_parameters):
    """
    Fetches the remove_text_noise parameter from the 'parameters' json column in services table
    :param json_parameters: the json string from the services table
    :return: the remove_text_noise int value
    """
    remove_text_noise = get_service_parameter_col(json_parameters,"remove_text_noise")
    
    try: 
        return int(remove_text_noise)
    except:
        return 0


def get_min_cardinality_parameter(json_parameters):
    """
    Fetches the min_cardinality parameter from the 'parameters' json column in services table
    :param json_parameters: the json string from the services table
    :return: the min_cardinality INTEGER value
    """
    
    try:
        min_cardinality = get_service_parameter_col(json_parameters,"min_cardinality")
        
        if min_cardinality is None or str(min_cardinality) == '':
            return None
        
        return int(min_cardinality)
    
    except Exception as e:
        print("=== ERROR === while retrieving the parameter min_cardinality from parameters dictionary - check parameters column on services table")
        print(e)
        traceback.print_exc()


def get_min_max_cardinality_ratio_parameter(json_parameters):
    """
    Fetches the min_max_cardinality_ratio parameter from the 'parameters' json column in services table
    :param json_parameters: the json string from the services table
    :return: the min_max_cardinality_ratio value
    """
    
    try:
        min_max_cardinality_ratio = get_service_parameter_col(json_parameters,"min_max_cardinality_ratio")
        
        if min_max_cardinality_ratio is not None and min_max_cardinality_ratio != "":
            return min_max_cardinality_ratio
        else:
            return False #default is disabled
    
    except Exception as e:
        print("=== ERROR === while retrieving the parameter min_max_cardinality_ratio from parameters dictionary - check parameters column on services table")
        print(e)
        traceback.print_exc()

    
def get_lemming_parameter(json_parameters):
    """
    Fetches the lemming parameter from the 'parameters' json column in services table
    :param json_parameters: the json string from the services table
    :return: the lemming boolean value
    """
    
    try:
        lemming = get_service_parameter_col(json_parameters,"lemming")
        
        if lemming is not None and lemming is True:
            return True
        
        return False
    
    except Exception as e:
        print("=== ERROR === while retrieving the parameter lemming from parameters dictionary - check parameters column on services table")
        print(e)
        traceback.print_exc()


def get_stemming_parameter(json_parameters):
    """
    Fetches the stemming parameter from the 'parameters' json column in services table
    :param json_parameters: the json string from the services table
    :return: the stemming boolean value
    """
    
    try:
        stemming = get_service_parameter_col(json_parameters,"stemming")
        
        if stemming is not None and stemming is True:
            return True
        
        return False
    
    except Exception as e:
        print("=== ERROR === while retrieving the parameter stemming from parameters dictionary - check parameters column on services table")
        print(e)
        traceback.print_exc()


def get_CNN_config_params(json_parameters):
    """
    Fetches the CNN_config parameter from the 'parameters' json column in services table
    example: "CNN_config":
                {    
                    "NB_WORDS" : 10000,				                        -->  number of words in the dictionary
                    "NB_EPOCHS" : 30,				                        -->  Number of epochs
                    "BATCH_SIZE" : 512, 				                    -->  Size of the batches used in the mini-batch gradient descent    
                    "MAX_LEN" : 400,				                        -->  Maximum number of words in a sequence
                    "EMBEDDING_DIM" : 150,				                    -->  Number of dimensions of the GloVe word embeddings
                    "NB_CONVOLUTION_FILTERS" : 128,		                    -->  Number of convolution filters 
                    "CONVOLUTION_KERNEL_SIZE" : 4,		                    -->  Convolution Kernel Size
                    "LABEL_SMOOTHING" : 0.3,			                    -->  label smoothing index
                    "EARLYSTOPPING_PATIENCE" : 10,		                    -->  number of epochs without improvement in the monitored param that the model waits before stopping
                    "EARLYSTOPPING_MONITOR_PARAM" : "val_loss",		        -->  the value monitored for early stopping 
                    "DROPOUT_PROB" : 0.5,				                    -->  dropout CNN index
                    "PARAMS_AUTOTUNING" : false,                            -->  enables CNN hyperparams autotuning via Keras tuner class
                    "MULTIGROUP_CNN" : false,                               -->  enables CNN MultiGroup custom embeddings mode 
                    "MG_GLOVE_EMB_FILE" : "itwiki_20180420_300d.txt",       -->  CNN MultiGroup Glove embeddings file name (has to be inside the embeddings folder)
                    "MG_FASTTEXT_EMB_FILE" : "embed_wiki_it_1.3M_52D.vec",  -->  CNN MultiGroup FastText embeddings file name (has to be inside the embeddings folder)
                    "MG_GLOVE_EMB_DIM" : 300,                               -->  CNN MultiGroup Glove embeddings vectors dimension
                    "MG_FASTTEXT_EMB_DIM" : 52,                             -->  CNN MultiGroup FastText embeddings vectors dimension
                }
                 
    The params are not mandatory, if not present the default value is going to be used
    
    :param json_parameters: the json string from the services table
    :return: the CNN config dictionary
    """
    CNN_config = get_service_parameter_col(json_parameters,"CNN_config")
    if CNN_config is None: 
        CNN_config = dict()
    
    #set config defaults when not set in json
    if "NB_WORDS" not in CNN_config: CNN_config["NB_WORDS"] = 10000
    if "NB_EPOCHS" not in CNN_config: CNN_config["NB_EPOCHS"] = 30
    if "BATCH_SIZE" not in CNN_config: CNN_config["BATCH_SIZE"] = 512
    if "MAX_LEN" not in CNN_config: CNN_config["MAX_LEN"] = 400
    if "EMBEDDING_DIM" not in CNN_config: CNN_config["EMBEDDING_DIM"] = 100
    if "NB_CONVOLUTION_FILTERS" not in CNN_config: CNN_config["NB_CONVOLUTION_FILTERS"] = 128
    if "CONVOLUTION_KERNEL_SIZE" not in CNN_config: CNN_config["CONVOLUTION_KERNEL_SIZE"] = 4
    if "LABEL_SMOOTHING" not in CNN_config: CNN_config["LABEL_SMOOTHING"] = 0.2
    if "EARLYSTOPPING_PATIENCE" not in CNN_config: CNN_config["EARLYSTOPPING_PATIENCE"] = 8
    if "EARLYSTOPPING_MONITOR_PARAM" not in CNN_config: CNN_config["EARLYSTOPPING_MONITOR_PARAM"] = 'val_loss'
    if "DROPOUT_PROB" not in CNN_config: CNN_config["DROPOUT_PROB"] = 0.4
    if "PARAMS_AUTOTUNING" not in CNN_config: CNN_config["PARAMS_AUTOTUNING"] = False
    if "MULTIGROUP_CNN" not in CNN_config: CNN_config["MULTIGROUP_CNN"] = False
    if "MG_GLOVE_EMB_FILE" not in CNN_config: CNN_config["MG_GLOVE_EMB_FILE"] = "itwiki_20180420_300d.txt"
    if "MG_FASTTEXT_EMB_FILE" not in CNN_config: CNN_config["MG_FASTTEXT_EMB_FILE"] = "embed_wiki_it_1.3M_52D.vec"
    if "MG_GLOVE_EMB_DIM" not in CNN_config: CNN_config["MG_GLOVE_EMB_DIM"] = 300
    if "MG_FASTTEXT_EMB_DIM" not in CNN_config: CNN_config["MG_FASTTEXT_EMB_DIM"] = 52
        
    return CNN_config


def get_Tree_config_params(json_parameters):
    """
    Fetches the Tree_config parameter from the 'parameters' json column in services table
    example: "Tree_config":
                {    
                    "tree_classifier_min_score" : 0.7,		            -->  min score to save a trained model 
                    "tree_classifier_min_cardinality" : 50              -->  min cardinality of a subset to be considered "classifiable"
                    "tree_classifier_print_result_table" : false        -->  prints all the original and predicted values to stdout 
                    "tree_classifier_result_table_col_width" : 50       -->  stdout result table column width
                    "tree_classifier_conformal_classification" : False  -->  enables/disables conformal predictor
                    "tree_classifier_conformal_significance" : 0.2      -->  conformal prediction significance param    
                    "tree_classifier_enable_semantic_cnn" : False       -->  semantic cnn conformal prediction    
                    "tree_classifier_conformal_recalibration_loops" : 5 -->  conformal model recalibration loops over wrong predictions
                    "load_pretrained_w2v_vocab" : false                 -->  load pretrained w2v vocabs file (must be named as the one generated in usual train in ./Models folder
                }
                 
    The params are not mandatory, if not present the default value is going to be used
    
    :param json_parameters: the json string from the services table
    :return: the Tree config dictionary
    """
    
    Tree_config = get_service_parameter_col(json_parameters,"Tree_config")
    if Tree_config is None: 
        Tree_config = dict()
    
    #set config defaults when not set in json
    if "tree_classifier_min_score" not in Tree_config: Tree_config["tree_classifier_min_score"] = 0.7
    if "tree_classifier_min_cardinality" not in Tree_config: Tree_config["tree_classifier_min_cardinality"] = 50
    if "tree_classifier_print_result_table" not in Tree_config: Tree_config["tree_classifier_print_result_table"] = False
    if "tree_classifier_result_table_col_width" not in Tree_config: Tree_config["tree_classifier_result_table_col_width"] = 50
    if "tree_classifier_conformal_classification" not in Tree_config: Tree_config["tree_classifier_conformal_classification"] = False
    if "tree_classifier_conformal_significance" not in Tree_config: Tree_config["tree_classifier_conformal_significance"] = 0.2
    if "tree_classifier_enable_semantic_cnn" not in Tree_config: Tree_config["tree_classifier_enable_semantic_cnn"] = False
    if "tree_classifier_conformal_recalibration_loops" not in Tree_config: Tree_config["tree_classifier_conformal_recalibration_loops"] = 5
    if "tree_classifier_load_pretrained_w2v_vocab" not in Tree_config: Tree_config["tree_classifier_load_pretrained_w2v_vocab"] = False
                
    return Tree_config

def get_sentiment_config_params(json_parameters):
    """
    Fetches the sentiment_config parameter from the 'parameters' json column in services table
    example: "sentiment_config":
                {    
                    "max_len" : 40,		                -->  max number of words kept when using sentiment method
                    "sentiment_output" : true,          -->  predicts the sentiment using BERT nauraly model (postitive, neutral, negative)
                    "emotion_output" : true             -->  predicts the emotion using BERT-based feelit library (fear,sadness,joy,anger)
                }
                 
    The params are not mandatory, if not present the default value is going to be used
    
    :param json_parameters: the json string from the services table
    :return: the Tree config dictionary
    """
    
    sentiment_config = get_service_parameter_col(json_parameters,"sentiment_config")
    if sentiment_config is None: 
        sentiment_config = dict()
    
    #set config defaults when not set in json
    if "max_len" not in sentiment_config: sentiment_config["max_len"] = 40
    if "sentiment_output" not in sentiment_config: sentiment_config["sentiment_output"] = True
    if "emotion_output" not in sentiment_config: sentiment_config["emotion_output"] = False
                
    return sentiment_config


def get_text_preprocessing_params(json_parameters, autotuning_enabled=None):
    """
    Fetches the text_preprocessing_params from the 'parameters' json column in services table
    EACH PARAM CAN BE SET AS A LIST OF VALUES (FOR AUTOTUNING)
    IF autotuning_enabled is False and a list of values is set, it will ignore the list and set to the default (single) value
    example: "text_preprocessing_params":
                {    
                    'deep_clean' : 1,               #cleanup punctuation, spaces, special chars, short words, numbers, convert to lowercase.... 
                    'spelling': [0,1],                  #enables word spelling autocorrect in text preprocess
                    'stemming': [0,1],                  #enables stemming in text preprocess
                    'lemming': [0,1],                   #enables lemming in text preprocess
                    'remove_oov': [0,1],                #Out Of Vocaboulary words removal (remove words not in the embedding vocab
                    'remove_stopwords': [0,1],          #removes stopwords
                    'custom_word_replace' : [0,1]       #if 1 replaces the words in the replace_list param
                    'replace_list' : {
                                        "word1" : "word1_replacement",
                                        "word2" : "word2_replacement",
                                        "word3" : "word3_replacement",
                                        ...
                                     }
                }
                  
    The params are not mandatory, if not present the default value is going to be used
    
    :param json_parameters: the json string from the services table
    :return: the SEMANTIC_CNN_config dictionary
    """
    text_preprocessing_params = get_service_parameter_col(json_parameters,"text_preprocessing_params")
    if text_preprocessing_params is None: 
        text_preprocessing_params = dict()
    
    #set config defaults when not set in json 
    if autotuning_enabled is True:
        if "deep_clean" not in text_preprocessing_params: text_preprocessing_params["deep_clean"] = [1]
        if "spelling" not in text_preprocessing_params: text_preprocessing_params["spelling"] = [0,1]
        if "stemming" not in text_preprocessing_params: text_preprocessing_params["stemming"] = [0,1]
        if "lemming" not in text_preprocessing_params: text_preprocessing_params["lemming"] = [0,1]
        if "remove_oov" not in text_preprocessing_params: text_preprocessing_params["remove_oov"] = [0]
        if "remove_stopwords" not in text_preprocessing_params: text_preprocessing_params["remove_stopwords"] = [1]
        if "custom_word_replace" not in text_preprocessing_params: text_preprocessing_params["custom_word_replace"] = [0]
    else:
        if "deep_clean" not in text_preprocessing_params: text_preprocessing_params["deep_clean"] = [1]
        if "spelling" not in text_preprocessing_params: text_preprocessing_params["spelling"] = [0]
        if "stemming" not in text_preprocessing_params: text_preprocessing_params["stemming"] = [1]
        if "lemming" not in text_preprocessing_params: text_preprocessing_params["lemming"] = [1]
        if "remove_oov" not in text_preprocessing_params: text_preprocessing_params["remove_oov"] = [0]
        if "remove_stopwords" not in text_preprocessing_params: text_preprocessing_params["remove_stopwords"] = [1]
        if "custom_word_replace" not in text_preprocessing_params: text_preprocessing_params["custom_word_replace"] = [0]

    return text_preprocessing_params


def get_cc_config_params(json_parameters):
    """
    Fetches the cc_config parameter from the 'parameters' json column in services table
    example: "cc_config":
                {    
                    "cc_min_score" : 0.8,		            -->  min score to save a trained model 
                    "cc_min_cardinality" : 20,              -->  min cardinality of a subset to be considered "classifiable"
                    "cc_min_cardinality_ratio" : 0.1,       -->  min cardinality ratio over the cardinality of the largest class
                    "crap_classes_size_multiplier": 0.8     -->  multiplies the cardinality of the smallest non-crap class to obtain the crap classes cardinality
                }
                 
    The params are not mandatory, if not present the default value is going to be used
    
    :param json_parameters: the json string from the services table
    :return: the cc_config dictionary
    """

    cc_config = get_service_parameter_col(json_parameters,"cc_config")
    if cc_config is None: 
        cc_config = dict()
    
    #set config defaults when not set in json
    if "cc_min_score" not in cc_config: cc_config["cc_min_score"] = 0.8
    if "cc_min_cardinality" not in cc_config: cc_config["cc_min_cardinality"] = 20
    if "cc_min_cardinality_ratio" not in cc_config: cc_config["cc_min_cardinality_ratio"] = 0.1
    if "crap_classes_size_multiplier" not in cc_config: cc_config["crap_classes_size_multiplier"] = 0.8
        
    return cc_config


def get_tf_config_params(json_parameters):
    """
    Fetches the tf_config parameter from the 'parameters' json column in services table
    example: "tf_config":
                {    
                    "tf_min_accuracy" : 0.8,		         -->  min accuracy
                    "tf_train_classes_subset_size" : 2       -->  number of classes to train/classify each time
                }
                 
    The params are not mandatory, if not present the default value is going to be used
    
    :param json_parameters: the json string from the services table
    :return: the tf_config dictionary
    """
    
    tf_config = get_service_parameter_col(json_parameters,"tf_config")
    if tf_config is None: 
        tf_config = dict()
    
    #set config defaults when not set in json
    if "tf_min_accuracy" not in tf_config: tf_config["tf_min_accuracy"] = 0.8
    if "tf_train_classes_subset_size" not in tf_config: tf_config["tf_train_classes_subset_size"] = 2
        
    return tf_config

   
def get_measurement_criteria(json_parameters):
    """
    Fetches the measure parameter from the 'parameters' json column in services table
    example: {"measure":{"type":"class-specific", "class":"Bloccante", "metric":"recall"}}
    example: {"measure":{"type":"global", "score":"accuracy", "metric":"f1-score"}}
    
    Accepted string values in json:
    "type": "class-specific" -> "class":/any class label name/            | "metric":/recall/f1-score/precision/support
    "type": "global" ->         "score":/accuracy/weighted avg/macro avg/ | "metric":/recall/f1-score/precision/support
    
    :param json_parameters: the json string from the services table
    :return: measure_metric, measure_score_or_class values
    """
    measure = get_service_parameter_col(json_parameters,"measure")
    measure_metric = None
    measure_score_or_class = "accuracy"

    if measure is None: return measure_metric, measure_score_or_class
    
    if measure["type"].lower().strip() == "global":
        measure_score_or_class = measure["score"]
        if measure_score_or_class == "accuracy":
            measure_metric = None
        else:
            measure_metric = measure["metric"]
        
    if measure["type"].lower().strip() == "class-specific":
        measure_metric = measure["metric"]
        measure_score_or_class = measure["class"]
    
    return measure_metric, measure_score_or_class
    

def get_metric(metric_dictionary, metric, score_or_class):
    """
    Fetches the wanted metric from the sklearn metric_dictionary
    :param metric_dictionary: the dictionary in output of the classification_report function 
    :param metric: the metric to be fetched
    :param score_or_class: the best 
    :return: the value of the fetched metric
    """
    try:
        sc = score_or_class.strip()
        if sc == "accuracy": return metric_dictionary[sc]
        m = metric.lower().strip()
        
        return metric_dictionary[sc][m]
    
    except Exception as e:
        print("=== ERROR === while retrieving metric from metric dictionary - check parameters column on services table")
        print(e)
        traceback.print_exc()

    
def filter_by_cardinality(dataframe, label_column, MIN_CARDINALITY):
    """
    Filters out all the labels with cardinality < MIN_CARDINALITY
    :param dataframe: the pandas dataframe containing all the training data
    :param label_column: the column name containing the labels
    :param min_cardinality: the minimum cardinality of labels to be classified 
    :return: -
    """
    try:
        if MIN_CARDINALITY is None or MIN_CARDINALITY == '': return dataframe
        
        #gets label count for each label 
        label_counts = dataframe[label_column].value_counts(sort=True,ascending=False)
        label_counts_dataframe = label_counts.to_frame()
        
        #gets the labels with cardinality <= MIN_CARDINALITY
        filtered_labels = label_counts_dataframe[label_counts_dataframe[label_column] <= MIN_CARDINALITY].index
        
        #removes all the rows with labels not in filtered_labels ( "~" is a "NOT" operator)
        fdf = dataframe[~dataframe[label_column].isin(filtered_labels)]
        
        print("EXLUDING THE FOLLOWING LABELS (CARDINALITY < " + str(MIN_CARDINALITY) + "): ")
        print(filtered_labels)
        
        return fdf
        
    except Exception as e:
        print("=== ERROR === while filtering by MIN_CARDINALITY")
        print(e)
        traceback.print_exc()


def read_embedding(path):
    """
    Read embedding values from file 
    :param path: embeddings file path
    :return: embeddings dict with values
    """

    embeddings_index = {}
    f = open(path)
    skipped = list()
    for line in f:
        try:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        except:
            skipped.append(line)
            continue
    print("======= SKIPPED the following lines while read_embedding in file: " + path)
    for s in skipped:
        print(s)
        
    f.close()

    return embeddings_index


def create_embedding_matrix(tokenizer, embedding_dim, embeddings_index):
    """
    Creates the embedding matrix for the model
    :param tokenizer: the word tokenizer instance 
    :param embedding_dim: the embedding word vect dimension 
    :param embeddings_index: the embeddings dictionary
    :return: embedding matrix 
    """

    word_index = tokenizer.word_index
    if tokenizer.num_words:  # if num words is set, get rid of words with too high index
        word_index = {key: word_index[key] for key in word_index.keys()
                      if word_index[key] < (tokenizer.num_words + 1)}
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

    
def set_modelid(layer,label):
    """
    generates a valid modelid to be set in the model filename for the TreeClassifier
    layer
    """
    id = slugify(layer) + "_" + slugify(label)
    
    return id
    

def dump_var_to_file(var, filename, filedir):
    file = os.path.join(os.path.join(os.getcwd(), filedir), filename)
    with open(file, "w") as f:
        pprint.pprint(var,f)
        
    
def is_crap_label(label, crap_label_suffix):
    if label is None: return False
    if type(label) is pd.Series:
        result = []
        for l in label:
            result.append(str(l).startswith(crap_label_suffix))
        return pd.Series(result)
    
    if label == "": return False
    if str(label).startswith(crap_label_suffix): return True
    return False


def split_dataset(dataframe, target_col, window_size):

    """
    splits the first window_size classes and the remaining ones into two dfferent datasets
    returns the two splitted parts and the label at which the split happened (the first excluded one)
    """
    
    #gets label count for each label 
    label_counts = dataframe[target_col].value_counts(sort=True,ascending=False)
    dfc = pd.DataFrame(columns=dataframe.columns)
    
    print("--- Preparing dataset for training with the following classes: (first " + str(window_size) + " classes sorted by cardinality) ")
    
    for i in range(window_size):
        label = label_counts.index[i]
        dfc = dfc.append(dataframe.loc[dataframe[target_col] == label])
        dataframe.drop(dataframe[dataframe[target_col] == label].index, inplace=True)
        print(label)
    
    return dfc, dataframe, label_counts.index[window_size]


def print_std_and_file(text,file,append = True):
    
    """
    prints both to stdout/console and file
    text : string to be printed
    file : file object already open to append
    """
    mode = 'a'
    if append == False:
        mode = 'w'
    
    if file is not None and file != "":
        with open(file, mode) as f:
            print(str(text),file=f)
    
    print(str(text))
        
    return


def balance_dataset_by_undersampling(df, target_col, min_max_cardinality_ratio, oversampling_ratio = None):

    """
    balance the dataset by undersampling the classes with:  class_cardinality > (min_cardinality parameter / min_max_cardinality_ratio)
    oversampling_ratio has to be between 0 and 1
    """
    try:
        if min_max_cardinality_ratio is None or min_max_cardinality_ratio is False:
            return df
        
        #gets label count for each label 
        label_counts = df[target_col].value_counts(sort=True,ascending=False)
        balanced_label_counts = dict()
        max_cardinality = label_counts[0]
        min_cardinality = label_counts[-1]
        mean_cardinality = label_counts.mean()
        
        balanced_class_max_cardinality = int(min_cardinality / min_max_cardinality_ratio)
        
        for i in label_counts.index: 
            if label_counts[i] > balanced_class_max_cardinality:
                balanced_label_counts[i] = balanced_class_max_cardinality
            else:
                balanced_label_counts[i] = label_counts[i]
                            
        undersampler = RandomUnderSampler(sampling_strategy = balanced_label_counts)
        df_undersampled, target_col_undersampled = undersampler.fit_resample(df, df[target_col])

        print("--- BALANCING DATASET BY UNDERSAMPLING CLASSES - New dataset has " + str(len(df_undersampled.index)) + " rows")

        if oversampling_ratio is not None:
            largest_class_cardinality_under_threshold = -1
            for i in label_counts.index: 
                if label_counts[i] >= balanced_class_max_cardinality: 
                    continue
                if largest_class_cardinality_under_threshold == -1:
                    largest_class_cardinality_under_threshold = label_counts[i]
                balanced_label_counts[i] = int(largest_class_cardinality_under_threshold / oversampling_ratio)

            oversampler = RandomOverSampler(sampling_strategy = balanced_label_counts)
            df_oversampled, target_col_oversampled = oversampler.fit_resample(df_undersampled, df_undersampled[target_col])
        
            print("--- BALANCING DATASET BY OVERSAMPLING CLASSES - New dataset has " + str(len(df_oversampled.index)) + " rows")
                
            return df_oversampled
        
        else:
            return df_undersampled
    
    except Exception as e:
        print("--- ERROR --- Error when undersampling classes")
        print(e)
        traceback.print_exc()
        

def get_id_list_from_file(inputfile):

    """
    Gets a list of ids from file. Accepted format:
    1. comma separated values like 1,2,3
    2. one id per line
    3. range format:  "start-end"  like 1-10 -> gets all the ids from 1 to 10 , both start and end included
    """
    ids = None
    with open(inputfile, "rb") as idsfile:
        content = idsfile.read()
        if b',' in content:
            #id list is in comma-separated format like 1,2,3....
            ids = content.decode("utf-8").replace('\r','').replace('\n','').split(",")
        elif b'-' in content:
            #id list in startid-endid format like 1-10 gets all ids from 1 to 10 - start and end included
            limits = content.decode("utf-8").replace('\r','').replace('\n','').split("-")
            ids = list(range(int(limits[0]),int(limits[1])+1))
        else:
            #id list has one id per line
            ids = content.decode("utf-8").splitlines()
            
    return ids


def force_garbage_collection(verbose=0):

    before_gc = psutil.virtual_memory().available / 10**6 #in MB
    if verbose == 2: print("--- GARBAGE COLLECTION START --- Current memory usage is ", psutil.virtual_memory().percent, "% - ", before_gc, "MB")
    deleted_objects = gc.collect()
    if verbose == 2: print("--- GARBAGE COLLECTION       --- collected: ", deleted_objects, " objects")
    after_gc = psutil.virtual_memory().available / 10**6 #in MB
    if verbose == 2: print("--- GARBAGE COLLECTION       --- Current memory usage is ", psutil.virtual_memory().percent, "% - ", after_gc, "MB")
    if verbose == 1: print("--- GARBAGE COLLECTION END   --- freed memory: ", after_gc - before_gc, "MB")
    

def print_current_memory_state(VERBOSE=0):

    print("--- MEMORY STATE - RAM  AVAILABLE: ", psutil.virtual_memory().available / 10**6, "MB - ", 100 - psutil.virtual_memory().percent, "%")
    if VERBOSE > 0:
        print("--- MEMORY STATE - SWAP AVAILABLE: ", psutil.swap_memory().free / 10**6, "MB - ", 100 - psutil.swap_memory().percent, "%")
      

def filter_column_by_max_lenght(dataframe, column, max_lenght_in_chars):
    
    dataframe[column] = dataframe[column].str.slice(start=0, stop=max_lenght_in_chars-1)
    
    return dataframe


def filter_by_max_lenght(texts, max_lenght_in_chars,max_lenght_in_words=0):
    #texts MUST BE A PANDAS SERIES OBJECT
    texts = texts.str.slice(start=0, stop=max_lenght_in_chars-1)
    if max_lenght_in_words > 0:
        texts = texts.str.split(n=max_lenght_in_words).str[:max_lenght_in_words].str.join(' ')
        
    return texts


def get_combination(self, params_dict_keys, values):
        res = {}
        for idx,k in enumerate(params_dict_keys):
            res[k] = values[idx]
            
        return res

#############################################################################################
##################    CONFORMAL PREDICTION AND SEMANTIC CNN FUNCS    ########################
#############################################################################################

def get_conformal_model(model, x_cal, y_cal, x_test, y_test, significance, labelencoder):
        
        model_nc = MyClassifierAdapter(model)

        #initialize conformal classifier
        nc = ClassifierNc(model_nc, MarginErrFunc())#,InverseProbabilityErrFunc())#MarginErrFunc()
        icp = IcpClassifier(nc)
        #fit conformal classifier
        
        #calibrate conformal classifier
        icp.calibrate(x_cal, y_cal)#np.array([np.where(r==1)[0][0] for r in y_cal]))

        #predict 
        predictions = icp.predict(x_test, significance = significance)
        
        pred_labels = []
        for p in predictions:
            pred_labels.append(labelencoder.classes_[np.where(p==True)[0]])
        
        accuracy = compute_model_accuracy(pred_labels, y_test)
                
            
        return accuracy, icp


def get_conformal_model_with_recalibration(model, x_cal, y_cal, x_test, y_test, significance, labelencoder, tfidf_vect=None, recalibration_loops = 5):
        
        #fit conformal classifier
        num_labels = len(labelencoder.classes_)
        best_accuracy = 0
        best_model = None
        
        for r in range(recalibration_loops):
            
            model_nc = MyClassifierAdapter(model)

            #initialize conformal classifier
            nc = ClassifierNc(model_nc, MarginErrFunc())#,InverseProbabilityErrFunc())#MarginErrFunc()
            icp = IcpClassifier(nc)
            
            pred_labels = []
            if tfidf_vect is None: 
                #semantic cnn model
                xcal,ycal = model.input_preprocess(x_cal,y_cal)
                xtest,ytest = model.input_preprocess(x_test,y_test)
                ycal = np.array([np.where(r==1)[0][0] for r in ycal])
            else:
                #sklearn models
                xcal = tfidf_vect.transform(x_cal)
                xtest = tfidf_vect.transform(x_test)
                ycal = labelencoder.transform(y_cal)

            #calibrate conformal classifier
            print("--- CONFORMAL MODEL CALIBRATION LOOP #", r+1) 
            icp.calibrate(xcal, ycal)

            #predict 
            predictions = icp.predict(xtest, significance = significance)

            x_recal = []
            y_recal = []
            for j,p in enumerate(predictions):
                #all unpredicted inputs goes into recalibration set
                conformal_pred_lenght = (p==True).sum()
                if conformal_pred_lenght == 0 or conformal_pred_lenght == num_labels:
                    y_recal.append(y_test.iloc[j])
                    x_recal.append(x_test.iloc[j])
                    
                pred_labels.append(labelencoder.classes_[np.where(p==True)[0]])
            
            accuracy = compute_model_accuracy(pred_labels, y_test, VERBOSE=1)            
            
            if accuracy > best_accuracy:
                if best_accuracy > 0: print("--- GOT BETTER ACCURACY AFTER ", r+1, " CALIBRATION LOOPS: ", best_accuracy ," -> ", accuracy)
                best_accuracy = accuracy
                best_model = icp
            
            if len(x_recal) < 2:
                break
                
            x_cal = pd.Series(x_recal)
            y_cal = pd.Series(y_recal)
            
        return best_accuracy, best_model


def compute_model_accuracy(predictions, true_values, VERBOSE=1):

        #accuracy = mean value of sum of 1/prediction_set_cardinality for each prediction
            
        trueval_in_preds = []

        c = 0
        for i,true_val in enumerate(true_values):
            if true_val in predictions[i]:
                trueval_in_preds.append(1/len(predictions[i])) 
            else:
                trueval_in_preds.append(0)
            c += len(predictions[i])

        output_pred_mean_lenght = c/len(true_values)
        output_pred_containing_trueval_mean_lenght = np.array([1/j for j in trueval_in_preds if j > 0]).mean()

        accuracy = np.array(trueval_in_preds).mean()
        
        if VERBOSE == 1:
            print("Mean lenght of output predictions set: " , int(output_pred_mean_lenght))#, " over a total of: " , len(classification_labels))
            print(np.count_nonzero(np.array(trueval_in_preds) > 0),  " / " , len(trueval_in_preds), " predictions contains the true value")
            print("The mean value of ratios (1/number of possible classes) for all the predictions is: " , np.array(trueval_in_preds).mean(), " ")
            print("The mean lenght of prediction sets containing the true value is: " ,output_pred_containing_trueval_mean_lenght)

            print("Prediction containing trueval - prediction lenght frequencies: ")
            print(tabulate(sorted(Counter(np.array([int(1/j) for j in trueval_in_preds if j > 0])).items()),headers=["lenght","text count"]))

        if VERBOSE == 2:
            print("Predictions: ")
            print(pred_labels)
                
        return accuracy

    
def load_spacy(language):

    try:
        torch.set_num_threads(1) # necessary to avoid deadlocks (pytorch + spacy3 issue https://github.com/pytorch/pytorch/issues/17199 )
        lang_code = pycountry.languages.get(name=language).alpha_2.lower()
        if lang_code == "en":
            spacy_model = spacy.load(lang_code + '_core_web_lg') #load the large dictionary
        else:
            if lang_code == "no":
                lang_code = "nb"
            spacy_model = spacy.load(lang_code + '_core_news_lg') #load the large dictionary
            
    except Exception as e:
        print("--- ERROR --- Error when loading spacy for language code: " + str(lang_code))
        print(e)
        traceback.print_exc()
        spacy_model = None
        
    return spacy_model


def elapsed_time(start, end):

    seconds = (end-start).seconds
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return '%dd%dh%dm%ds' % (days, hours, minutes, seconds)
    elif hours > 0:
        return '%dh%dm%ds' % (hours, minutes, seconds)
    elif minutes > 0:
        return '%dm%ds' % (minutes, seconds)
    else:
        return '%ds' % (seconds,)


def batch_generator(iterable, batch_size=100):
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx:min(ndx + batch_size, l)]
        
 
def text_preprocessing_fast(texts, spacy_model, language, preprocessing_params, use_batch_generator = False):
    """
    input: raw texts
    clean text + spelling 
    returns: cleaned word sequences (GENSIM Word2Vec input ready)
    """
   
    start_pp = datetime.datetime.now()
    print("GENSIM Word2Vec TEXT PREPROCESSING - START: ", start_pp)
    print_current_memory_state(VERBOSE=1)

    texts = filter_by_max_lenght(texts,SPACY_MAX_LENGHT)
    LEMMING = preprocessing_params['lemming']
    STEMMING = preprocessing_params['stemming']
    stemmer = Stemmer.Stemmer(language)
    
    if preprocessing_params['spelling']:
        contextualSpellCheck.add_to_pipe(spacy_model)

    REMOVE_OOV = preprocessing_params['remove_oov'] 
    pos_filters = set(["INTJ","DET"]) #["PROPN","INTJ","DET"]
    sequences = []
    
    for texts_batch in batch_generator(texts, batch_size=(2000 if use_batch_generator else len(texts) + 1)):
        for s in spacy_model.pipe(texts_batch,n_process=int(CORES*0.8), batch_size=200):#divmod(len(texts),CORES)[0]):
            seq = []
            #add meaningful stuff to vec representation
            for token in s:
                if (    not token.is_punct 
                    and not token.is_stop 
                    and not token.is_space 
                    and not token.is_digit
                    and not token.like_num
                    and not token.is_quote
                    and not (token.is_oov if REMOVE_OOV else 0)
                    and not token.is_bracket
                    #and not token.is_currency
                    #and not token.like_url 
                    and token.pos_ not in pos_filters
                    ):

                    word = token.text
                    if token.like_email: 
                        word = "mail"
                    if token.pos_ == "PROPN":
                        if token.ent_type_ != "":
                            word = token.ent_type_
                        else: 
                            word = "_ENTITY_"
                            seq.append(word)
                            continue
                    if LEMMING:
                        word = token.lemma_
                    if STEMMING:
                        word = stemmer.stemWord(word)
                    
                    seq.append(word)

            sequences.append(seq)

    print("GENSIM Word2Vec TEXT PREPROCESSING -   END: ",datetime.datetime.now()," - total exec time: ",elapsed_time(start_pp,datetime.datetime.now()))

    return sequences


def generate_word2vec_vocab(sequences, embedding_params, preprocessing_params, spacy_model, language, save_filename = None, load_pretrained_filename = None):
    #initialize gensim word dictionary
    start_w2v = datetime.datetime.now()
    print("GENSIM Word2Vec - START: ", start_w2v)
    
    if spacy_model is None: #unsupported languages
        print("GENSIM Word2Vec - UNSUPPORTED LANGUAGE - ",language, " - ABORTING...")
        return None
    
    if load_pretrained_filename is not None and os.path.exists(load_pretrained_filename):
        #load gensim pretrained vocab from file
        print("--- Loading pretrained gensim wrod2vec dictionary from file: ", load_pretrained_filename)
        vocab = gensim.models.Word2Vec.load(load_pretrained_filename)
    else:
        print("--- Generating GENSIM Word2Vec dictionary over " + str(len(sequences)) + " sequences, for language: " , language)
        vocab = gensim.models.Word2Vec(
            text_preprocessing_fast(sequences, spacy_model, language, preprocessing_params, use_batch_generator = True), #requires a list of "list of words"
            window = embedding_params['window_size'],
            vector_size = embedding_params['embedding_size'],
            sg = embedding_params['sg_true'],
            min_count = embedding_params['min_count'],
            workers=CORES
        )   
        if save_filename is not None:
            vocab.save(save_filename)
            print("--- Saving WORD2VEC vocab for language: " , language, " to file: ", save_filename)
            
    print("--- GENSIM dictionary size: " + str(len(vocab.wv.index_to_key)))
    print("GENSIM Word2Vec -   END: ",datetime.datetime.now()," - total exec time: ",elapsed_time(start_w2v,datetime.datetime.now()))

    return vocab


def get_semantic_features_embeddings(language):

    TAGS_DICT = {}
    POS_TAGS = np.append(np.array(list(spacy_tags_index.POS_GLOBAL.keys())),["ROOT"])
    DEP_TAGS = np.append(np.array(list(spacy_tags_index.DEP_EN.keys())),["ROOT"])
    NER_TAGS = np.append(np.array(list(spacy_tags_index.NER.keys())),["ROOT"])
    if language == "english":
        POS_TAGS_EN = np.array(list(spacy_tags_index.POS_EN.keys()))
        POS_TAGS = np.append(POS_TAGS,POS_TAGS_EN)
    if language == "german":
        POS_TAGS_DE = np.array(list(spacy_tags_index.POS_DE.keys()))
        POS_TAGS = np.append(POS_TAGS,POS_TAGS_DE)
        DEP_TAGS_DE = np.array(list(spacy_tags_index.DEP_DE.keys()))
        DEP_TAGS = np.append(DEP_TAGS,DEP_TAGS_DE)

    TAGS_DICT["pos"] = POS_TAGS
    TAGS_DICT["dep"] = DEP_TAGS
    TAGS_DICT["ent"] = NER_TAGS
    TAGS_DICT["tag"] = POS_TAGS #one shared dict as both tag and pos are part-of-speech elements


    #SPACY SEMANTIC FEATURES VOCABULARIES - ONE HOT ENCODED TAGS EMBEDDING MATRIX
    SEMANTIC_FEATURES_EMBEDDINGS = {}
    for f in TAGS_DICT.keys():
        #one hot encoding tags
        tle = TolerantLabelEncoder(ignore_unknown=True)
        tle.fit(TAGS_DICT[f])
        tags_tle = tle.transform(TAGS_DICT[f])
        #add a 0-vector to represent the "not-present/not-decoded" tag
        tags_one_hot = [np.zeros(len(TAGS_DICT[f]))]
        tags_one_hot = np.concatenate((tags_one_hot,to_categorical(tags_tle)),axis=0)
        
        embeddings_params = {}
        embeddings_params['name'] = str(f)
        embeddings_params['encoder'] = tle
        embeddings_params['feats_vocab_len'] = len(tags_one_hot)
        embeddings_params['feats_embedding_dim'] = len(tags_one_hot[0])
        embeddings_params['feats_embedding_matrix'] = tags_one_hot
        embeddings_params['feats_labels'] = TAGS_DICT[f]
        
        SEMANTIC_FEATURES_EMBEDDINGS[f] = embeddings_params
        
    return SEMANTIC_FEATURES_EMBEDDINGS


def get_semantic_cnn_params():
    
    #TODO - get params from JSON in db services table - parameters columns 
       
    #gensim word2vec
    EMBEDDING_PARAMS = {
        'embedding_size': 150,
        'sg_true': 1,
        'window_size': 4,
        'min_count': 2
    }

    #input preprocessing
    PREPROCESSING_PARAMS = {
        'spelling': 0,
        'stemming': 0,
        'lemming': 0,
        'remove_oov': 0,
        'max_length': 400,
        'semantic_features':1
    }

    #semantic features included in input vectors
    SEMANTIC_FEATURES_PARAMS = {
            'pos':1, #coarse-grained part of speech
            'dep':1, #dependency
            'ent':1, #entity type
            'tag':1, #fine-grained part of speech
            'dep_lev':0 #dependency level ("distance" from the parent token/word)
    }
    
    return EMBEDDING_PARAMS, PREPROCESSING_PARAMS, SEMANTIC_FEATURES_PARAMS
    
    
def get_SEMANTIC_CNN_config_params(json_parameters):
    """
    Fetches the SEMANTIC_CNN_config parameter from the 'parameters' json column in services table
    example: "SEMANTIC_CNN_config":
                {    
                    #GENSIM WORD2VEC EMBEDDING PARAMS
                    "EMBEDDING_PARAMS":
                    {
                        'embedding_size': 150,          #vector lenght
                        'sg_true': 1,                   #1: skipgram, 0: cbow 
                        'window_size': 4,               #word2vec window size
                        'min_count': 2                  #minimum count of occurences of a word (over the whole dataset) to be kept in the vocab
                    },
 
                    "PREPROCESSING_PARAMS":
                    {
                        'spelling': 0,                  #enables word spelling autocorrect in text preprocess
                        'stemming': 1,                  #enables stemming in text preprocess
                        'lemming': 1,                   #enables lemming in text preprocess
                        'remove_oov': 1,                #Out Of Vocaboulary words removal (remove words not in the embedding vocab
                        'max_length': 400,              #max number of words for each input text
                        'semantic_features': 1          #enables semantic features
                    },
                    
                    #SEMNATIC FEATURES PARAMS
                    "SEMANTIC_FEATURES_PARAMS":
                    {
                        'pos':1,                        #coarse-grained part of speech
                        'dep':1,                        #dependency
                        'ent':1,                        #entity type
                        'tag':1,                        #fine-grained part of speech
                        'dep_lev':0                     #dependency level ("distance" from the parent token/word)
                    },
                    
                    #MODEL/TRAIN RELATED PARAMS
                    "CNN_MODEL_TRAIN_PARAMS":
                    {    
                        "NB_EPOCHS" : 40,				                        -->  Number of epochs
                        "BATCH_SIZE" : 512, 				                    -->  Size of the batches used in the mini-batch gradient descent    
                        "NB_CONVOLUTION_FILTERS" : 128,		                    -->  Number of convolution filters 
                        "CONVOLUTION_KERNEL_SIZE" : 4,		                    -->  Convolution Kernel Size
                        "LABEL_SMOOTHING" : 0.3,			                    -->  label smoothing index
                        "EARLYSTOPPING_PATIENCE" : 10,		                    -->  number of epochs without improvement in the monitored param that the model waits before stopping
                        "EARLYSTOPPING_MONITOR_PARAM" : "val_loss",		        -->  the value monitored for early stopping 
                        "DROPOUT_PROB" : 0.5				                    -->  dropout CNN index
                    }
                }
                  
        
    The params are not mandatory, if not present the default value is going to be used
    
    :param json_parameters: the json string from the services table
    :return: the SEMANTIC_CNN_config dictionary
    """
    SEMANTIC_CNN_config = get_service_parameter_col(json_parameters,"SEMANTIC_CNN_config")
    if SEMANTIC_CNN_config is None: 
        SEMANTIC_CNN_config = dict()
    if "EMBEDDING_PARAMS" not in SEMANTIC_CNN_config:
        SEMANTIC_CNN_config["EMBEDDING_PARAMS"] = dict()
    if "PREPROCESSING_PARAMS" not in SEMANTIC_CNN_config:
        SEMANTIC_CNN_config["PREPROCESSING_PARAMS"] = dict()
    if "SEMANTIC_FEATURES_PARAMS" not in SEMANTIC_CNN_config:
        SEMANTIC_CNN_config["SEMANTIC_FEATURES_PARAMS"] = dict()
    if "CNN_MODEL_TRAIN_PARAMS" not in SEMANTIC_CNN_config:
        SEMANTIC_CNN_config["CNN_MODEL_TRAIN_PARAMS"] = dict()
       
    EMBEDDING_PARAMS = SEMANTIC_CNN_config["EMBEDDING_PARAMS"]   
    PREPROCESSING_PARAMS = SEMANTIC_CNN_config["PREPROCESSING_PARAMS"]
    SEMANTIC_FEATURES_PARAMS = SEMANTIC_CNN_config["SEMANTIC_FEATURES_PARAMS"]
    CNN_MODEL_TRAIN_PARAMS = SEMANTIC_CNN_config["CNN_MODEL_TRAIN_PARAMS"]
       
    #set config defaults when not set in json
    #GENSIM WORD2VEC EMBEDDING PARAMS
    if "embedding_size" not in EMBEDDING_PARAMS: EMBEDDING_PARAMS["embedding_size"] = 150
    if "sg_true" not in EMBEDDING_PARAMS: EMBEDDING_PARAMS["sg_true"] = 1
    if "window_size" not in EMBEDDING_PARAMS: EMBEDDING_PARAMS["window_size"] = 5
    if "min_count" not in EMBEDDING_PARAMS: EMBEDDING_PARAMS["min_count"] = 2
    
    #INPUT PREPROCESSING PARAMS
    if "spelling" not in PREPROCESSING_PARAMS: PREPROCESSING_PARAMS["spelling"] = 0
    if "stemming" not in PREPROCESSING_PARAMS: PREPROCESSING_PARAMS["stemming"] = 1
    if "lemming" not in PREPROCESSING_PARAMS: PREPROCESSING_PARAMS["lemming"] = 1
    if "remove_oov" not in PREPROCESSING_PARAMS: PREPROCESSING_PARAMS["remove_oov"] = 1
    if "max_length" not in PREPROCESSING_PARAMS: PREPROCESSING_PARAMS["max_length"] = 400
    if "semantic_features" not in PREPROCESSING_PARAMS: PREPROCESSING_PARAMS["semantic_features"] = 1
    
    #SEMNATIC FEATURES PARAMS
    if "pos" not in SEMANTIC_FEATURES_PARAMS: SEMANTIC_FEATURES_PARAMS["pos"] = 1
    if "dep" not in SEMANTIC_FEATURES_PARAMS: SEMANTIC_FEATURES_PARAMS["dep"] = 1
    if "ent" not in SEMANTIC_FEATURES_PARAMS: SEMANTIC_FEATURES_PARAMS["ent"] = 1
    if "tag" not in SEMANTIC_FEATURES_PARAMS: SEMANTIC_FEATURES_PARAMS["tag"] = 1
    if "dep_lev" not in SEMANTIC_FEATURES_PARAMS: SEMANTIC_FEATURES_PARAMS["dep_lev"] = 0
            
    #MODEL/TRAIN RELATED PARAMS
    if "NB_EPOCHS" not in CNN_MODEL_TRAIN_PARAMS: CNN_MODEL_TRAIN_PARAMS["NB_EPOCHS"] = 40
    if "BATCH_SIZE" not in CNN_MODEL_TRAIN_PARAMS: CNN_MODEL_TRAIN_PARAMS["BATCH_SIZE"] = 512
    if "NB_CONVOLUTION_FILTERS" not in CNN_MODEL_TRAIN_PARAMS: CNN_MODEL_TRAIN_PARAMS["NB_CONVOLUTION_FILTERS"] = 128
    if "CONVOLUTION_KERNEL_SIZE" not in CNN_MODEL_TRAIN_PARAMS: CNN_MODEL_TRAIN_PARAMS["CONVOLUTION_KERNEL_SIZE"] = 4
    if "LABEL_SMOOTHING" not in CNN_MODEL_TRAIN_PARAMS: CNN_MODEL_TRAIN_PARAMS["LABEL_SMOOTHING"] = 0.3
    if "EARLYSTOPPING_PATIENCE" not in CNN_MODEL_TRAIN_PARAMS: CNN_MODEL_TRAIN_PARAMS["EARLYSTOPPING_PATIENCE"] = 10
    if "EARLYSTOPPING_MONITOR_PARAM" not in CNN_MODEL_TRAIN_PARAMS: CNN_MODEL_TRAIN_PARAMS["EARLYSTOPPING_MONITOR_PARAM"] = 'val_loss'
    if "DROPOUT_PROB" not in CNN_MODEL_TRAIN_PARAMS: CNN_MODEL_TRAIN_PARAMS["DROPOUT_PROB"] = 0.5
    
    return EMBEDDING_PARAMS, PREPROCESSING_PARAMS, SEMANTIC_FEATURES_PARAMS, CNN_MODEL_TRAIN_PARAMS







#############################################################################################
##################    CATEGORICAL PREDICTION CLASSES AND FUNCTIONS    #######################
#############################################################################################
TEXT_COL = "text_input"

class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)


class MyTfidfVectorizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = TfidfVectorizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x[TEXT_COL])
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x[TEXT_COL]).toarray()

    
def categorical_input_preprocess(dataframe):
    #setup columns type in categorical or text and categorizes labels
    for col in dataframe.columns:
        if len(dataframe[col].unique()) >= 0.7 * len(dataframe[col]): #text column
            dataframe[col] = dataframe[col].astype(str)
            if col != TEXT_COL:
                dataframe[TEXT_COL] = dataframe[TEXT_COL].map(str) + ' ' + dataframe[col].map(lambda x: str(x) if not pd.isna(x) else "undefined" )
        else: #categorical column
            if is_numeric_dtype(dataframe[col]):
                dataframe[col] = dataframe[col].map(lambda x: str(x) if not pd.isna(x) else -1)
            dataframe[col] = dataframe[col].astype(str).astype(CategoricalDtype(categories=dataframe[col].unique())).cat.codes
            dataframe[col] = dataframe[col].astype("category")
    
    return dataframe