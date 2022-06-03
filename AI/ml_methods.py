#***********************************
# SPDX-FileCopyrightText: 2009-2020 Vtenext S.r.l. <info@vtenext.com> and KLONDIKE S.r.l. <info@klondike.ai>
# SPDX-License-Identifier: AGPL-3.0-only
#***********************************

# coding=utf-8
import torch
#torch.multiprocessing.set_start_method('spawn', force=True)#avoid deadlock in spacy pipe

from datetime import date
from scipy.sparse import vstack
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import  SpatialDropout1D, GlobalMaxPooling1D, MaxPooling1D, ZeroPadding1D, Convolution1D
from tensorflow.keras.layers import Embedding, Input, BatchNormalization, Flatten, Dense, Dropout, AlphaDropout, ThresholdedReLU, Activation, concatenate
from tensorflow.keras.optimizers import Adam,Nadam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping,TensorBoard
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import regularizers
from kerastuner.tuners import RandomSearch
from kerastuner import HyperModel
from keras.utils.np_utils import to_categorical
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier as Knn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import *
from utilities.db_handler import *
from utilities.utils import *
from multiprocessing import Manager
from threading import Thread
from django.utils.text import slugify
from tabulate import tabulate
import traceback
import numpy as np
import pandas as pd
import dill as pickle
#import pickle
import os
import os.path
from os import path
import concurrent.futures
import threading
import json
import pdb
import sys

import pprint

from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from feel_it import EmotionClassifier, SentimentClassifier

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")



#########################################################################################################        
#####################                           GLOBALS                          ########################      
#########################################################################################################  



DEBUG_MODE = False
ROOT_MODEL_LABEL = "root"
TRUE_VALUE_SUFFIX = "_trueval"
ERROR_MESSAGE_MODEL_MISSING = "ERROR - MODEL NOT CREATED IN TRAIN"
MODEL_FILENAME_SUFFIX = "_Model_data.pkl"
CNN_MODEL_FILENAME_SUFFIX = "_CNN_model.h5"
SEMANTIC_CNN_MODEL_FILENAME_SUFFIX = "_SEMANTIC_CNN_model.h5"
SEMANTIC_CNN_MODEL_CLASS_FILENAME_SUFFIX = "_SEMANTIC_CNN_class.pkl" 
SEMANTIC_CNN_MODEL_BEST_WEIGHTS_SUFFIX = '_SEMANTIC_CNN_best_weights.h5'
WORD2VEC_VOCAB_FILENAME = 'word2vec_vocab'
WORD2VEC_VOCAB = None

N_JOBS = CORES

PRETRAINED_MODEL_PATH = "./pretrained_models"



#########################################################################################################        
#####################               COMMON CLASSIFIERS METHODS                   ########################     
#########################################################################################################  

# class "override" to get the return value from a thread instead of using 
# a unique shared object for all the parallel threads
class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        #print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,**self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

# class for CNN hyperparams autotuning
class CNNHyperModel(HyperModel):
   
    def __init__(self, num_labels, CNN_config):
        self.num_labels = num_labels
        self.CNN_config = CNN_config

    def build(self, hp):

        model = keras.Sequential()
        model.add(Embedding(input_dim=hp.Int('input_dim',
                                            min_value=5000,
                                            max_value=self.CNN_config["NB_WORDS"],
                                            step = 1000),
                                  output_dim=hp.Int('output_dim',
                                            min_value=200,
                                            max_value=self.CNN_config["EMBEDDING_DIM"],
                                            step = 100),
                                  input_length = self.CNN_config["MAX_LEN"]))
        model.add(Convolution1D(
                    filters=hp.Int('filters',
                                            min_value=32,
                                            max_value=self.CNN_config["NB_CONVOLUTION_FILTERS"],
                                            step = 32),
                    kernel_size=hp.Int('kernel_size',
                                            min_value=3,
                                            max_value=self.CNN_config["CONVOLUTION_KERNEL_SIZE"],
                                            step = 2),
                    padding='same',
                    activation='relu')),
        model.add(BatchNormalization())
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dropout(self.CNN_config["DROPOUT_PROB"]))
        model.add(Dense(units=hp.Int('units',
                                            min_value=64,
                                            max_value=256,
                                            step=32),
                               activation='relu'))
        model.add(Dropout(self.CNN_config["DROPOUT_PROB"]))
        model.add(Dense(self.num_labels, activation='softmax'))
        
        #LABEL SMOOTHING
        loss = CategoricalCrossentropy(label_smoothing=self.CNN_config["LABEL_SMOOTHING"])
        
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate',
                        values=[1e-2, 1e-3, 1e-4])),
                        loss=loss,
                        metrics=['accuracy'])
        
        return model


def random_forest_autotuning_classification(train, test, train_labels, test_labels, save_prefix):
    """

    :param train: training data, iterable/list
    :param test: testing data iterable/list
    :param train_labels: training labels
    :param test_labels: testing labels
    :param res: shared dictionary used for multi-threading
    :return: / --> Saves data in folder "Results"
    """
    print("Training  GridSearch AutoTuning Random Forest Classifier - START: " + str(datetime.datetime.now()))
    
    param_grid = {
        'n_estimators': [50, 80]#, 250],
        #'max_features': [0.25, 0.5]#0.5, 0.75, 1.0, 'sqrt'],
        #'min_samples_split': [2,6]#2,4,6
    }
    
    rf = RandomForestClassifier(random_state=1, n_jobs=N_JOBS)
    rfgs = GridSearchCV(rf, param_grid=param_grid, refit = True, cv=5, n_jobs=N_JOBS)
    
    le = TolerantLabelEncoder(ignore_unknown=True)
    le.fit(train_labels)
    labels_le = le.transform(train_labels)
    
    best_model = rfgs.fit(train, labels_le)

    prediction = best_model.predict(test)
    report, confmat, metrics = report_and_confmat(test_labels, le.inverse_transform(prediction), save_prefix + "_" + "RandomForest")
    #score = best_model.score(test, test_labels)

    res = {}
    res["RandomForestClassifierAutoTuning"] = {"model": best_model, "accuracy": metrics["accuracy"], "name": "RandomForestClassifierAutoTuning", "report": metrics, "confmat": confmat, "LabelEncoder": le}
    print("RandomForest with GridSearch AutoTuning - END: " + str(datetime.datetime.now()))
    return res


def random_forest_classification(train, test, train_labels, test_labels, save_prefix):
    """

    :param train: training data, iterable/list
    :param test: testing data iterable/list
    :param train_labels: training labels
    :param test_labels: testing labels
    :param res: shared dictionary used for multi-threading
    :return: / --> Saves data in folder "Results"
    """
    print("Training  Random Forest Classifier...")
    rand = RandomForestClassifier(n_estimators=70, max_depth=None, n_jobs=N_JOBS)
    
    le = TolerantLabelEncoder(ignore_unknown=True)
    le.fit(train_labels)
    labels_le = le.transform(train_labels)
    
    rand.fit(train, labels_le)

    prediction = rand.predict(test)
    report, confmat, metrics = report_and_confmat(test_labels, le.inverse_transform(prediction), save_prefix + "_" + "RandomForest")
    #score = rand.score(test, test_labels)

    res = {}
    res["RandomForestClassifier"] = {"model": rand, "accuracy": metrics["accuracy"], "name": "RandomForestClassifier", "report": metrics, "confmat": confmat, "LabelEncoder": le}
    print("RandomForest ended...")
    return res


def SVC_classification(train, test, train_labels, test_labels, save_prefix):
    """

    :param train: training data, iterable/list
    :param test: testing data, iterable/list
    :param train_labels: training labels, iterable/list
    :param test_labels: testing labels, iterable/list
    :param res: shared dictionary used for multi-threading
    :return: / --> Saves data in folder "Results"
    """
    print("Training  SVC...")

    svc = SVC(kernel='poly', gamma='scale')

    le = TolerantLabelEncoder(ignore_unknown=True)
    le.fit(train_labels)
    labels_le = le.transform(train_labels)

    svc.fit(train, labels_le)

    prediction = svc.predict(test)
    report, confmat, metrics = report_and_confmat(test_labels, le.inverse_transform(prediction),  save_prefix + "_" + "SVC")
    #score = svc.score(test, test_labels)
    res = {}
    res["SVC"] = {"model": svc, "accuracy": metrics["accuracy"], "name": "SVC", "report": metrics, "confmat": confmat, "LabelEncoder": le}
    print("SVC ended...")
    return res


def LinearSVC_classification(train, test, train_labels, test_labels, save_prefix):
    """
    :param train: training data, iterable/list
    :param test: testing data, iterable/list
    :param train_labels: training labels, iterable/list
    :param test_labels: testing labels, iterable/list
    :param res: shared dictionary used for multi-threading
    :return: / --> Saves data in folder "Results"
    """
    print("Training  LinearSVC...")

    model_svc = LinearSVC()
    linear_svc = CalibratedClassifierCV(model_svc) 
  
    le = TolerantLabelEncoder(ignore_unknown=True)
    le.fit(train_labels)
    labels_le = le.transform(train_labels)

    linear_svc.fit(train, labels_le)

    prediction = linear_svc.predict(test)
    report, confmat, metrics = report_and_confmat(test_labels, le.inverse_transform(prediction),  save_prefix + "_" + "LinearSVC")
    #score = linear_svc.score(test, test_labels)
    
    res = {}
    res["LinearSVC"] = {"model": linear_svc, "accuracy": metrics["accuracy"], "name": "LinearSVC", "report": metrics, "confmat": confmat, "LabelEncoder": le}
    print("LinearSVC ended...")
    return res
    
    
def LinearSVC_autotuning_classification(train, test, train_labels, test_labels, save_prefix):
    """

    :param train: training data, iterable/list
    :param test: testing data, iterable/list
    :param train_labels: training labels, iterable/list
    :param test_labels: testing labels, iterable/list
    :param res: shared dictionary used for multi-threading
    :return: / --> Saves data in folder "Results"
    """
    print("Training  GridSearch AutoTuning LinearSVC - START: " + str(datetime.datetime.now()))

    param_grid = {'C': [0.00001, 0.0001, 0.0005],
                  'dual': (True, False), 'random_state': [42]
                 }
    
    ls = LinearSVC()
    lsgs = GridSearchCV(ls, param_grid=param_grid, refit = True, cv=5, n_jobs=N_JOBS)
   
    le = TolerantLabelEncoder(ignore_unknown=True)
    le.fit(train_labels)
    labels_le = le.transform(train_labels)
    
    best_model = lsgs.fit(train, labels_le)

    prediction = best_model.predict(test)
    report, confmat, metrics = report_and_confmat(test_labels, le.inverse_transform(prediction),  save_prefix + "_" + "LinearSVC")
    #score = best_model.score(test, test_labels)
    res = {}
    res["LinearSVCAutoTuning"] = {"model": best_model, "accuracy": metrics["accuracy"], "name": "LinearSVCAutoTuning", "report": metrics, "confmat": confmat, "LabelEncoder": le}
    print("LinearSVC GridSearch AutoTuning - END: " + str(datetime.datetime.now()))
    return res


def MultinomialNB_classification(train, test, train_labels, test_labels, save_prefix):
    """

    :param train: training data, iterable/list
    :param test: testing data, iterable/list
    :param train_labels: training labels, iterable/list
    :param test_labels: testing labels, iterable/list
    :return: / --> Saves data in folder "Results"
    """
    multiNB = MultinomialNB()
    
    le = TolerantLabelEncoder(ignore_unknown=True)
    le.fit(train_labels)
    labels_le = le.transform(train_labels)
    
    multiNB.fit(train, labels_le)

    prediction = multiNB.predict(test)
    report, confmat, metrics = report_and_confmat(test_labels, le.inverse_transform(prediction),  save_prefix + "_" + "MultinomialNB")
    #score = multiNB.score(test, test_labels)
    res = {}
    res["MultinomialNB"] = {"model": multiNB, "accuracy": metrics["accuracy"], "name": "MultinomialNB", "report": metrics, "confmat": confmat, "LabelEncoder": le}
    print("Multinomial ended...")
    return res


def ComplementNB_classification(train, test, train_labels, test_labels, save_prefix):
    """

    :param train: training data, iterable/list
    :param test: testing data, iterable/list
    :param train_labels: training labels, iterable/list
    :param test_labels: testing labels, iterable/list
    :return: / --> Saves data in folder "Results"
    """
    print("Training  Complement Nive Bayes...")

    complNB = ComplementNB()

    le = TolerantLabelEncoder(ignore_unknown=True)
    le.fit(train_labels)
    labels_le = le.transform(train_labels)
    
    complNB.fit(train, labels_le)

    prediction = complNB.predict(test)
    report, confmat, metrics = report_and_confmat(test_labels, le.inverse_transform(prediction),  save_prefix + "_" + "ComplementNB")
    #score = complNB.score(test, test_labels)
    res = {}
    res["ComplementNB"] = {"model": complNB, "accuracy": metrics["accuracy"], "name": "ComplementNB", "report": metrics, "confmat": confmat, "LabelEncoder": le}
    print("Complement ended...")
    return res


def BernoulliNB_classification(train, test, train_labels, test_labels, save_prefix):
    """

    :param train: training data, iterable/list
    :param test: testing data, iterable/list
    :param train_labels: training labels, iterable/list
    :param test_labels: testing labels, iterable/list
    :return: / --> Saves data in folder "Results"
    """
    print("Training  Bernoulli Nive Bayes...")

    bernNB = BernoulliNB(alpha=0.7)

    le = TolerantLabelEncoder(ignore_unknown=True)
    le.fit(train_labels)
    labels_le = le.transform(train_labels)

    bernNB.fit(train, labels_le)

    prediction = bernNB.predict(test)
    report, confmat, metrics = report_and_confmat(test_labels, le.inverse_transform(prediction),  save_prefix + "_" + "BernoulliNB")
    #score = bernNB.score(test, test_labels)
    res = {}
    res["BernoulliNB"] = {"model": bernNB, "accuracy": metrics["accuracy"], "name": "BernoulliNB", "report": metrics, "confmat": confmat, "LabelEncoder": le}
    print("Bernoulli ended...")

    return res


def GradientBoosting_classification(train, test, train_labels, test_labels, save_prefix):
    """

    :param train: training data, iterable/list
    :param test: testing data, iterable/list
    :param train_labels: training labels, iterable/list
    :param test_labels: testing labels, iterable/list
    :return: / --> Saves data in folder "Results"
    """
    print("Training  Gradient Boosting...")

    gradb = GradientBoostingClassifier(n_estimators=100)

    le = TolerantLabelEncoder(ignore_unknown=True)
    le.fit(train_labels)
    labels_le = le.transform(train_labels)

    gradb.fit(train, labels_le)

    prediction = gradb.predict(test)
    report, confmat, metrics = report_and_confmat(test_labels, le.inverse_transform(prediction),  save_prefix + "_" + "GradientBoosting")
    #score = gradb.score(test, test_labels)
    res = {}    
    res["GradientBoostingClassifier"] = {"model": gradb, "accuracy": metrics["accuracy"], "name": "GradientBoostingClassifier", "report": metrics, "confmat": confmat, "LabelEncoder": le}
    print("GradientBoosting ended...")

    return res


def AdaBoost_classification(train, test, train_labels, test_labels, save_prefix):
    """

    :param train: training data, iterable/list
    :param test: testing data, iterable/list
    :param train_labels: training labels, iterable/list
    :param test_labels: testing labels, iterable/list
    :return: / --> Saves data in folder "Results"
    """
    print("Training  AdaBoost...")

    # Uso l'svc perche' e' quello che funziona meglio per ora
    Linsvc = LinearSVC()

    adab = AdaBoostClassifier(base_estimator=Linsvc, algorithm='SAMME', n_estimators=50)

    le = TolerantLabelEncoder(ignore_unknown=True)
    le.fit(train_labels)
    labels_le = le.transform(train_labels)

    adab.fit(train, labels_le)

    prediction = adab.predict(test)
    report, confmat, metrics = report_and_confmat(test_labels, le.inverse_transform(prediction),  save_prefix + "_" + "AdaBoost")
    #score = adab.score(test, test_labels)
    print("Adaboost ended...")
    res = {}    
    res["AdaBoostClassifier"] = {"model": adab, "accuracy": metrics["accuracy"], "name": "AdaBoostClassifier", "report": metrics, "confmat": confmat, "LabelEncoder": le}

    return res


def AdaBoost_autotuning_classification(train, test, train_labels, test_labels, save_prefix):
    """

    :param train: training data, iterable/list
    :param test: testing data, iterable/list
    :param train_labels: training labels, iterable/list
    :param test_labels: testing labels, iterable/list
    :return: / --> Saves data in folder "Results"
    """
    print("Training  GridSearch AutoTuning AdaBoost - START: " + str(datetime.datetime.now()))

    # Uso l'svc perche' e' quello che funziona meglio per ora
    Linsvc = LinearSVC()
    
    param_grid = {'n_estimators': (25, 50,100),
                  #'base_estimator__C': [0.00001,0.0005],#[0.00001, 0.0001, 0.0005]
                  'algorithm': ['SAMME']#,'SAMME.R']
                 }
    
    ab = AdaBoostClassifier(base_estimator=Linsvc)
    abgs = GridSearchCV(ab, param_grid=param_grid, refit = True, cv=5, n_jobs=N_JOBS)

    le = TolerantLabelEncoder(ignore_unknown=True)
    le.fit(train_labels)
    labels_le = le.transform(train_labels)

    best_model = abgs.fit(train, labels_le)

    prediction = best_model.predict(test)
    report, confmat, metrics = report_and_confmat(test_labels, le.inverse_transform(prediction),  save_prefix + "_" + "AdaBoost")
    #score = best_model.score(test, test_labels)
    print("Adaboost GridSearch AutoTuning - END: " + str(datetime.datetime.now()))
    res = {}    
    res["AdaBoostClassifierAutoTuning"] = {"model": best_model, "accuracy": metrics["accuracy"], "name": "AdaBoostClassifierAutoTuning", "report": metrics, "confmat": confmat, "LabelEncoder": le}

    return res


def VotingClassifier_classification(train, test, train_labels, test_labels, save_prefix):
    """

    :param train: training data, iterable/list
    :param test: testing data, iterable/list
    :param train_labels: training labels, iterable/list
    :param test_labels: testing labels, iterable/list
    :return: / --> Saves data in folder "Results"
    """
    print("Training  Voting classifier...")

    cl1 = LogisticRegression(max_iter=250, multi_class='auto')
    cl6 = MultinomialNB()
    cl3 = AdaBoostClassifier(base_estimator=cl1, algorithm='SAMME', n_estimators=200)
    cl4 = GradientBoostingClassifier()
    cl5 = ComplementNB()
    cl7 = Knn(algorithm='auto', n_neighbors=5)
    cl8 = RandomForestClassifier(n_estimators=70, max_depth=None)
    cl9 = ExtraTreesClassifier()

    vote = VotingClassifier(estimators=[('LogisticReg', cl1), ('AdaBoost', cl3), ('GradBoost', cl4),
                            ('ComplementNB', cl5), ('MultinomialNB', cl6), ('Knn', cl7), ('RandomForest', cl8),
                            ('ExtraTree', cl9)], voting='soft', n_jobs=N_JOBS)

    le = TolerantLabelEncoder(ignore_unknown=True)
    le.fit(train_labels)
    labels_le = le.transform(train_labels)

    vote.fit(train, labels_le, n_jobs=N_JOBS)

    prediction = vote.predict(test)
    report, confmat, metrics = report_and_confmat(test_labels, le.inverse_transform(prediction),  save_prefix + "_" + "VotingClass")
    #score = vote.score(test, test_labels)
    print("Voting ended...")
    res = {}
    res["VotingClassifier"] = {"model": vote, "accuracy": metrics["accuracy"], "name": "VotingClassifier", "report": metrics, "confmat": confmat, "LabelEncoder": le}

    return res 


def LogisticRegression_classification(train, test, train_labels, test_labels, save_prefix):
    """

    :param train: training data, iterable/list
    :param test: testing data, iterable/list
    :param train_labels: training labels, iterable/list
    :param test_labels: testing labels, iterable/list
    :return: / --> Saves data in folder "Results"
    """

    print("Training  LogisticRegression...")

    # TODO CONTROLLARE I SOLVER DIVERSI
    reg = LogisticRegression(max_iter=250, multi_class='multinomial', solver='newton-cg', n_jobs=N_JOBS)

    le = TolerantLabelEncoder(ignore_unknown=True)
    le.fit(train_labels)
    labels_le = le.transform(train_labels)

    reg.fit(train, labels_le)

    prediction = reg.predict(test)
    report, confmat, metrics = report_and_confmat(test_labels, le.inverse_transform(prediction),  save_prefix + "_" + "LogisticReg")
    #score = reg.score(test, test_labels)
    res = {}
    res["LogisticRegression"] = {"model": reg, "accuracy": metrics["accuracy"], "name": "LogisticRegression", "report": metrics, "confmat": confmat, "LabelEncoder": le}
    print("Logistic Regression ended...")
    return res


def LogisticRegression_autotuning_classification(train, test, train_labels, test_labels, save_prefix):
    """

    :param train: training data, iterable/list
    :param test: testing data, iterable/list
    :param train_labels: training labels, iterable/list
    :param test_labels: testing labels, iterable/list
    :return: / --> Saves data in folder "Results"
    """

    print("Training  GridSearch AutoTuning LogisticRegression - START: " + str(datetime.datetime.now()))

    param_grid = {
                    'C': [0.001,1,100],
                    'solver': ['newton-cg','saga', 'lbfgs'],
                    'multi_class': ['ovr', 'multinomial'],
                    #'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    #'penalty': ['l1', 'l2'],
                    #'max_iter': list(range(100,800,100)),
                    #'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                    #'multi_class': ['multinomial']
                }
    
    lr = LogisticRegression()
    lrgs = GridSearchCV(lr, param_grid=param_grid, refit = True, cv=5, n_jobs=N_JOBS)

    le = TolerantLabelEncoder(ignore_unknown=True)
    le.fit(train_labels)
    labels_le = le.transform(train_labels)

    best_model = lrgs.fit(train, labels_le)

    prediction = best_model.predict(test)
    report, confmat, metrics = report_and_confmat(test_labels, le.inverse_transform(prediction),  save_prefix + "_" + "LogisticReg")
    #score = best_model.score(test, test_labels)
    res = {}
    res["LogisticRegressionAutoTuning"] = {"model": best_model, "accuracy": metrics["accuracy"], "name": "LogisticRegressionAutoTuning", "report": metrics, "confmat": confmat, "LabelEncoder": le}
    print("Logistic Regression GridSearch AutoTuning - END: " + str(datetime.datetime.now()))
    return res


def ExtrExtraTrees_classification(train, test, train_labels, test_labels, save_prefix):
    """

    :param train: training data, iterable/list
    :param test: testing data, iterable/list
    :param train_labels: training labels, iterable/list
    :param test_labels: testing labels, iterable/list
    :return: / --> Saves data in folder "Results"
    """
    print("Training  ExtraTrees...")

    extra = ExtraTreesClassifier(n_jobs=N_JOBS)

    le = TolerantLabelEncoder(ignore_unknown=True)
    le.fit(train_labels)
    labels_le = le.transform(train_labels)

    extra.fit(train, labels_le)
    prediction = extra.predict(test)

    report, confmat, metrics = report_and_confmat(test_labels, le.inverse_transform(prediction),  save_prefix + "_" + "ExtraTrees")
    #score = extra.score(test, test_labels)
    res = {}
    res["ExtraTrees"] = {"model": extra, "accuracy": metrics["accuracy"], "name": "ExtraTreesClassifier", "report": metrics, "confmat": confmat, "LabelEncoder": le}
    print("ExtraTrees ended...")
    return res


def ExtrExtraTrees_autotuning_classification(train, test, train_labels, test_labels, save_prefix):
    """

    :param train: training data, iterable/list
    :param test: testing data, iterable/list
    :param train_labels: training labels, iterable/list
    :param test_labels: testing labels, iterable/list
    :return: / --> Saves data in folder "Results"
    """
    print("Training  GridSearch AutoTuning ExtraTrees - START: " + str(datetime.datetime.now()))

    param_grid = { 'n_estimators': [16, 32, 64] }
     
    et = ExtraTreesClassifier(n_jobs=N_JOBS)
    etgs = GridSearchCV(et, param_grid=param_grid, refit = True, cv=5, n_jobs=N_JOBS)

    le = TolerantLabelEncoder(ignore_unknown=True)
    le.fit(train_labels)
    labels_le = le.transform(train_labels)

    best_model = etgs.fit(train, labels_le)
    prediction = etgs.predict(test)

    report, confmat, metrics = report_and_confmat(test_labels, le.inverse_transform(prediction),  save_prefix + "_" + "ExtraTrees")
    #score = etgs.score(test, test_labels)
    res = {}
    res["ExtraTreesAutoTuning"] = {"model": etgs, "accuracy": metrics["accuracy"], "name": "ExtraTreesAutoTuning", "report": metrics, "confmat": confmat, "LabelEncoder": le}
    print("ExtraTrees GridSearch AutoTuning - END: " + str(datetime.datetime.now()))
    return res


def multigroup_normconstraint_cnn(tokenizer, num_labels, CNN_config):

    # EMBEDDINGS FILE 
    folder = 'embeddings/'
    GLOVE_EMBEDDING_DIM = CNN_config["MG_GLOVE_EMB_DIM"]
    FASTTEXT_EMBEDDING_DIM = CNN_config["MG_FASTTEXT_EMB_DIM"]
    FASTTEXT_EMBEDDINGS_FILE = folder + CNN_config["MG_FASTTEXT_EMB_FILE"] #'wiki-news-300d-1M.vec' for ENG
    GLOVE_EMBEDDINGS_FILE = folder + CNN_config["MG_GLOVE_EMB_FILE"] #'glove.6B.100d.txt' for ENG
    
    if DEBUG_MODE: pdb.set_trace()
    embeddings = read_embedding(GLOVE_EMBEDDINGS_FILE)

    fasttext_embeddings = read_embedding(FASTTEXT_EMBEDDINGS_FILE)
    # 100-d GloVe embeddings
    trained_embeddings1 = create_embedding_matrix(tokenizer, GLOVE_EMBEDDING_DIM, embeddings)
    # 100-d GloVe embeddings
    trained_embeddings2 = create_embedding_matrix(tokenizer, GLOVE_EMBEDDING_DIM, embeddings)
    # 300-d fastText embeddings
    trained_embeddings3 = create_embedding_matrix(tokenizer, FASTTEXT_EMBEDDING_DIM, fasttext_embeddings)
    
    
    text_seq_input = Input(shape=(CNN_config["MAX_LEN"],), dtype='int32')
    text_embedding1 = Embedding(
        trained_embeddings1.shape[0],
        GLOVE_EMBEDDING_DIM,
        weights=[trained_embeddings1],
        input_length=CNN_config["MAX_LEN"],
        trainable=True)(text_seq_input)
    text_embedding2 = Embedding(
        trained_embeddings2.shape[0],
        GLOVE_EMBEDDING_DIM,
        weights=[trained_embeddings2],
        input_length=CNN_config["MAX_LEN"],
        trainable=False)(text_seq_input)
    text_embedding3 = Embedding(
        trained_embeddings3.shape[0],
        FASTTEXT_EMBEDDING_DIM,
        weights=[trained_embeddings3],
        input_length=CNN_config["MAX_LEN"],
        trainable=True)(text_seq_input)

    k_top = 4
    filter_sizes = [3, 5]

    conv_pools = []
    for text_embedding in [text_embedding1, text_embedding2, text_embedding3]:
        for filter_size in filter_sizes:
            l_zero = ZeroPadding1D(
                (filter_size - 1, filter_size - 1))(text_embedding)
            l_conv = Convolution1D(
                filters=16,
                kernel_size=filter_size,
                padding='same',
                activation='tanh')(l_zero)
            l_pool = GlobalMaxPooling1D()(l_conv)
            conv_pools.append(l_pool)

    l_merge = concatenate(conv_pools, axis=1)
    l_dense = Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001))(l_merge)
    l_out = Dense(num_labels, activation='softmax')(l_dense)
    model = Model(inputs=[text_seq_input], outputs=l_out, name = 'MultiGroupNormConstraintCNN')
    
    return model


def semantic_cnn_classification(X_train, Y_train, X_test, Y_test, language, save_prefix, retrain_over_full_dataset, parameters, pretrained_word2vec_vocab = None):
    
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("--- Training  SEMANTIC_CNN...")

    try:
        reset_keras()
        
        spacy_model = load_spacy(language)
        if spacy_model is None: #unsupported languages
            return None
        
        X_train_noval, X_val, Y_train_noval, Y_val = train_test_split(X_train, Y_train,  test_size = 0.1, random_state = 42, stratify=Y_train)
        
        embedding_params, preprocessing_params, semantic_features_params, cnn_model_train_params = get_SEMANTIC_CNN_config_params(parameters)
        semantic_features_embeddings = get_semantic_features_embeddings(language)

        save_path = os.path.join(os.getcwd(), "Models")
        w2v_filename = os.path.join(save_path, SAVE_PREFIX_GLOBAL + "_" + WORD2VEC_VOCAB_FILENAME + "_" + language)
        if pretrained_word2vec_vocab != None:
            embedding_vocab = pretrained_word2vec_vocab
        else:
            embedding_vocab = generate_word2vec_vocab(X_train, embedding_params, preprocessing_params, spacy_model, language, save_filename=w2v_filename)
        
        scnn = semantic_cnn(Y_train_noval, X_val, Y_val, language, spacy_model, embedding_vocab, embedding_params, preprocessing_params, semantic_features_params, cnn_model_train_params, semantic_features_embeddings)
        scnn.checkpoint_file = "./Checkpoints/" + save_prefix + SEMANTIC_CNN_MODEL_BEST_WEIGHTS_SUFFIX
        
        print("--- GPU AVAILABLE MEMORY: ",get_gpu_memory(),"MB ---")
        scnn.compile()
        xtrain,ytrain = scnn.input_preprocess(X_train_noval, Y_train_noval)
        scnn.fit(xtrain,ytrain)

        preds = scnn.probs_to_labels(scnn.predict(scnn.input_preprocess(X_test)))
        report, confmat, metrics = report_and_confmat(Y_test, preds, save_prefix + "_" + "SEMANTIC_CNN_")
        accuracy = metrics["accuracy"]
        scnn.accuracy = accuracy

        if retrain_over_full_dataset is True:
            print("---------------------------------------------------------------------------------")
            print("--- RETRAINING THE SEMANTIC CNN MODEL ON THE FULL DATASET (train+test+validation)")
            print("---------------------------------------------------------------------------------")
            X_train_full = pd.concat([X_train, X_test], ignore_index=True)#np.concatenate([X_train , X_test])
            Y_train_full = pd.concat([Y_train, Y_test], ignore_index=True)#np.concatenate([Y_train , Y_test])

            """
            if pretrained_word2vec_vocab != None:
                embedding_vocab = pretrained_word2vec_vocab
            else:
                embedding_vocab = generate_word2vec_vocab(X_train_full, embedding_params, preprocessing_params, spacy_model, language, save_filename=w2v_filename)
            """
            scnn = semantic_cnn(Y_train_full, None, None, language, spacy_model, embedding_vocab, embedding_params, preprocessing_params, semantic_features_params, cnn_model_train_params, semantic_features_embeddings)
            scnn.checkpoint_file = "./Checkpoints/" + save_prefix + SEMANTIC_CNN_MODEL_BEST_WEIGHTS_SUFFIX
            scnn.compile()
            xtrain , ytrain = scnn.input_preprocess(X_train_full, Y_train_full)
            scnn.fit(xtrain, ytrain, validation=False)


        res = {}
        res["SEMANTIC_CNN"] = {"model": scnn, "accuracy": accuracy, "name": "SEMANTIC_CNN", "report": metrics, "confmat": confmat, "LabelEncoder": scnn.le}
        
    except Exception as e:
        print("--- ERROR --- in method semantic_cnn")
        print(e)
        traceback.print_exc()
        return None
        
    print("--- SEMANTIC_CNN ended...")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    return res






#data and params for retrain over full dataset
CNN_dataset_full = None
CNN_labels_full = None
NB_EPOCHS = 30
BATCH_SIZE = 512
checkpoints = []

CNN_FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

def CNN_classification(train_labels, test_labels, data_train = None, data_test = None, save_prefix = None, parameters = None, train_over_full_dataset = False):
    """

    :param train: training data, iterable/list
    :param test: testing data, iterable/list
    :param train_labels: training labels, iterable/list
    :param test_labels: testing labels, iterable/list
    :param data_train: train dataset
    :param data_test: test dataset
    :param save_prefix: cron and services tables ids used to name the model and weights files
    :param parameters: the parameters from services table
    :return: / --> Saves data in folder "Results"
    """
    
    #made global in case of retraining the CNN model over the full dataset
    global NB_EPOCHS
    global BATCH_SIZE
    global CNN_labels_full
    global CNN_dataset_full
    global checkpoints
    
    ##################################################
    # Parameters for CNN model training and prediction
    ##################################################
    CNN_config = get_CNN_config_params(parameters)

    NB_WORDS = CNN_config["NB_WORDS"]                                       # default: 10000        # number of words in the dictionary
    NB_EPOCHS = CNN_config["NB_EPOCHS"]                                     # default: 30           # Number of epochs
    BATCH_SIZE = CNN_config["BATCH_SIZE"]                                   # default: 512          # Size of the batches used in the mini-batch gradient descent    
    MAX_LEN = CNN_config["MAX_LEN"]                                         # default: 400          # Maximum number of words in a sequence
    EMBEDDING_DIM = CNN_config["EMBEDDING_DIM"]                             # default: 150          # Number of dimensions of the GloVe word embeddings
    NB_CONVOLUTION_FILTERS = CNN_config["NB_CONVOLUTION_FILTERS"]           # default: 128          # Number of convolution filters 
    CONVOLUTION_KERNEL_SIZE = CNN_config["CONVOLUTION_KERNEL_SIZE"]         # default: 4            # Convolution Kernel Size
    LABEL_SMOOTHING = CNN_config["LABEL_SMOOTHING"]                         # default: 0.3          # label smoothing index
    EARLYSTOPPING_PATIENCE = CNN_config["EARLYSTOPPING_PATIENCE"]           # default: 10           # number of epochs without improvement in the monitored param that the model waits before stopping
    EARLYSTOPPING_MONITOR_PARAM = CNN_config["EARLYSTOPPING_MONITOR_PARAM"] # default: "val_loss"   # the value monitored for early stopping 
    DROPOUT_PROB = CNN_config["DROPOUT_PROB"]                               # default: 0.5          # dropout CNN index
    PARAMS_AUTOTUNING = CNN_config["PARAMS_AUTOTUNING"]                     # default: False        # enables CNN hyperparams autotuning via Keras tuner class
    MULTIGROUP_CNN = CNN_config["MULTIGROUP_CNN"]                           # default: False        # enables CNN MultiGroup custom embeddings mode


    
    #np.set_printoptions(threshold=sys.maxsize)
    print("Training  CNN...")
    print("LABEL TEST: ")
    print(np.unique(np.array(list(test_labels))))
    print("LABEL TRAIN: ")
    print(np.unique(np.array(list(train_labels))))
    print("CNN CONFIGURATION PARAMS:")
    print(CNN_config)

    try: 
        # Tokenize the dataset on training data
        # tokenization with max words defined and filters
        tk = Tokenizer(num_words=NB_WORDS,
                       filters=CNN_FILTERS,
                       lower=True,
                       split=" ")
        tk.fit_on_texts(data_train)

        # Sentence Distribution in training data
        # understand the sequence distribution for max length 
        seq_lengths = data_train.apply(lambda x: len(x.split(' ')))
        seq_lengths.describe()

        # Convert Train and Test to fixed length sequences
        X_train_entire_seq_tok = tk.texts_to_sequences(data_train)
        X_test_seq_tok = tk.texts_to_sequences(data_test)

        # pad the sequences 
        X_train_entire_seq = pad_sequences(X_train_entire_seq_tok, maxlen=MAX_LEN)
        X_test_seq = pad_sequences(X_test_seq_tok, maxlen=MAX_LEN)

        # perform encoding of 
        le = TolerantLabelEncoder(ignore_unknown=True)
        y_train_le = le.fit_transform(train_labels)
        y_test_le = le.transform(test_labels)
        y_train_one_hot = to_categorical(y_train_le)
        y_test_one_hot = to_categorical(y_test_le)

        # labels from encoder mapping 
        TARGET_TEXT_LABELS = le.classes_
        TEXT_LABELS = le.classes_

        # Validation Dataset from Training
        proportional = True
        stratify_param = get_service_parameter_col(parameters,"stratify_split")
        if stratify_param is not None and stratify_param is False:
            proportional = False
        
        stratify = None
        if proportional:
            stratify = y_train_one_hot
        X_train_seq, X_valid_seq, y_train, y_valid = train_test_split(
        X_train_entire_seq, y_train_one_hot, test_size=0.25, stratify=stratify, random_state=42)

        assert X_valid_seq.shape[0] == y_valid.shape[0]
        assert X_train_seq.shape[0] == y_train.shape[0]

        print('Shape of training set:', X_train_seq.shape)
        print('Shape of validation set:', X_valid_seq.shape)

    except Exception as e:
        errfile = os.path.join(os.path.join(os.getcwd(), "logs"), save_prefix + "_TOK_ERROR_DUMP_CNN_vars.txt")
        print("--- ERROR --- in method CNN_classification in TOKENIZING - check local variables DUMP into file:  " + str(errfile))
        print(e)
        traceback.print_exc()
        with open(errfile, "w") as file:
            file.write(str(e) + "\n")
            pprint.pprint(locals(),file)

    try:
        # uses the kerastuner lib to autotune the CNN model hyperparams 
        # and returns the best model based on val_accuracy score
        if PARAMS_AUTOTUNING:
            print("========= CNN AUTOTUNING ENABLED =========")
            hypermodel = CNNHyperModel(len(y_train[0]),CNN_config)
        
            tuner = RandomSearch(
                hypermodel,
                objective='val_accuracy',
                max_trials=5,
                executions_per_trial=3,
                project_name='CNN_tuning')
            
            tuner.search_space_summary()

            tuner.search(X_train_seq, y_train,
                            epochs=NB_EPOCHS,
                            batch_size=BATCH_SIZE,
                            validation_data=(
                                X_valid_seq,
                                y_valid),
                            verbose=1)

            model = tuner.get_best_models(num_models=1)[0]
            tuner.results_summary()

        # runs the CNN model with hyperparams values set in CNN_config 
        # json in services table or default values
        else:
            if MULTIGROUP_CNN:
                print("========= USING MULTIGROUP CNN =========")
                model = multigroup_normconstraint_cnn(tk, len(y_train[0]), CNN_config)

            else:
                print("========= USING STANDARD CNN =========")
                model = Sequential(
                    [
                        Embedding(
                            input_dim=NB_WORDS,
                            output_dim=EMBEDDING_DIM,
                            input_length=MAX_LEN),
                        Convolution1D(
                            filters=NB_CONVOLUTION_FILTERS,
                            kernel_size=CONVOLUTION_KERNEL_SIZE,
                            padding='same',
                            activation='relu'),
                        BatchNormalization(),
                        MaxPooling1D(),
                        Flatten(),
                        Dropout(DROPOUT_PROB),
                        Dense(
                            100,
                            activation='relu'),
                        Dropout(DROPOUT_PROB),
                        Dense(
                            len(y_train[0]),
                            activation='softmax')],name='CNNClassifier')

            #LABEL SMOOTHING
            loss = CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)

            checkpoints = []
            model.compile(
                optimizer=Nadam(),
                loss=loss, # 'categorical_crossentropy',
                metrics=['accuracy'])
            
            checkpoint_file = './Checkpoints/'+ save_prefix +'_Train-' + model.name + '-best_weights.h5'
            
            checkpoints.append(
                ModelCheckpoint(
                    checkpoint_file,
                    monitor=EARLYSTOPPING_MONITOR_PARAM,
                    verbose=0,
                    save_best_only=True,
                    save_weights_only=True,
                    mode='auto',
                    period=1))
            
            checkpoints.append(
                TensorBoard(
                    log_dir='./logs',
                    histogram_freq=0,
                    write_graph=True,
                    write_images=False,
                    embeddings_freq=0,
                    embeddings_layer_names=None,
                    embeddings_metadata=None))
            
            checkpoints.append(EarlyStopping(monitor=EARLYSTOPPING_MONITOR_PARAM, patience=EARLYSTOPPING_PATIENCE, restore_best_weights=True))
               
            history = model.fit(
                X_train_seq,
                y_train,
                epochs=NB_EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=(
                    X_valid_seq,
                    y_valid),
                verbose=1,
                callbacks=checkpoints)#,workers=6,use_multiprocessing=True)
            #load the weights from the checkpoint file, the best weights
            model.load_weights(checkpoint_file)
        
    except Exception as e:
        errfile = os.path.join(os.path.join(os.getcwd(), "logs"), save_prefix + "_FIT_ERROR_DUMP_CNN_vars.txt")
        print("--- ERROR --- in method CNN_classification in MODEL FIT - check local variables DUMP into file:  " + str(errfile))
        print(e)
        traceback.print_exc()
        with open(errfile, "w") as file:
            file.write(str(e) + "\n")
            pprint.pprint(locals(),file)
    
    print("MODEL SUMMARY: ")
    print(model.summary())
    
    report = confmat = metrics = score = None
    
    try:
        results = model.evaluate(X_test_seq, y_test_one_hot)#,workers=6,use_multiprocessing=True)
        y_softmax = model.predict(X_test_seq)#,workers=6,use_multiprocessing=True)
        y_class_index = []
        y_pred_index = []
        confidence = []

        for i in range(len(y_test_one_hot)):
            probs = y_test_one_hot[i]
            index_arr = np.nonzero(probs)
            one_hot_index = index_arr[0].item(0)
            y_class_index.append(one_hot_index)

        for i in range(0, len(y_softmax)):
            probs = y_softmax[i]
            predicted_index = np.argmax(probs)
            y_pred_index.append(TEXT_LABELS[predicted_index])
            confidence.append(probs[predicted_index])
             
        average_precision = average_precision_score(
            y_test_one_hot, y_softmax, average='weighted')
             
        report, confmat, metrics = report_and_confmat(test_labels, y_pred_index,  save_prefix + "_" + "CNN")
        score = results[1]

    except Exception as e:
        errfile = os.path.join(os.path.join(os.getcwd(), "logs"), save_prefix + "_EVAL_ERROR_DUMP_CNN_vars.txt")
        print("--- ERROR --- in method CNN_classification in MODEL EVALUATE - check local variables DUMP into file:  " + str(errfile))
        print(e)
        traceback.print_exc()
        with open(errfile, "w") as file:
            file.write(str(e) + "\n")
            pprint.pprint(locals(),file)

    
    #save data for retraining the best model over the whole dataset instead of splitting in train and test subdatasets
    if train_over_full_dataset is not None and train_over_full_dataset is True:
        CNN_dataset_full = np.concatenate([ X_train_seq , X_valid_seq ])
        CNN_labels_full = np.concatenate([ y_train , y_valid ])
    
    # save model to disk
    save_path = os.path.join(os.getcwd(), "Models")
    filename = os.path.join(save_prefix + CNN_MODEL_FILENAME_SUFFIX)
    model.save(os.path.join(save_path, filename))
    res = {}
    res["CNN"] = {"model": model.to_json(), "accuracy": score, "name": "CNNClassifier", "report": metrics, "confmat": confmat, "CNN_model_file": filename, "CNN_tokenizer":tk, "CNN_labelencoder":le, "CNN_checkpoint_file":checkpoint_file}
    print("CNN ended...")
    return res


def get_CNN_predictions_confidence(model, X_test, CNN_config, tokenizer, labelencoder):
    """
    return the predictions and confidence on the test set
    :param model: the Keras Classifier Model
    :param X_test: test sequences
    :param y_test: labels
    :return: a list of actual labels
    """
    # Tokenize the dataset on training data
    # tokenization with max words defined and filters
    tokenizer.fit_on_texts(X_test)

    # Sentence Distribution in training data
    # understand the sequence distribution for max length 
    seq_lengths = X_test.apply(lambda x: len(x.split(' ')))
    seq_lengths.describe()

    # Convert Train and Test to fixed length sequences
    X_test_seq_tok = tokenizer.texts_to_sequences(X_test)

    # pad the sequences 
    X_test_seq = pad_sequences(X_test_seq_tok, maxlen=CNN_config["MAX_LEN"])
    
    y_softmax = model.predict(X_test_seq)
    pred = []
    confidence = []
    
    for i in range(0, len(y_softmax)):
        probs = y_softmax[i]
        predicted_index = np.argmax(probs)
        pred.append(labelencoder.inverse_transform(predicted_index))
        confidence.append(probs[predicted_index])
        
    return pred, confidence, y_softmax


def classifiers_pipeline(train_bow, test_bow, label_train, label_test, save_path):
    """
    Calls all the classifiers functions in order to choose and save the best one.
    :param train_bow: training BagOfWords, iterable/list
    :param test_bow: testing BagOfWords, iterable/list
    :param label_train: training labels, iterable/list
    :param label_test: testing labels, iterable/list
    :param save_path: (fixed to Models directory)
    :return: /
    """
    best = 0
    best_model = None
    name = 'error'
    res = {}
    rand_score, rand_model = random_forest_classification(train_bow, test_bow, label_train, label_test, res)
    if rand_score > best:
        name = 'RandomForestClassifier'
        best = rand_score
        best_model = rand_model

    svc_score, svc_model = SVC_classification(train_bow, test_bow, label_train, label_test, res)
    if svc_score > best:
        name = 'SVC'
        best = svc_score
        best_model = svc_model

    lin_svc_score, lin_svc_model = LinearSVC_classification(train_bow, test_bow, label_train, label_test, res)
    if lin_svc_score > best:
        name = 'LinearSVC'
        best = lin_svc_score
        best_model = lin_svc_model

    multiNB_score, multiNB_model = MultinomialNB_classification(train_bow, test_bow, label_train, label_test, res)
    if multiNB_score > best:
        name = 'MultinomialNB'
        best = multiNB_score
        best_model = multiNB_model

    complNB_score, complNB_model = ComplementNB_classification(train_bow, test_bow, label_train, label_test, res)
    if complNB_score > best:
        name = 'ComplementNB'
        best = complNB_score
        best_model = complNB_model

    bernNB_score, bernNB_model = BernoulliNB_classification(train_bow, test_bow, label_train, label_test, res)
    if bernNB_score > best:
        name = 'BernoulliNB'
        best = bernNB_score
        best_model = bernNB_model

    gradboost_score, gradboost_model = GradientBoosting_classification(train_bow, test_bow, label_train, label_test, res)
    if gradboost_score > best:
        name = 'GradientBoostingClassifier'
        best = gradboost_score
        best_model = gradboost_model

    logReg_score, logReg_model = LogisticRegression_classification(train_bow, test_bow, label_train, label_test, res)
    if logReg_score > best:
        name = 'LogisticRegression'
        best = logReg_score
        best_model = logReg_model

    adaBoost_score, adaBoost_model = AdaBoost_classification(train_bow, test_bow, label_train, label_test, res)
    if adaBoost_score > best:
        name = 'AdaBoostClassifier'
        best = adaBoost_score
        best_model = adaBoost_model

    voting_score, voting_model = VotingClassifier_classification(train_bow, test_bow, label_train, label_test, res)
    if voting_score > best:
        name = 'VotingClassifier'
        best = voting_score
        best_model = voting_model

    extraTree_score, extraTree_model = ExtrExtraTrees_classification(train_bow, test_bow, label_train, label_test, res)
    if extraTree_score > best:
        name = 'ExtraTreesClassifier'
        best = extraTree_score
        best_model = extraTree_model

    CNN_score, CNN_model = CNN_classification(train_bow, test_bow, label_train, label_test, res)
    if CNN_score > best:
        name = 'CNNClassifier'
        best = CNN_score
        best_model = CNN_model

    data = {"model": best_model, "accuracy": best}

    return data, name


def classifiers_parallel_pipeline(train_bow, test_bow, label_train, label_test, data_train = None, data_test = None, 
                                  save_prefix = None, measure_metric = None, measure_score_or_class = "accuracy", parameters = None, train_over_full_dataset = False):
    """
    Calls all the classifiers functions in order to choose and save the best one using threading.
    Might save around 25% of execution time.
    :param train_bow: training BagOfWords, iterable/list/sparse matrix
    :param test_bow: testing BagOfWords, iterable/list/sparse matrix
    :param label_train: training labels, iterable/list
    :param label_test: testing labels, iterable/list
    :param save_path: (fixed to Models directory)
    :param measure_metric: measure metric criteria to decide the best classifier
    :param measure_class: measure class to decide the best classifier
    :param parameters: parameters from service table
    :return: model data dictionary and model name
    """
    manager = Manager()
    
    # Variable shared between threads in which the result is stored
    return_dict = manager.dict()

    # Use a shared variable in order to get the results
    proc = []
    
    #check if the dataset has at least 2 classes, if not excludes the algorithms that needs at least 2 classes:
    # LogisticRegression_classification, AdaBoost_classification, LinearSVC_classification
    dataset_with_unique_class_found = False
    if len(np.unique(label_train)) == 1:
        print("--- WARNING --- IN CREATE_TRAIN_TEST FOUND A TRAIN SUBSET WITH ONLY ONE UNIQUE LABEL VALUE: " + str(label_train.values[0]))
        print("--- WARNING --- SKIPPING LogisticRegression, AdaBoost, LinearSVC as they need at least 2 classes")
        dataset_with_unique_class_found = True


    # Declaration of the functions to be used in parallel computing 
    fncs1 = [random_forest_classification, MultinomialNB_classification,BernoulliNB_classification,#GradientBoosting_classification,
             ComplementNB_classification, ExtrExtraTrees_classification] 
        
    if dataset_with_unique_class_found is False:
        fncs1.extend([LinearSVC_classification, LogisticRegression_classification, AdaBoost_classification])
    
    gridsearch_autotuning = get_service_parameter_col(parameters,"gridsearch_autotuning")
    
    if gridsearch_autotuning:
        if dataset_with_unique_class_found is False:
            fncs1.remove(LinearSVC_classification)
            fncs1.remove(LogisticRegression_classification)
            fncs1.remove(AdaBoost_classification)
            fncs1.extend([LinearSVC_autotuning_classification, LogisticRegression_autotuning_classification, AdaBoost_autotuning_classification])
        #fncs1.remove(random_forest_classification)
        fncs1.remove(ExtrExtraTrees_classification)
        fncs1.extend([ExtrExtraTrees_autotuning_classification])
        #fncs1.extend([random_forest_autotuning_classification, ExtrExtraTrees_autotuning_classification])
    
    disable_CNN = get_service_parameter_col(parameters,"disable_CNN")
    if disable_CNN is not True:
        fncs1.append(CNN_classification)
    
    disable_extratree_rfa = get_service_parameter_col(parameters,"disable_extratree_rfa")
    if disable_extratree_rfa is True:
        if gridsearch_autotuning is True:
            fncs1.remove(ExtrExtraTrees_autotuning_classification)
            fncs1.remove(random_forest_autotuning_classification)
            if dataset_with_unique_class_found is False:
                fncs1.remove(AdaBoost_autotuning_classification)
        else:
            fncs1.remove(ExtrExtraTrees_classification)
            fncs1.remove(random_forest_classification)
            if dataset_with_unique_class_found is False:
                fncs1.remove(AdaBoost_classification)

    returnVals = list()
    # instantiating process with arguments 
    # each thread has its own return value to avoid memory issues using one common shared object
    for fn in fncs1:
        if fn is CNN_classification:
            p = ThreadWithReturnValue(target=fn, args=(label_train, label_test, data_train, data_test, save_prefix, parameters, train_over_full_dataset))
        else:
            p = ThreadWithReturnValue(target=fn, args=(train_bow, test_bow, label_train, label_test, save_prefix))
        proc.append(p)
        p.start()
    for p in proc:
        returnVals.append(p.join())

    # Declaration of variables to mute warnings
    best = 0
    name = ''
    model = None
    metrics = None
    confmat = None
    CNN_model_file = ''

    # saving the best model in the output dictionary
    for elem in returnVals:
        if elem is None: continue
        dict = list(elem.values())[0]
        if dict["model"] is None: continue
        metric_val  = get_metric(dict["report"],measure_metric,measure_score_or_class)
        if metric_val is None: continue
        if metric_val > best:
            best = metric_val
            name = dict["name"]
            model = dict["model"]
            confmat = dict["confmat"]
            metrics = dict["report"]
            labelencoder = dict["LabelEncoder"]
            if dict["name"] == "CNNClassifier":
                CNN_model_file = dict["CNN_model_file"]
                CNN_tokenizer = dict["CNN_tokenizer"]
                CNN_labelencoder = dict["CNN_labelencoder"]
                CNN_checkpoint_file = dict["CNN_checkpoint_file"]
     
    if name == "CNNClassifier":
        data = {"model": model, "score_or_class":measure_score_or_class, "metric":measure_metric, "metric_value": best, "report": metrics, "confmat": confmat, "CNN_model_file": CNN_model_file, "CNN_tokenizer":CNN_tokenizer,"CNN_labelencoder":CNN_labelencoder,"CNN_checkpoint_file":CNN_checkpoint_file}
    else:
        data = {"model": model, "score_or_class":measure_score_or_class, "metric":measure_metric, "metric_value": best, "report": metrics, "confmat": confmat, "LabelEncoder": labelencoder}
    
    return data, name


def train_classifiers(column_list, table, table_key, target_col, where_clause = "", save_prefix = "", parameters = None, preset_dataframe = None, disable_min_cardinality_filter = False, tree_classifier_layers = None, guessed_language = None, data_columns = None):

    """
    Train the classification model on training data.
    First the data is downloaded from the database, then the text is pre-processed, divided into training and testing
    sets: Finally the vocabulary is computed, the data are transformed into a float sparse matrix, the "transofmer" is
    saved and training is executed.
    :param parameters: contains a json with all the additional parameters (lemming,stemming,...)
    :param preset_dataframe: if set it's used instead of querying the database
    """
    
    results = dict()
    
    # Check if Model folder exists
    check_for_folders()

    #if the argument preset_dataframe is set, skip db query and use that as dataset
    if preset_dataframe is not None:
        data = preset_dataframe
    else:
        # Get tickets ids, title and description after 2015
        print("Fetching data from db...")
        data = get_db_training_data(column_list, table, table_key, target_col, where_clause)
        
    #data.dropna(inplace=True)

    # Filters out labels with cardinality < ", "min_cardinality" parameter value if set in "parameters" column in services table
    if disable_min_cardinality_filter is not True:
        MIN_CARDINALITY = get_min_cardinality_parameter(parameters)
        dataframe = filter_by_cardinality(data, target_col, MIN_CARDINALITY)
    else:
        dataframe = data
    
    print("Dataframe shape:")
    print(dataframe.shape)

    # Creation of text column and union of columns returned by the query
    print("Preprocessing text...")
    dataframe["text"] = ''
    if data_columns is not None:
        for col in data_columns:
            if col != target_col:
                dataframe["text"] = dataframe["text"].map(str) + ' ' + dataframe[col].map(str)
    else:
        for col in dataframe.columns:
            if col != target_col and col != IDS_COLUMN_SUFFIX and col != "text" and (tree_classifier_layers is None or (tree_classifier_layers is not None and col not in tree_classifier_layers)):
                dataframe["text"] = dataframe["text"].map(str) + ' ' + dataframe[col].map(str)

    # # Fetch parameters from service table json
    # fetch the flag for retraining over the ful dataset
    train_over_full_dataset = get_service_parameter_col(parameters,"train_over_full_dataset")
    
    #CONFORMAL PREDICTION
    Tree_config = get_Tree_config_params(parameters)
    significance = Tree_config["tree_classifier_conformal_significance"]
    conformal_recalibration_loops = Tree_config["tree_classifier_conformal_recalibration_loops"]
    
    # fetch the text language from db
    language = get_language_parameter(parameters)
    
    #if the language has been guessed and it's not in the parameters I use the guessed value for text preprocessing operations
    if (language is '' or language is None) and guessed_language is not None:
        language = guessed_language

    #SEMANTIC_CNN
    enable_semantic_cnn_param = Tree_config["tree_classifier_enable_semantic_cnn"]
    ENABLE_SEMANTIC_CNN = enable_semantic_cnn_param is not None and enable_semantic_cnn_param is True and language != '' and language is not None
    
    # fetch measurement criteria for choosing the best classifier
    measure_metric, measure_score_or_class = get_measurement_criteria(parameters)
    
    # # Clean text
    dataframe['text'] = dataframe['text'].map(lambda x: clean_text(x, language))

    #spell autocorrection --> applies only if spelling_autocorrect parameter is set to True or "fast"
    #dataframe['text'] = word_spelling_corrector(dataframe['text'], language, parameters)
    dataframe['text'] = parallel_speller(dataframe['text'], language, parameters)

    remove_text_noise_param = get_remove_text_noise_parameter(parameters)
    if remove_text_noise_param > 0:
        print("Removing text noise - START: " + str(datetime.datetime.now()))
        words_count_dict = Counter(' '.join(dataframe['text']).split(' '))#.most_common()
        #words_count_dict = pd.Series(' '.join(dataframe['text']).split(" ")).value_counts(sort=True,ascending=False)
        #words_count_dict.where(words_count_dict <= remove_text_noise_param, inplace=True)
        #words_count_dict.dropna(inplace=True)
        #dataframe['text'] = dataframe['text'].map(lambda x: replace_words(x,words_count_dict.index))
        #dataframe['text'] = parallel_remove_words(dataframe['text'],words_count_dict.index)
        dataframe['text'] = parallel_remove_words(dataframe['text'],[w for w in words_count_dict if words_count_dict[w] <= remove_text_noise_param])
        print("Removing text noise - END: " + str(datetime.datetime.now()))

    #remove oov (out of vocaboulary words from dataset
    #dataframe['text'] = parallel_remove_oov(dataframe['text'], language, parameters)
    dataframe['text'] = remove_oov_words(dataframe['text'].tolist(), language, parameters)
    
    # Lemmatize text 
    if get_lemming_parameter(parameters):
        dataframe['text'] = parallel_lemming(dataframe['text'], language)
    
    # Stemmatize text 
    if get_stemming_parameter(parameters):
        print("Applying STEMMING")
        dataframe['text'] = dataframe['text'].map(lambda x: word_stemming(x, language))    
    
    test_size = 0.25
    calibration_size = 0
    #CONFORMAL PREDICTION
    data_cal = None
    label_cal = None
    if CONFORMAL_PREDICTION:
        test_size = 0.2
        calibration_size = 0.2
    
    # Creating training and testing set
    data_train, data_test, label_train, label_test = create_train_test(dataframe["text"], dataframe[target_col], test_size=test_size + calibration_size, json_parameters=parameters)
    
    # Creating vocabulary tf-idf-wighted
    print("Creating vectorizer and transforming data...")
    unwanted = None
    if language != '' and language is not None:
        unwanted = stopwords.words(language)

    tfidf_vect = TfidfVectorizer(analyzer='word', ngram_range=range(1, 3), max_features=10000, stop_words=unwanted,
                                 max_df=0.7, min_df=3)
    # Fitting the model
    tfidf_vect.fit(data_train)                             
    
    if CONFORMAL_PREDICTION:
        # Conformal prediction
        data_cal, data_test, label_cal, label_test = create_train_test(data_test, label_test, test_size=0.5, json_parameters=parameters)
        #cal_bow = tfidf_vect.transform(data_cal)

    # Creating training and testing bag of words weighted by tfidf
    train_bow = tfidf_vect.transform(data_train)
    test_bow = tfidf_vect.transform(data_test)
        
    force_garbage_collection(verbose=1)

    # CLASSIFICATION
    print("--------------------------------------------------------------------------------")
    print("TRAINING CLASSIFIERS over target: ", target_col, " - save prefix: ", save_prefix)
    print("--------------------------------------------------------------------------------")
    model_dictionary, model_name = classifiers_parallel_pipeline(train_bow, test_bow, label_train, label_test, data_train, data_test, save_prefix, measure_metric, measure_score_or_class, parameters, train_over_full_dataset)
    model_dictionary["columns"] = column_list 
    if target_col in model_dictionary["columns"]:
        model_dictionary["columns"].remove(target_col) # NB type(column_list) = STRING
    model_dictionary["target_col"] = target_col
    model_dictionary["vectorizer"] = tfidf_vect
    model_dictionary["model_name"] = model_name
    model_dictionary["createdtime"] = datetime.datetime.now()

    #train SEMANTIC_CNN
    if ENABLE_SEMANTIC_CNN:
        #TODO : light clean text instead ???
        semantic_cnn_results = semantic_cnn_classification(data_train, label_train, data_test, label_test, language, save_prefix, train_over_full_dataset, parameters, pretrained_word2vec_vocab = WORD2VEC_VOCAB)

        if semantic_cnn_results is not None:
            
            if model_dictionary["metric_value"] < semantic_cnn_results["SEMANTIC_CNN"]["accuracy"]:
                print("-------------------------------------------------------------------")
                print("-------------------------------------------------------------------")
                print("--- SEMANTIC_CNN MODEL HAD HIGHGER ACCURACY THAN THE BEST MODEL ---")
                print("-------------------------------------------------------------------")
                print("-------------------------------------------------------------------")
            save_path = os.path.join(os.getcwd(), "Models")
            keras_model_filename = os.path.join(save_prefix + SEMANTIC_CNN_MODEL_FILENAME_SUFFIX)
            semantic_cnn_object_filename = os.path.join(save_prefix + SEMANTIC_CNN_MODEL_CLASS_FILENAME_SUFFIX)
            semantic_cnn_model = semantic_cnn_results["SEMANTIC_CNN"]["model"]
            results["semantic_cnn_accuracy"] = semantic_cnn_results["SEMANTIC_CNN"]["accuracy"]

            if CONFORMAL_PREDICTION: 
                #xcal,ycal = semantic_cnn_model.input_preprocess(data_cal,label_cal)
                #ycal = np.array([np.where(r==1)[0][0] for r in ycal])
                #xtest,ytest = semantic_cnn_model.input_preprocess(data_test,label_test)
                print("--- CONFORMAL PREDICTION RESULTS FOR SEMANTIC CNN MODEL over target: ", target_col, " - save prefix: ", save_prefix )
                
                #semantic_cnn_conformal_model_accuracy, semantic_cnn_conformal_model = get_conformal_model_with_recalibration(semantic_cnn_model, xcal, ycal, xtest, label_test, significance, semantic_cnn_model.le)
                semantic_cnn_conformal_model_accuracy, semantic_cnn_conformal_model = get_conformal_model_with_recalibration(semantic_cnn_model, data_cal, label_cal, data_test, label_test, significance, semantic_cnn_model.le, recalibration_loops = conformal_recalibration_loops)
                semantic_cnn_model.conformal_model = semantic_cnn_conformal_model
                semantic_cnn_model.conformal_model_accuracy = semantic_cnn_conformal_model_accuracy
                results["semantic_cnn_conformal_model_accuracy"] = semantic_cnn_conformal_model_accuracy
            #save semantic_cnn model
            semantic_cnn_model.save_model(save_path, keras_model_filename, semantic_cnn_object_filename)
            #with open(os.path.join(save_path, semantic_cnn_object_filename), "wb") as pklfile:
            #    pickle.dump(semantic_cnn_model, pklfile)
            
        
    ###np.set_printoptions(threshold=sys.maxsize)
    ###pd.set_option('display.max_rows', None)
    ###pd.set_option('display.max_columns', None)
    ###pd.set_option('display.width', None)
    ###pd.set_option('display.max_colwidth', -1)
    ###dump_var_to_file(data_test, save_prefix + "_" + model_name + "_train_test_data", "logs")
    
    force_garbage_collection(verbose=1)
    #retrain the best model over the whole dataset instead of splitting in train and test subdatasets
    if train_over_full_dataset is not None and train_over_full_dataset is True and model_dictionary["model"] is not None:
        print("RETRAINING MODEL ", model_name ," OVER THE FULL DATASET over target: ", target_col, " - save prefix: ", save_prefix)
        if model_name == "CNNClassifier":
            path = os.path.join(os.getcwd(), "Models")
            filename = save_prefix + CNN_MODEL_FILENAME_SUFFIX
            filename_path = os.path.join(path, filename)
            CNN_model = load_model(filename_path)
            CNN_model.fit(
                CNN_dataset_full,
                CNN_labels_full,
                epochs=NB_EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=1,
                callbacks=checkpoints)
            CNN_model.load_weights(model_dictionary["CNN_checkpoint_file"])
            # save model to disk
            CNN_model.save(filename_path)

        else:
            model = model_dictionary["model"]
            dataset_full = vstack((train_bow,test_bow))
            labels_full = label_train.append(label_test)
            le = model_dictionary["LabelEncoder"]
            le.fit(labels_full)
            labels_le = le.transform(labels_full)
            model.fit(dataset_full, labels_le)
            model_dictionary["model"] = model
            model_dictionary["LabelEncoder"] = le

    save_path = os.path.join(os.getcwd(), "Models")
    checkpoint_path = os.path.join(os.getcwd(), "Checkpoints")
    
    # Delete the CNN model file if not the best one
    if model_name != "CNNClassifier":
        try:
            filename = os.path.join(save_path, save_prefix + CNN_MODEL_FILENAME_SUFFIX)
            if os.path.exists(filename):
                os.remove(filename)
                print("Successfully deleted unused CNN model file: " + filename)
        except Exception as e:
            print("Error when deleting the CNN model file " + filename)
        try:
            filename_best_weights = os.path.join(checkpoint_path, save_prefix + '_Train-' + model_dictionary["model_name"]  + '-best_weights.h5')
            if os.path.exists(filename_best_weights):
                os.remove(filename_best_weights)
                print("Successfully deleted unused CNN model weights file: " + filename_best_weights)
        except Exception as e:
            print("Error when deleting the CNN model weights file " + filename_best_weights)
    
    
    if CONFORMAL_PREDICTION:
        
        print("--- CONFORMAL PREDICTION RESULTS FOR ",model_name," MODEL over target: ", target_col, " - save prefix: ", save_prefix)
        if model_name != "CNNClassifier":
            #label_cal = model_dictionary["LabelEncoder"].transform(label_cal)
            #conformal_model_accuracy, conformal_model = get_conformal_model_with_recalibration(model_dictionary["model"], cal_bow, label_cal, test_bow, label_test, significance, model_dictionary["LabelEncoder"])
            conformal_model_accuracy, conformal_model = get_conformal_model_with_recalibration(model_dictionary["model"], data_cal, label_cal, data_test, label_test, significance, model_dictionary["LabelEncoder"], tfidf_vect = tfidf_vect, recalibration_loops = conformal_recalibration_loops)
        else:
            conformal_model_accuracy, conformal_model = get_conformal_model_with_recalibration(CNN_model, data_cal, label_cal, data_test, label_test, significance, recalibration_loops = conformal_recalibration_loops)
        model_dictionary["conformal_model"] = conformal_model
        model_dictionary["conformal_model_accuracy"] = conformal_model_accuracy
        results["conformal_model_accuracy"] = conformal_model_accuracy
       
    # Saving the model, vectorizer and parameters to file
    filename = save_prefix + MODEL_FILENAME_SUFFIX
    filename_path = os.path.join(save_path, filename)
    with open(filename_path, "wb") as pklfile:
        pickle.dump(model_dictionary, pklfile)
    
    results["model_name"] = model_dictionary["model_name"]
    results["score_or_class"] = model_dictionary["score_or_class"]
    results["metric"] = model_dictionary["metric"]
    results["metric_value"] = model_dictionary["metric_value"]
    
    return results


def predict_label(cron_id, ticket_id, table_name, key_name, data_columns, model_prefix = "",  parameters = None, inputfile = None):
    """
    Predicts the label of a given ticket. Given the ticket_id it gets the text from the database and then it saves the
    result in the database
    :param ticket_id: integer corresponding to the ticket to classify
    :param parameters: contains a json with all the additional parameters (lemming,stemming,...)
    :return: /
    """

    ##################################################
    # Parameters for CNN model training and prediction
    ##################################################
    CNN_config = get_CNN_config_params(parameters)
    le = None

    try:
        model_folder = os.path.join(os.getcwd(), "Models")
        if model_prefix == "":
            for fname in os.listdir(model_folder):
                if "_data.pkl" in fname:
                    to_load = os.path.join(model_folder, fname)
                    with open(to_load, "rb") as file:
                        model_dict = pickle.load(file)
                        model = model_dict["model"]
                        vectorizer = model_dict["vectorizer"]
                        #column_list = model_dict["columns"]
                        column_list = data_columns.split(",")
                        if "LabelEncoder" in model_dict.keys():
                            le = model_dict["LabelEncoder"]

        else:
            fname = model_prefix + MODEL_FILENAME_SUFFIX                
            to_load = os.path.join(model_folder, fname)
            with open(to_load, "rb") as file:
                        model_dict = pickle.load(file)
                        model = model_dict["model"]
                        vectorizer = model_dict["vectorizer"]
                        #column_list = model_dict["columns"]
                        column_list = data_columns.split(",")
                        if "LabelEncoder" in model_dict.keys():
                            le = model_dict["LabelEncoder"]
                        
        model_name = model_dict["model_name"]
        model_date = str(model_dict["createdtime"])

        print("Classifying with " + model_name + " created " + str(model_date) + " for " + model_prefix )                
        
        #gets a list of ids to be predicted from file (as an argument in predict command)
        #and fetch data from db
        ids = None
        if inputfile is not None:
            ids = get_id_list_from_file(inputfile)
            INclause = str(ids).replace("]",")").replace("[","(")
            where_clause = " " + key_name + " IN " + INclause
            data = get_db_training_data(column_list, table_name, key_name, key_name, where_clause)
        else:
            data = get_object_description(ticket_id, table_name, key_name, column_list)
        
        # Preprocess the text before classification (stopwords removal, lemming, stemming)
        data = text_preprocess(data, "text", parameters)
        
        # if predicting with CNN Classifier it needs to load the model and weights from h5 file
        CNNmodel = None
        prediction = None  
        final_output = None
        
        if model_name == "CNNClassifier":
            #CNNmodel = load_model(os.path.join(model_folder,model_dict["CNN_model_file"]))
            CNNmodel = load_model(os.path.join(model_folder, model_prefix + CNN_MODEL_FILENAME_SUFFIX))
            prediction = get_CNN_predictions_confidence(CNNmodel, data["text"], CNN_config, model_dict["CNN_tokenizer"], model_dict["CNN_labelencoder"])
            if inputfile is not None:
                final_output = list()
                for p in prediction[0]:
                    final_output.append(p.item(0))
            else:
                final_output = prediction[0][0].item(0)
        else:
            data_bow = vectorizer.transform(data["text"])
            prediction = model.predict(data_bow)
            #for old models compatibility
            if le != None: 
                pred_values = le.inverse_transform(prediction)
            else:
                pred_values = prediction

            if inputfile is not None:
                final_output = pred_values
            else:
                final_output  = pred_values[0]
        
        
        print("----- Predictions: ")
        results = pd.DataFrame(columns=["id","prediction"])
        if inputfile is not None:
            j = 0
            for id in ids:
                r = [{"id":str(id),"prediction":final_output[j]}]
                results = results.append(r)
                #print(str(id)+ " --> " + final_output[j])
                j += 1

            print(tabulate(results,headers="keys",showindex="never"))
            save_predictions_in_db_batch(cron_id, ids, final_output)
       
        else:
            print(data.loc[0, "text"])
            print(final_output)
            save_predictions_in_db(cron_id, ticket_id, final_output)


    except RuntimeError as e:
        print("Error when making prediction")
        print(e)
        traceback.print_exc()




def train_classifiers_preprocess_autotuning(column_list, table, table_key, target_col, where_clause = "", save_prefix = "", parameters = None, preset_dataframe = None, disable_min_cardinality_filter = False, tree_classifier_layers = None, guessed_language = None, data_columns = None):

    """
    Train the classification model on training data.
    First the data is downloaded from the database, then the text is pre-processed, divided into training and testing
    sets: Finally the vocabulary is computed, the data are transformed into a float sparse matrix, the "transofmer" is
    saved and training is executed.
    :param parameters: contains a json with all the additional parameters (lemming,stemming,...)
    :param preset_dataframe: if set it's used instead of querying the database
    """
    
    results = dict()
    
    # Check if Model folder exists
    check_for_folders()

    #if the argument preset_dataframe is set, skip db query and use that as dataset
    if preset_dataframe is not None:
        data = preset_dataframe
    else:
        # Get tickets ids, title and description after 2015
        print("Fetching data from db...")
        data = get_db_training_data(column_list, table, table_key, target_col, where_clause)
        
    #data.dropna(inplace=True)

    # Filters out labels with cardinality < ", "min_cardinality" parameter value if set in "parameters" column in services table
    if disable_min_cardinality_filter is not True:
        MIN_CARDINALITY = get_min_cardinality_parameter(parameters)
        dataframe = filter_by_cardinality(data, target_col, MIN_CARDINALITY)
    else:
        dataframe = data
    
    print("Dataframe shape:")
    print(dataframe.shape)

    # Creation of text column and union of columns returned by the query
    print("Preprocessing text...")
    dataframe["text"] = ''
    if data_columns is not None:
        for col in data_columns:
            if col != target_col:
                dataframe["text"] = dataframe["text"].map(str) + ' ' + dataframe[col].map(str)
    else:
        for col in dataframe.columns:
            if col != target_col and col != IDS_COLUMN_SUFFIX and col != "text" and (tree_classifier_layers is None or (tree_classifier_layers is not None and col not in tree_classifier_layers)):
                dataframe["text"] = dataframe["text"].map(str) + ' ' + dataframe[col].map(str)

    # # Fetch parameters from service table json
    # fetch the flag for retraining over the ful dataset
    train_over_full_dataset = get_service_parameter_col(parameters,"train_over_full_dataset")
    
    #CONFORMAL PREDICTION
    Tree_config = get_Tree_config_params(parameters)
    significance = Tree_config["tree_classifier_conformal_significance"]
    conformal_recalibration_loops = Tree_config["tree_classifier_conformal_recalibration_loops"]
    
    # fetch the text language from db
    language = get_language_parameter(parameters)
    
    #if the language has been guessed and it's not in the parameters I use the guessed value for text preprocessing operations
    if (language is '' or language is None) and guessed_language is not None:
        language = guessed_language

    #SEMANTIC_CNN
    enable_semantic_cnn_param = Tree_config["tree_classifier_enable_semantic_cnn"]
    ENABLE_SEMANTIC_CNN = enable_semantic_cnn_param is not None and enable_semantic_cnn_param is True and language != '' and language is not None
    
    # fetch measurement criteria for choosing the best classifier
    measure_metric, measure_score_or_class = get_measurement_criteria(parameters)

    test_size = 0.25
    calibration_size = 0
    #CONFORMAL PREDICTION
    data_cal = None
    label_cal = None
    if CONFORMAL_PREDICTION:
        test_size = 0.2
        calibration_size = 0.2
    
    best_model_accuracy = 0
    best_tfidf_vect = None
    best_train_bow = None
    best_test_bow = None
    best_model_dictionary = best_model_name = None
    best_data_cal = best_data_test = best_label_cal = best_label_test = None
    best_data_train = best_label_train = None
    
    TEXT_PREPROCESSING_AUTOTUNING = get_service_parameter_col(parameters,"text_preprocessing_autotuning")
    
    print("================ TRAINING WITH TEXT PREPROCESSING AUTOTUNING ==================")
    
    start_autotuning = datetime.datetime.now()
    print("--- TEXT PREPROCESSING AUTOTUNING - START: ", start_autotuning)

    text_preprocessing_params = get_text_preprocessing_params(parameters, autotuning_enabled=TEXT_PREPROCESSING_AUTOTUNING) # gets params value ranges from db
    text_preprocessing_params_combinations = [get_combination(text_preprocessing_params.keys(),v) for v in list(product(*text_preprocessing_params.values()))]
    count = 1
    total = len(text_preprocessing_params_combinations)
                    
    for tpp in text_preprocessing_params_combinations:

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        start_round = datetime.datetime.now()
        print("--- TRAINING WITH TEXT PREPROCESSING AUTOTUNING - round: " , count , " / " , total , " - START " , start_round)
        print("PARAMETERS SET: ")
        print(tpp)
        
        dataframe['text'] = text_preprocessing(dataframe['text'], tpp)
           
        # Creating training and testing set
        data_train, data_test, label_train, label_test = create_train_test(dataframe["text"], dataframe[target_col], test_size=test_size + calibration_size, json_parameters=parameters)
        
        # Creating vocabulary tf-idf-wighted
        print("Creating vectorizer and transforming data...")
        tfidf_vect = TfidfVectorizer(analyzer='word', ngram_range=range(1, 3), max_features=10000, max_df=0.7, min_df=3)
        tfidf_vect.fit(data_train)                             
        
        if CONFORMAL_PREDICTION:
            data_cal, data_test, label_cal, label_test = create_train_test(data_test, label_test, test_size=0.5, json_parameters=parameters)

        # Creating training and testing bag of words weighted by tfidf
        train_bow = tfidf_vect.transform(data_train)
        test_bow = tfidf_vect.transform(data_test)
            
        force_garbage_collection(verbose=1)

        print("--------------------------------------------------------------------------------")
        print("TRAINING CLASSIFIERS over target: ", target_col, " - save prefix: ", save_prefix)
        print("--------------------------------------------------------------------------------")
        model_dictionary, model_name = classifiers_parallel_pipeline(train_bow, test_bow, label_train, label_test, data_train, data_test, save_prefix, measure_metric, measure_score_or_class, parameters, train_over_full_dataset)
        model_dictionary["columns"] = column_list 
        if target_col in model_dictionary["columns"]:
            model_dictionary["columns"].remove(target_col) # NB type(column_list) = STRING
        model_dictionary["target_col"] = target_col
        model_dictionary["vectorizer"] = tfidf_vect
        model_dictionary["model_name"] = model_name
        model_dictionary["createdtime"] = datetime.datetime.now()
        model_dictionary["text_preprocessing_params"] = tpp
        
        if model_dictionary["metric_value"] > best_model_accuracy:
            best_model_accuracy = model_dictionary["metric_value"]
            best_tfidf_vect = tfidf_vect
            best_train_bow = train_bow
            best_test_bow = test_bow
            best_model_dictionary = model_dictionary
            best_model_name = model_name
            best_data_cal = data_cal
            best_data_test = data_test
            best_label_cal = label_cal
            best_label_test = label_test
            best_data_train = data_train
            best_label_train = label_train
        

        print("--- TRAINING WITH TEXT PREPROCESSING AUTOTUNING - round: " , count , " / " , total , " - END: ",datetime.datetime.now()," - total exec time: ",elapsed_time(start_round,datetime.datetime.now()))
        print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        count += 1
        
    print("--- TEXT PREPROCESSING AUTOTUNING -   END: ",datetime.datetime.now()," - total exec time: ",elapsed_time(start_autotuning,datetime.datetime.now()))
        

    #train SEMANTIC_CNN
    if ENABLE_SEMANTIC_CNN:
        semantic_cnn_results = semantic_cnn_classification(best_data_train, best_label_train, best_data_test, best_label_test, language, save_prefix, train_over_full_dataset, parameters, pretrained_word2vec_vocab = WORD2VEC_VOCAB)

        if semantic_cnn_results is not None:            
            if best_model_dictionary["metric_value"] < semantic_cnn_results["SEMANTIC_CNN"]["accuracy"]:
                print("-------------------------------------------------------------------")
                print("-------------------------------------------------------------------")
                print("--- SEMANTIC_CNN MODEL HAD HIGHGER ACCURACY THAN THE BEST MODEL ---")
                print("-------------------------------------------------------------------")
                print("-------------------------------------------------------------------")
            save_path = os.path.join(os.getcwd(), "Models")
            keras_model_filename = os.path.join(save_prefix + SEMANTIC_CNN_MODEL_FILENAME_SUFFIX)
            semantic_cnn_object_filename = os.path.join(save_prefix + SEMANTIC_CNN_MODEL_CLASS_FILENAME_SUFFIX)
            semantic_cnn_model = semantic_cnn_results["SEMANTIC_CNN"]["model"]
            results["semantic_cnn_accuracy"] = semantic_cnn_results["SEMANTIC_CNN"]["accuracy"]

            if CONFORMAL_PREDICTION: 
                print("--- CONFORMAL PREDICTION RESULTS FOR SEMANTIC CNN MODEL over target: ", target_col, " - save prefix: ", save_prefix )
                semantic_cnn_conformal_model_accuracy, semantic_cnn_conformal_model = get_conformal_model_with_recalibration(semantic_cnn_model, best_data_cal, best_label_cal, best_data_test, best_label_test, significance, semantic_cnn_model.le, recalibration_loops = conformal_recalibration_loops)
                semantic_cnn_model.conformal_model = semantic_cnn_conformal_model
                semantic_cnn_model.conformal_model_accuracy = semantic_cnn_conformal_model_accuracy
                results["semantic_cnn_conformal_model_accuracy"] = semantic_cnn_conformal_model_accuracy
            
            #save semantic_cnn model
            semantic_cnn_model.save_model(save_path, keras_model_filename, semantic_cnn_object_filename)

    
    force_garbage_collection(verbose=1)
    #retrain the best model over the whole dataset instead of splitting in train and test subdatasets
    if train_over_full_dataset is not None and train_over_full_dataset is True and best_model_dictionary["model"] is not None:
        print("RETRAINING MODEL ", best_model_name ," OVER THE FULL DATASET over target: ", target_col, " - save prefix: ", save_prefix)
        if best_model_name == "CNNClassifier":
            path = os.path.join(os.getcwd(), "Models")
            filename = save_prefix + CNN_MODEL_FILENAME_SUFFIX
            filename_path = os.path.join(path, filename)
            CNN_model = load_model(filename_path)
            CNN_model.fit(
                CNN_dataset_full,
                CNN_labels_full,
                epochs=NB_EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=1,
                callbacks=checkpoints)
            CNN_model.load_weights(best_model_dictionary["CNN_checkpoint_file"])
            # save model to disk
            CNN_model.save(filename_path)

        else:
            model = best_model_dictionary["model"]
            dataset_full = vstack((best_train_bow,best_test_bow))
            labels_full = best_label_train.append(best_label_test)
            le = best_model_dictionary["LabelEncoder"]
            le.fit(labels_full)
            labels_le = le.transform(labels_full)
            model.fit(dataset_full, labels_le)
            best_model_dictionary["model"] = model
            best_model_dictionary["LabelEncoder"] = le

    save_path = os.path.join(os.getcwd(), "Models")
    checkpoint_path = os.path.join(os.getcwd(), "Checkpoints")
    
    # Delete the CNN model file if not the best one
    if best_model_name != "CNNClassifier":
        try:
            filename = os.path.join(save_path, save_prefix + CNN_MODEL_FILENAME_SUFFIX)
            if os.path.exists(filename):
                os.remove(filename)
                print("Successfully deleted unused CNN model file: " + filename)
        except Exception as e:
            print("Error when deleting the CNN model file " + filename)
        try:
            filename_best_weights = os.path.join(checkpoint_path, save_prefix + '_Train-' + best_model_dictionary["model_name"]  + '-best_weights.h5')
            if os.path.exists(filename_best_weights):
                os.remove(filename_best_weights)
                print("Successfully deleted unused CNN model weights file: " + filename_best_weights)
        except Exception as e:
            print("Error when deleting the CNN model weights file " + filename_best_weights)
    
    
    if CONFORMAL_PREDICTION:
        
        print("--- CONFORMAL PREDICTION RESULTS FOR ",best_model_name," MODEL over target: ", target_col, " - save prefix: ", save_prefix)
        if best_model_name != "CNNClassifier":
            conformal_model_accuracy, conformal_model = get_conformal_model_with_recalibration(best_model_dictionary["model"], best_data_cal, best_label_cal, best_data_test, best_label_test, significance, best_model_dictionary["LabelEncoder"], tfidf_vect = best_tfidf_vect, recalibration_loops = conformal_recalibration_loops)
        else:
            conformal_model_accuracy, conformal_model = get_conformal_model_with_recalibration(CNN_model, best_data_cal, best_label_cal, best_data_test, best_label_test, significance, recalibration_loops = conformal_recalibration_loops)
        best_model_dictionary["conformal_model"] = conformal_model
        best_model_dictionary["conformal_model_accuracy"] = conformal_model_accuracy
        results["conformal_model_accuracy"] = conformal_model_accuracy
       
    # Saving the model, vectorizer and parameters to file
    filename = save_prefix + MODEL_FILENAME_SUFFIX
    filename_path = os.path.join(save_path, filename)
    with open(filename_path, "wb") as pklfile:
        pickle.dump(best_model_dictionary, pklfile)
    
    results["model_name"] = best_model_dictionary["model_name"]
    results["score_or_class"] = best_model_dictionary["score_or_class"]
    results["metric"] = best_model_dictionary["metric"]
    results["metric_value"] = best_model_dictionary["metric_value"]
    
    return results







#########################################################################################################        
#####################               SENTIMENT CLASSIFIER METHODS                 ########################     
#########################################################################################################  


def predict_sentiment(data_columns, cron_id, ticket_id, table_name, key_name, parameters = None, inputfile = None):

    #predicts italian sentiment and/or emotion of input texts using pretrained BERT models and feelit library
    try:
        print(" --- SENTIMENT CLASSIFIER --- PREDICT START at: " + str(datetime.datetime.now()))
        sent_conf = get_sentiment_config_params(parameters)
        SENTIMENT = sent_conf["sentiment_output"]
        EMOTION = sent_conf["emotion_output"]
        MAXLENGHT = sent_conf["max_len"]
        final_output = dict()
        
        #fetch input data from DB
        ids = None
        if inputfile is not None:
            ids = get_id_list_from_file(inputfile)
            INclause = str(ids).replace("]",")").replace("[","(")
            where_clause = " " + key_name + " IN " + INclause
            data = get_db_training_data(data_columns, table_name, key_name, key_name, where_clause)
        else:
            data = get_object_description(ticket_id, table_name, key_name, data_columns)
        
        #concatenates all text data columns
        data["text"] = ""
        for c in data_columns:
            data["text"] = data["text"].map(str) + ' ' + data[c].map(str)
         
        #basic text preprocessing for BERT models
        print("Preprocessing text...")
        data["text"] = list(map(lambda x: str(bert_clean_text(x)), data["text"]))
        data.reset_index(inplace=True,drop=True)
        test_texts = data["text"].to_list()
        
        if SENTIMENT: 
                
            print("Predicting sentiment...")
            tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_PATH)
            model = TFAutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL_PATH)

            test_labels = ["negative", "neutral", "positive"] #MUST be this order!!
            test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=MAXLENGHT)
            predict_input = [tokenizer.encode(t,truncation=True,padding="max_length",return_tensors="tf",max_length=MAXLENGHT) for t in test_texts]
            predict_labels = []
            for p in predict_input:
                #idx = np.argmax(model(p).logits)
                idx = np.argmax(model(p))
                predict_labels.append(test_labels[idx])
            
            if inputfile is not None:
                final_output["sentiment"] = predict_labels
            else:
                final_output["sentiment"] = predict_labels[0]
        
        if EMOTION:
        
            print("Predicting emotion...")
            emotion_classifier = EmotionClassifier()
            sentiment_classifier = SentimentClassifier()
            
            predict_labels = []
            predict_labels = emotion_classifier.predict(test_texts)
            #print(sentiment_classifier.predict(test_texts))
            if inputfile is not None:
                final_output["emotion"] = predict_labels
            else:
                final_output["emotion"] = predict_labels[0]
            
        
        print("----- Predictions: ")
        results = pd.DataFrame(columns=["id","prediction"])
        if inputfile is not None:
            for j,id in enumerate(ids):
                p = dict()
                if SENTIMENT: p["sentiment"] = final_output["sentiment"][j]
                if EMOTION: p["emotion"] = final_output["emotion"][j]
                r = [{"id":str(id),"prediction":p}]
                results = results.append(r)
            
            print(tabulate(results,headers="keys",showindex="never"))
            save_predictions_in_db_batch(cron_id, ids, list(map(lambda x: json.dumps(x),list(results["prediction"]))))
        else:
            print(data.loc[0,:])
            print(final_output)
            save_predictions_in_db(cron_id, ticket_id, final_output)

       
    except Exception as e:
        print("ERROR when predicting with Sentiment Classifier")
        traceback.print_exc()
    
   
    print(" --- SENTIMENT CLASSIFIER --- PREDICT END at: " + str(datetime.datetime.now()))











#########################################################################################################        
#####################               CATEGORICAL CLASSIFIER METHODS               ########################     
#########################################################################################################  

def train_classifiers_categorical(column_list, table, table_key, target_col, where_clause = "", save_prefix = "", parameters = None):
    #detects text and categorical column data types and put them through the classifiers pipeline
    # --> all text columns are concatenated into one TEXT_COL 
        
    # Check if Model folder exists
    check_for_folders()
    save_path = os.path.join(os.getcwd(), "Models")
    
    global SAVE_PREFIX_GLOBAL
    SAVE_PREFIX_GLOBAL = save_prefix

    # Get tickets ids, title and description
    print(" --- COLLECTING DATA FROM DB - START at: " + str(datetime.datetime.now()))
    data = get_db_training_data(column_list, table, table_key, target_col, where_clause)
    print(" --- COLLECTING DATA FROM DB - END at: " + str(datetime.datetime.now()))
    #data.dropna(inplace=True)

    # fetch the text language from db
    language = get_language_parameter(parameters)
    
    # Filters out labels with cardinality < ", "min_cardinality" parameter value if set in "parameters" column in services table
    # filters based on the last layer that is the actual classification target 
    MIN_CARDINALITY = get_min_cardinality_parameter(parameters)
    df = filter_by_cardinality(data, target_col, MIN_CARDINALITY)
    
    print("Dataframe shape:")
    print(df.shape)
    
    train_over_full_dataset = get_service_parameter_col(parameters,"train_over_full_dataset")
    
    # fetch measurement criteria for choosing the best classifier
    measure_metric, measure_score_or_class = get_measurement_criteria(parameters)
    
    proportional = True
    stratify_param = get_service_parameter_col(parameters,"stratify_split")
    if stratify_param is not None and stratify_param is False:
        proportional = False

    cat_cols = []
    cat_pipelines = []
    df[TEXT_COL] = "" 
    tfidf_vect = None

    # Creating training and testing set
    X_train, X_test, y_train, y_test = create_train_test(df.drop(columns=[target_col]), df[target_col], test_size=0.2, json_parameters=parameters, proportional=proportional)

    X_train = categorical_input_preprocess(X_train)
    X_test = categorical_input_preprocess(X_test)
    print("Dataset column types: ")
    print(X_train.dtypes)
    
    for col in X_train.columns:
        if type(X_train[col].dtype) is CategoricalDtype:
            cat_cols.append(col)
            cat_pipelines.append( ("catpipe_"+str(col), Pipeline([('lb', MyLabelBinarizer())]), col) )

    all_pipelines = cat_pipelines
    
    CATEGORICAL_COLS_ONLY = len(cat_cols) == len(X_train.columns)
    if not CATEGORICAL_COLS_ONLY:
        #clean text column (merged result of all text columns)
        X_train[TEXT_COL] = X_train[TEXT_COL].map(lambda x: clean_text(x, language))
        X_test[TEXT_COL] = X_test[TEXT_COL].map(lambda x: clean_text(x, language))
        # Define text pipeline
        tfidf_vect = MyTfidfVectorizer(analyzer='word', ngram_range=range(1, 3), max_features=10000, max_df=0.7, min_df=3)
        text_pipe = Pipeline([('vectorizer', tfidf_vect)])
        all_pipelines.append(('text', text_pipe, [TEXT_COL]))
    
    preprocessor = ColumnTransformer(all_pipelines)
    xtrain = preprocessor.fit_transform(X_train)
    xtest = preprocessor.transform(X_test)
   
    model_dictionary, model_name = classifiers_parallel_pipeline(xtrain, xtest, y_train, y_test, None, None, save_prefix, measure_metric, measure_score_or_class, parameters, train_over_full_dataset)
    model_dictionary["columns"] = column_list 
    if target_col in model_dictionary["columns"]:
        model_dictionary["columns"].remove(target_col) # NB type(column_list) = STRING
    model_dictionary["target_col"] = target_col
    model_dictionary["preprocessor"] = preprocessor
    model_dictionary["categorical_cols_only"] = CATEGORICAL_COLS_ONLY
    model_dictionary["model_name"] = model_name
    model_dictionary["createdtime"] = datetime.datetime.now()
    
    if train_over_full_dataset is not None and train_over_full_dataset is True:
        print("RETRAINING MODEL ", model_name ," OVER THE FULL DATASET over target: ", target_col, " - save prefix: ", save_prefix)
        model = model_dictionary["model"]
        dataset_full = vstack((xtrain, xtest))
        y_full = y_train.append(y_test)
        le = model_dictionary["LabelEncoder"]
        le.fit(y_full)
        labels_le = le.transform(y_full)
        model.fit(dataset_full, labels_le)
        model_dictionary["model"] = model
        model_dictionary["LabelEncoder"] = le
       
    
    # Saving the model, vectorizer and parameters to file
    filename = save_prefix + MODEL_FILENAME_SUFFIX
    filename_path = os.path.join(save_path, filename)
    with open(filename_path, "wb") as pklfile:
        pickle.dump(model_dictionary, pklfile)
    
    results = dict()
    results["model_name"] = model_dictionary["model_name"]
    results["score_or_class"] = model_dictionary["score_or_class"]
    results["metric"] = model_dictionary["metric"]
    results["metric_value"] = model_dictionary["metric_value"]
    
    return results
    

def predict_categorical(data_columns, cron_id, ticket_id, table_name, key_name, model_prefix, train_target_column, parameters = None, inputfile = None):
    #predicts value and saves prediction to db

    ### LOADING MODEL FROM FILE
    try:
        print(" --- CATEGORICAL CLASSIFIER --- PREDICT START at: " + str(datetime.datetime.now()))
        
        #load the tree dictionary from file
        model_folder = os.path.join(os.getcwd(), "Models")
        fname = model_prefix + MODEL_FILENAME_SUFFIX                
        to_load = os.path.join(model_folder, fname)
        with open(to_load, "rb") as file:
            model_dict = pickle.load(file)
            model = model_dict["model"]
            preprocessor = model_dict["preprocessor"]
            categorical_cols_only = model_dict["categorical_cols_only"]
            column_list = data_columns.split(",")
            if "LabelEncoder" in model_dict.keys():
                le = model_dict["LabelEncoder"]
                    
        model_name = model_dict["model_name"]
        model_date = str(model_dict["createdtime"])

        print("Classifying with " + model_name + " created " + str(model_date) + " for " + model_prefix )
                
    except Exception as e:
        print("ERROR when loading Categorical Classifier model dictionary from file: " + to_load)
        traceback.print_exc()
    
    try:
        #gets a list of ids to be predicted from file (as an argument in predict command)
        #and fetch data from db
        ids = None
        if inputfile is not None:
            ids = get_id_list_from_file(inputfile)
            INclause = str(ids).replace("]",")").replace("[","(")
            where_clause = " " + key_name + " IN " + INclause
            data = get_db_training_data(column_list, table_name, key_name, key_name, where_clause)
        else:
            data = get_object_description(ticket_id, table_name, key_name, column_list)
        
        language = get_language_parameter(parameters)
                
        prediction = None  
        final_output = None
        data[TEXT_COL] = "" 

        # Predict training data
        data = categorical_input_preprocess(data)
        if not categorical_cols_only:
            data[TEXT_COL] = data[TEXT_COL].map(lambda x: clean_text(x, language))
         
        inp_test = preprocessor.transform(data)

        y_train_pred = model.predict(inp_test)
        pred_values = le.inverse_transform(y_train_pred)
        
        if inputfile is not None:
            final_output = pred_values
        else:
            final_output  = pred_values[0]
        
        print("----- Predictions: ")
        results = pd.DataFrame(columns=["id","prediction"])
        if inputfile is not None:
            for j,id in enumerate(ids):
                r = [{"id":str(id),"prediction":final_output[j]}]
                results = results.append(r)

            print(tabulate(results,headers="keys",showindex="never"))
            save_predictions_in_db_batch(cron_id, ids, final_output)
       
        else:
            print(data.loc[0,:])
            print(final_output)
            save_predictions_in_db(cron_id, ticket_id, final_output)


    except RuntimeError as e:
        print("ERROR when making prediction with Categorical Classifier model ", fname)
        print(e)
        traceback.print_exc()
    
    
    
    print(" --- CATEGORICAL CLASSIFIER --- PREDICT END at: " + str(datetime.datetime.now()))
    


#########################################################################################################        
#####################               RECURSIVE TREE CLASSIFIER METHODS               #####################        
#########################################################################################################  
CONFORMAL_PREDICTION = False
CONFORMAL_SUFFIX = "__HINTS"
SEMANTIC_CNN_CONFORMAL_SUFFIX = "__SEMANTIC_CNN_HINTS"
SAVE_PREFIX_GLOBAL = ""


def recursive_tree_classifier_train_fast(column_list, table, table_key, target_col, where_clause = "", save_prefix = "", parameters = None):
    
    # Check if Model folder exists
    check_for_folders()
    
    classification_layers = target_col.split(',') 
    print("Classifying layers: ")
    print(classification_layers)

    global SAVE_PREFIX_GLOBAL
    SAVE_PREFIX_GLOBAL = save_prefix

    # classification final target is the last element of the classification_layers as is the goal of the TreeClassifier
    final_target = classification_layers[-1]
    
    # Get tickets ids, title and description
    print(" --- COLLECTING DATA FROM DB - START at: " + str(datetime.datetime.now()))
    data = get_db_training_data(column_list, table, table_key, classification_layers, where_clause, get_ids = True)
    print(" --- COLLECTING DATA FROM DB - END at: " + str(datetime.datetime.now()))
    #data.dropna(inplace=True)

    # # Fetch parameters from service table json
    Tree_config = get_Tree_config_params(parameters)
    
    #CONFORMAL PREDICTION switch
    global CONFORMAL_PREDICTION
    CONFORMAL_PREDICTION = Tree_config["tree_classifier_conformal_classification"]
    
    #column width in case of verbose output -> param tree_classifier_print_result_table = True
    tree_classifier_result_table_col_width = Tree_config["tree_classifier_result_table_col_width"] 

    # Filters out labels with cardinality < ", "min_cardinality" parameter value if set in "parameters" column in services table
    # filters based on the last layer that is the actual classification target 
    MIN_CARDINALITY = get_min_cardinality_parameter(parameters)
    filtered_data = filter_by_cardinality(data, final_target, MIN_CARDINALITY)

    print("Dataframe shape:")
    print(filtered_data.shape)
    
    # Fetch min_max_cardinality_ratio parameter from parameters column
    min_max_cardinality_ratio = get_min_max_cardinality_ratio_parameter(parameters)
    
    # pull out a subset of the dataset to be used as global test for the RecursiveTreeClassifier
    # stratify parameters makes the split proportional between labels (percentage of ticket per label will be the same in the splitted datasets)
    # dataframe_global_test is used to test the RecursiveTreeClassifier global performance
    # split proportions based on the last layer that is the actual classification target 
    proportional = True
    stratify_param = get_service_parameter_col(parameters,"stratify_split")
    if stratify_param is not None and stratify_param is False:
        proportional = False
   
    stratify = None
    if proportional:
        stratify = filtered_data[final_target]
    dataframe, dataframe_global_test = train_test_split(filtered_data, stratify=stratify,test_size=0.10)
    
    # balancing dataset by undersampling
    balanced_data = balance_dataset_by_undersampling(dataframe, final_target, min_max_cardinality_ratio, oversampling_ratio = 0.5)
    
    # store the model to use for every level/label
    tree_dictionary = dict()
    errors = dict()
    correct_predictions = dict()
    data_cols = column_list
        
    for l in classification_layers:
        correct_predictions[l] = 0
        errors[l] = 0
        if l in data_cols:
            data_cols.remove(l)

    recursive_tree_recursive_train(balanced_data, data_cols, classification_layers, 0, ROOT_MODEL_LABEL, save_prefix, column_list, table, table_key, parameters, tree_dictionary, Tree_config)
    
    # evaluate the global performance of the RecursiveTreeClassifier by predicting on the global test dataset dataframe_global_test
    print(" --- RECURSIVE TREE CLASSIFIER --- EVALUATE START at: " + str(datetime.datetime.now()))
    
    #initialize results dictionaries
    pred = dict()
    conformal_pred = dict()
    semantic_cnn_conformal_pred = dict()
    names = dict()
    results = dict()
    correct_counters = dict()
    error_counters = dict()
    reports = dict()
    confmats = dict()
    metrics = dict()
    final_output = dict()
    conformal_models_accuracy = dict()
    semantic_cnn_conformal_models_accuracy = dict()
    
    
    recursive_tree_classifier_evaluate_fast(dataframe_global_test, classification_layers, data_cols, 0, ROOT_MODEL_LABEL, save_prefix, tree_dictionary, parameters, pred, conformal_pred, semantic_cnn_conformal_pred, names, do_text_preprocess = True)
    
    print(" --- RECURSIVE TREE CLASSIFIER --- EVALUATE END at: " + str(datetime.datetime.now()))
    
    #display the results
    for k in classification_layers:
        results[k] = list(map(lambda x: str(x)[-tree_classifier_result_table_col_width:], pred[ k + TRUE_VALUE_SUFFIX ]))
        results[k + "_PRED"] = list(map(lambda x: ERROR_MESSAGE_MODEL_MISSING if pd.isnull(x) else str(x)[-tree_classifier_result_table_col_width:], pred[k]))
        results[k + "_MODEL"] = names[k]
        correct_counters[k] = np.sum(np.array(pred[k + TRUE_VALUE_SUFFIX]) == np.array(pred[k]))  
        error_counters[k] = len([n for n in names[k] if n == ERROR_MESSAGE_MODEL_MISSING])
        #confusion matrix and accuracy
        reports[k], confmats[k], metrics[k] = report_and_confmat(pred[ k + TRUE_VALUE_SUFFIX ], pred[k], save_prefix + "_RecursiveTreeClassifier - " + str(k), save_data=True)
        final_output[k] = metrics[k]["accuracy"]
                
    if CONFORMAL_PREDICTION:
        for k in classification_layers:
            print(" --- LAYER: " + k)
            print(" --- |--> CONFORMAL PREDICTION: ")
            print(" --- |--> ACCURACY: ", compute_model_accuracy(conformal_pred[k], np.array(pred[k + TRUE_VALUE_SUFFIX])))
            print("\n")
    
    #SEMANTIC_CNN
    enable_semantic_cnn_param = Tree_config["tree_classifier_enable_semantic_cnn"]
    ENABLE_SEMANTIC_CNN = enable_semantic_cnn_param is not None and enable_semantic_cnn_param is True
    
    if ENABLE_SEMANTIC_CNN:
        for k in classification_layers:
            print("\n")
            print(" --- LAYER: " + k)
            print(" --- |--> SEMANTIC CNN CONFORMAL PREDICTION: ")
            if semantic_cnn_conformal_pred[k] is not None and len(semantic_cnn_conformal_pred[k]) > 0:
                print(" --- |--> ACCURACY: ", compute_model_accuracy(semantic_cnn_conformal_pred[k], np.array(pred[k + TRUE_VALUE_SUFFIX])))
            else:
                print(" --- |--> ACCURACY: NO SEMANTIC MODEL CREATED IN TRAIN FOR THIS TARGET")
            print("\n")
    
    
    if Tree_config["tree_classifier_print_result_table"] is True:
        print(tabulate(results,headers="keys",showindex="never"))
        print("\n")
    
    predictions_count = str(len(pred[classification_layers[0]]))
    
    for k in classification_layers:
        print(" --- LAYER: " + k)
        print(" --- |--> CORRECT PREDICTIONS: " + str(correct_counters[k]) + "/" + predictions_count)
        print(" --- |--> ERRORS - MODEL NOT CREATED: " + str(error_counters[k]) + "/" + predictions_count)
        print("\n")
    
    #saving result tree to file
    save_path = os.path.join(os.getcwd(), "Models")
    filename = save_prefix + "_RecursiveTreeClassifier_results.pkl"
    full_filename = os.path.join(save_path, filename)
    with open(full_filename, "wb") as pklfile:
        pickle.dump(tree_dictionary, pklfile)

    #nicely print the tree dictionary 
    print("                                                     ")
    print("      *   =================================   *      ")
    print("     ***  === RECURSIVE TREE CLASSIFIER ===  ***     ")
    print("    ***** ===          RESULTS          === *****    ")
    print("      |   =================================   |      ")
    print("                                                     ")
    
  
    #pprint.pprint(tree_dictionary)
    save_dir = os.path.join(os.getcwd(), "Results")
    file = os.path.join(save_dir, save_prefix + "_RecursiveTreeClassifier_results.txt")
    with open(file, 'wt') as f:
        pprint.pprint(tree_dictionary, f) #saves to file
    
    return final_output


def recursive_tree_recursive_train(dataframe, data_cols, levels, target_index, model_idx, save_prefix, column_list, table, table_key, parameters, model_dict, Tree_config, guessed_language = None):

    # fetch the tree classifier metric threshold (min score value to consider the classification usable)
    tree_classifier_min_score =  Tree_config["tree_classifier_min_score"]
    # fetch the tree classifier min cardinality threshold (min cardinality of a class to be classified)
    tree_classifier_min_cardinality =  Tree_config["tree_classifier_min_cardinality"]
    
    #if the current target column is the language I use the predicted value for text preprocessing operations
    language_target_col = get_language_target_col_parameter(parameters)
    current_target = levels[target_index]
    
    save_path = os.path.join(os.getcwd(), "Models")
    
    # fetch the text language from db
    language_param = get_language_parameter(parameters)
    if guessed_language is None or guessed_language == "" and language_param is not None and language_param != "":
        guessed_language = language_param
    
    print("\n")
    print("******************************************************************")
    print("--- RECURSIVE TRAINING - model: ", save_prefix + "_" + model_idx + "  over target: " + str(current_target) )
    print("******************************************************************")    
    
    if guessed_language != None:
        print("----------------------------------------------------------------")
        print("--- FOUND LANGUAGE: ", guessed_language)
        print("----------------------------------------------------------------")
    else:
        print("----------------------------------------------------------------")
        print("--- LANGUAGE UNKNOWN --- CHECK THE language_target_col PARAMETER")
        print("----------------------------------------------------------------")
    
    #get the new subset, if its cardinality < tree_classifier_min_cardinality it gets excluded
    df = filter_by_cardinality(dataframe, current_target, tree_classifier_min_cardinality)
    
    #train the model over the current target and store in the dictionary
    try:
        model = train_classifiers(column_list, table, table_key, current_target, "", save_prefix + "_" + model_idx, parameters, df, tree_classifier_layers = levels, guessed_language = guessed_language , disable_min_cardinality_filter = True)
    except Exception as e:
        model = None
        model_dict[model_idx] = None
        print(e)
        traceback.print_exc()
        print("--- ERROR --- Error in training model " + save_prefix + "_" + model_idx + "  over target: " + str(current_target))
    
    
    #check  the model score
    if model is not None and (model["metric_value"] is None or (model["metric_value"] is not None and float(model["metric_value"]) < tree_classifier_min_score)):
        print("In level: " + str(current_target) + " excluding model: " + str(model_idx) + " for model classification score: "
                           + str(model["metric_value"]) + " < tree_classifier_min_score parameter (" + str(tree_classifier_min_score) + ")")
        model_dict[model_idx] = None

        # Delete the model file
        try:
            
            if model["model_name"] == "CNNClassifier":
                filename = os.path.join(save_path, save_prefix + "_" + model_idx + CNN_MODEL_FILENAME_SUFFIX)
                if os.path.exists(filename):
                    os.remove(filename)
                    print("Successfully deleted unused CNN model file: " + filename)
                
            dict_filename = os.path.join(save_path, save_prefix + "_" + model_idx + MODEL_FILENAME_SUFFIX)
            if os.path.exists(dict_filename):
                os.remove(dict_filename)
                print("Successfully deleted unused dict model file: " + dict_filename)
        except Exception as e:
            print("Error when deleting the model file " + dict_filename + "for score lower than tree_classifier_min_score ")
    else:    
        model_dict[model_idx] = model

    #reached the deepest tree level 
    if levels[-1] == current_target: return 
    
    #gets all the unique values for the current target
    unique_level_vals = np.unique(np.array(list(df[current_target])))
    
    #recursively run through the tree to create all the models
    for lval in unique_level_vals:
        if lval == ERROR_MESSAGE_MODEL_MISSING:
            continue
        mindex = set_modelid(model_idx,lval) #slugify values
       
        #get the new subset, if its cardinality < tree_classifier_min_cardinality it gets excluded
        #temp_df = filter_by_cardinality(df.loc[df[current_target] == lval], current_target, tree_classifier_min_cardinality)
        temp_df = df.loc[df[current_target] == lval]
        
        if temp_df is None or temp_df is not None and len(temp_df.index) == 0:
            print("In level: " + str(current_target) + " excluding label: " +   str(lval) + 
            " for label with cardinality < tree_classifier_min_cardinality parameter (" + str(tree_classifier_min_cardinality) + ")")
            model_dict[mindex] = None
            continue
            
        if current_target == language_target_col:
            guessed_language = get_language_string(lval) #get the lang code in ISO 639-1 Code format

            enable_semantic_cnn_param = Tree_config["tree_classifier_enable_semantic_cnn"]
            ENABLE_SEMANTIC_CNN = enable_semantic_cnn_param is not None and enable_semantic_cnn_param is True and guessed_language != '' and guessed_language is not None
            if ENABLE_SEMANTIC_CNN:
                # global var containing the gensim word2vec vocaboulary for the current language generated on the full dataset 
                # save the vocab to file for each language, to be loaded afterwards in the predict/evaluate phase
                ep, pp, sfp, cmtp = get_SEMANTIC_CNN_config_params(parameters)
                w2v_filename = os.path.join(save_path, save_prefix + "_" + WORD2VEC_VOCAB_FILENAME + "_" + guessed_language)
                w2v_df = pd.DataFrame(columns=["text"])
                w2v_df["text"] = ""
                for col in data_cols:
                    w2v_df["text"] = w2v_df["text"].map(str) + ' ' + temp_df[col].map(str)
                
                force_garbage_collection(verbose=1)
                
                global WORD2VEC_VOCAB
                if Tree_config["tree_classifier_load_pretrained_w2v_vocab"] is True:
                    WORD2VEC_VOCAB = generate_word2vec_vocab(w2v_df["text"], ep, pp, load_spacy(guessed_language), guessed_language, load_pretrained_filename = w2v_filename)
                else:
                    WORD2VEC_VOCAB = generate_word2vec_vocab(w2v_df["text"], ep, pp, load_spacy(guessed_language), guessed_language, save_filename = w2v_filename)

        recursive_tree_recursive_train(temp_df, data_cols, levels, target_index+1, mindex, save_prefix, column_list, table, table_key, parameters, model_dict, Tree_config, guessed_language = guessed_language)
    

def recursive_tree_classifier_evaluate_fast(data, levels, data_cols, target_index, model_idx, save_prefix, tree_dict, parameters, prediction_results, conformal_prediction_results, semantic_cnn_conformal_pred_results , model_names, guessed_language = None, do_text_preprocess = False):
    
    """
    Evaluates the performance of the RecursiveTreeClassifier over the given test dataset 
    Itarates over all the entries in the dataset and generates the confusion matrix for each layer of the classification hierarchy 
    :return: a dictionary like {"target_label_liv1":"predicted_label", "target_label_liv2":"predicted_label",....}
    """
    
    try:
        #THIS CHECK HAS TO BE THE FIRST INSTRUCTION DO BE EXECUTED IN THIS METHOD
        if target_index >= len(levels): 
            return

        #if the current target column is the language I use the predicted value for text preprocessing operations
        language_target_col = get_language_target_col_parameter(parameters)
        
        # fetch the text language from db
        language_param = get_language_parameter(parameters)
        if guessed_language is None or guessed_language == "" and language_param is not None and language_param != "":
            guessed_language = language_param
        
        if guessed_language != None:
            print("----------------------------------------------------------------")
            print("--- FOUND LANGUAGE: ", guessed_language)
            print("----------------------------------------------------------------")
        else:
            print("----------------------------------------------------------------")
            print("--- LANGUAGE UNKNOWN --- CHECK THE language_target_col PARAMETER")
            print("----------------------------------------------------------------")
        
        
        #ENABLE SEMANTIC_CNN PARAM
        Tree_config = get_Tree_config_params(parameters)
        enable_semantic_cnn_param = Tree_config["tree_classifier_enable_semantic_cnn"]
        ENABLE_SEMANTIC_CNN = enable_semantic_cnn_param is not None and enable_semantic_cnn_param is True and guessed_language != '' and guessed_language is not None
        global WORD2VEC_VOCAB
        
        if do_text_preprocess:
            data = text_preprocess(data, "text", parameters, data_cols, guessed_language)
        text = data["text"]
        
        current_level = levels[target_index]
        current_true_val_key = current_level + TRUE_VALUE_SUFFIX
        current_ids_key = current_level + IDS_COLUMN_SUFFIX
        
        #initialize results dictionaries
        if current_level not in semantic_cnn_conformal_pred_results.keys(): semantic_cnn_conformal_pred_results[current_level] = []
        if current_level not in conformal_prediction_results.keys(): conformal_prediction_results[current_level] = []
        if current_level not in prediction_results.keys(): prediction_results[current_level] = []
        if current_true_val_key not in prediction_results.keys(): prediction_results[current_true_val_key] = []
        if current_ids_key not in prediction_results.keys(): prediction_results[current_ids_key] = []
        if current_level not in model_names.keys(): model_names[current_level] = []
    
        if model_idx not in tree_dict.keys() or (model_idx in tree_dict.keys() and tree_dict[model_idx] is None):
            for l in levels[target_index:]:#set empty/error predictions all along the branch till the deepest level
                true_val_keys = l + TRUE_VALUE_SUFFIX
                ids_keys = l + IDS_COLUMN_SUFFIX
                
                if l not in semantic_cnn_conformal_pred_results.keys(): semantic_cnn_conformal_pred_results[l] = []
                if l not in conformal_prediction_results.keys(): conformal_prediction_results[l] = []
                if l not in prediction_results.keys(): prediction_results[l] = []
                if true_val_keys not in prediction_results.keys(): prediction_results[true_val_keys] = []
                if ids_keys not in prediction_results.keys(): prediction_results[ids_keys] = []
                if l not in model_names.keys(): model_names[l] = []
                
                for t in data["text"]: 
                    semantic_cnn_conformal_pred_results[l].extend([ERROR_MESSAGE_MODEL_MISSING])
                    conformal_prediction_results[l].extend([ERROR_MESSAGE_MODEL_MISSING])
                    prediction_results[l].extend([ERROR_MESSAGE_MODEL_MISSING])
                    model_names[l].extend([ERROR_MESSAGE_MODEL_MISSING])
                prediction_results[true_val_keys].extend(data[l])
                prediction_results[ids_keys].extend(data[IDS_COLUMN_SUFFIX])
                print("--- Evaluating layer: "+ str(current_level) +" dataset lenght: " + str(len(data.index)) + " -- using model: " + str(model_idx) + " -- " + ERROR_MESSAGE_MODEL_MISSING)
            return
        
        print("--- Evaluating layer: "+ str(current_level) +" dataset lenght: " + str(len(data.index)) + " using model: " + str(model_idx))

        if len(data.index) == 0:
            print("--- empty tree branch --- going on with next one...")
            return
        
        #load the model from file       
        model_folder = os.path.join(os.getcwd(), "Models")
        model_name = tree_dict[model_idx]["model_name"]
        model_filename = save_prefix + "_" + model_idx + MODEL_FILENAME_SUFFIX
        to_load = os.path.join(model_folder, model_filename)
        
        with open(to_load, "rb") as file:
            model_dict = pickle.load(file)
        model_date = str(model_dict["createdtime"])
        vectorizer = model_dict["vectorizer"]
        le = model_dict["LabelEncoder"]
        model = model_dict["model"]
        
        # if predicting with CNN Classifier it needs to load the model and weights from h5 file
        CNNmodel = None
        prediction = None
        conformal_model = None
        conformal_predictions = []
        semantic_cnn_conformal_predictions = []
        conformal_model_accuracy = None
        semantic_cnn_conformal_model_accuracy = None
        Tree_config = get_Tree_config_params(parameters)
        significance = Tree_config["tree_classifier_conformal_significance"]
        
        if model_name == "CNNClassifier":
            #CNNmodel = load_model(os.path.join(model_folder,model_dict["CNN_model_file"]))
            CNNmodel = load_model(os.path.join(model_folder,save_prefix + "_" + model_idx + CNN_MODEL_FILENAME_SUFFIX))
            CNN_config = get_CNN_config_params(parameters)
            prediction = get_CNN_predictions_confidence(CNNmodel, text, CNN_config, model_dict["CNN_tokenizer"], model_dict["CNN_labelencoder"])
            pred_values = list(map(lambda x: str(x), prediction[0]))
            if CONFORMAL_PREDICTION:
                conformal_model = model_dict["conformal_model"]
                conformal_predictions = conformal_model.predict(text, significance = significance)
        else:
            data_bow = vectorizer.transform(text)
            prediction = model.predict(data_bow)
            pred_values = le.inverse_transform(prediction)
            if CONFORMAL_PREDICTION:
                conformal_model = model_dict["conformal_model"]
                pred_labels = conformal_model.predict(data_bow, significance = significance)
                pred_indices = [np.where(p==True)[0] for p in pred_labels]
                #pred_probas = conformal_model.predict(data_bow)
                #prediction probabs sorted by ascending probability
                #ppsorted = np.argsort(pred_probas, axis=1)
                for pi in pred_indices:
                #for pi in ppsorted:
                    if len(pi) == 0:
                        conformal_predictions.append([""])
                    else:
                        #prediction labels sorted by descending probability, filter now by significance
                        #conformal_predictions.append(np.flip(le.inverse_transform(pi)))
                        conformal_predictions.append(le.inverse_transform(pi))
                
                if ENABLE_SEMANTIC_CNN:
                    #semantic_cnn_conformal_model = model_dict["semantic_cnn_conformal_model"]
                    semantic_cnn_object_filename = os.path.join(model_folder, save_prefix + "_" + model_idx + SEMANTIC_CNN_MODEL_CLASS_FILENAME_SUFFIX)
                    semantic_cnn_model_filename = os.path.join(model_folder, save_prefix + "_" + model_idx + SEMANTIC_CNN_MODEL_FILENAME_SUFFIX)
                    semantic_cnn_w2v_vocab = os.path.join(model_folder, save_prefix + "_" + WORD2VEC_VOCAB_FILENAME + "_" + guessed_language)
                    if WORD2VEC_VOCAB == None and os.path.exists(semantic_cnn_w2v_vocab):
                        print("--- LOADING PRETRAINED WORD2VEC_VOCAB for set language attribute:",guessed_language," from file:", semantic_cnn_w2v_vocab)
                        WORD2VEC_VOCAB = gensim.models.Word2Vec.load(semantic_cnn_w2v_vocab)

                    # check if the semantic cnn model files exist (in case of unsupported languages or errors during training)
                    if os.path.exists(semantic_cnn_object_filename) and  os.path.exists(semantic_cnn_model_filename) and WORD2VEC_VOCAB is not None: #os.path.exists(semantic_cnn_w2v_vocab):
                        with open(semantic_cnn_object_filename, "rb") as file:
                            semantic_cnn_object = pickle.load(file)
                        semantic_cnn_object.load_internals(model_folder, semantic_cnn_model_filename, WORD2VEC_VOCAB, guessed_language)
                        pred_labels = semantic_cnn_object.conformal_model.predict(semantic_cnn_object.input_preprocess(text), significance = significance)
                        pred_indices = [np.where(p==True)[0] for p in pred_labels]
                        for pi in pred_indices:
                            if len(pi) == 0:
                                semantic_cnn_conformal_predictions.append([""])
                            else:
                                #semantic_cnn_conformal_predictions.append(le.inverse_transform(pi))
                                semantic_cnn_conformal_predictions.append(semantic_cnn_object.le.inverse_transform(pi))
                    else:
                        for t in range(len(text)):
                            semantic_cnn_conformal_predictions.append(["ERROR SEMANTIC CNN MODEL NOT FOUND"])
                else:
                    for t in range(len(text)):
                        semantic_cnn_conformal_predictions.append([""])
                        #semantic_cnn_conformal_predictions.append(["SEMANTIC CNN DISABLED FOR THIS TARGET"])
   
    except Exception as e:
    
        #print("\n")
        print("--- ERROR in recursive_tree_classifier_evaluate_fast method --- model " + str(model_idx) + " target_index " + str(target_index))
        #print("--- WARNING --- try change tree_classifier_min_cardinality or tree_classifier_min_score parameters")
        traceback.print_exc()
        print("\n")
        return 
        
        
    semantic_cnn_conformal_pred_results[current_level].extend(semantic_cnn_conformal_predictions)
    conformal_prediction_results[current_level].extend(conformal_predictions)
    prediction_results[current_level].extend(pred_values)
    prediction_results[current_true_val_key].extend(data[current_level])
    prediction_results[current_ids_key].extend(data[IDS_COLUMN_SUFFIX])
    
    for p in pred_values: model_names[current_level].extend([model_name])
    
    #if evaluating a batch of texts, group the texts by same value of the previous layer
    #tmp_df = pd.DataFrame({"text":text,"pred":pred_values})
    data["pred"] = pred_values
    
    current_value = ""
    unique_pred_values = np.unique(prediction_results[current_level])
    for p in unique_pred_values:
        mindex = set_modelid(model_idx, p)
        
        if current_level == language_target_col:
            guessed_language = get_language_string(str(p)) #get the lang code in ISO 639-1 Code format
            do_text_preprocess = True #if I know the language I need to apply text preprocess operations that are language tuned
            semantic_cnn_w2v_vocab = os.path.join(model_folder, save_prefix + "_" + WORD2VEC_VOCAB_FILENAME + "_" + guessed_language)
            if os.path.exists(semantic_cnn_w2v_vocab):
                WORD2VEC_VOCAB = gensim.models.Word2Vec.load(semantic_cnn_w2v_vocab)
                print("--- LOADING PRETRAINED WORD2VEC VOCAB for guessed language:",guessed_language," from file:", semantic_cnn_w2v_vocab)
            else:
                WORD2VEC_VOCAB = None
        else: 
            do_text_preprocess = False
            
        #recursive call to go further down along the tree
        recursive_tree_classifier_evaluate_fast(data.loc[data["pred"] == p], levels, data_cols, target_index+1, mindex, save_prefix, tree_dict, parameters, prediction_results, conformal_prediction_results, semantic_cnn_conformal_pred_results, model_names, guessed_language=guessed_language, do_text_preprocess=do_text_preprocess)
    
    return


def recursive_tree_classifier_predict_fast(data_columns, cron_id, ticket_id, table_name, key_name, model_prefix, train_target_columns, parameters = None, inputfile = None, check_predictions = False):
    
    """
    Predicts the label of a given ticket using the RecursiveTreeClassifier. 
    Given the ticket_id it gets the text from the database and then it saves the result in the database
    :param ticket_id: integer corresponding to the ticket to classify
    :param parameters: contains a json with all the additional parameters (lemming,stemming,...)
    :inputfile: file containing data ids 
    :return: /
    """

    ### LOADING MODEL FROM FILE
    try:
        print(" --- RECURSIVE TREE CLASSIFIER --- PREDICT START at: " + str(datetime.datetime.now()))
        
        #load the tree dictionary from file
        model_folder = os.path.join(os.getcwd(), "Models")
        fname = model_prefix + "_RecursiveTreeClassifier_results.pkl"
        to_load = os.path.join(model_folder, fname)
        with open(to_load, "rb") as file:
            tree_dictionary = pickle.load(file)
        print("RecursiveTreeClassifier dictionary loaded from file: " + to_load)
        
                
    except Exception as e:
        print("ERROR when loading RecursiveTreeClassifier dictionary from file: " + to_load)
        traceback.print_exc()
        
    ### PREDICTING VALUE
    try:
        
        # # Fetch parameters from service table json
        Tree_config = get_Tree_config_params(parameters)
        tree_classifier_result_table_col_width = Tree_config["tree_classifier_result_table_col_width"] 
        #CONFORMAL PREDICTION switch
        global CONFORMAL_PREDICTION 
        CONFORMAL_PREDICTION = Tree_config["tree_classifier_conformal_classification"]
        
        #SEMANTIC_CNN
        enable_semantic_cnn_param = Tree_config["tree_classifier_enable_semantic_cnn"]
        ENABLE_SEMANTIC_CNN = enable_semantic_cnn_param is not None and enable_semantic_cnn_param is True
        
        data_cols = data_columns.split(',')
        target_cols = train_target_columns.split(',')
        
        # Fecth texts from db
        allcols = data_cols + target_cols
        
        #gets a list of ids to be predicted from file (as an argument in predict command)
        #and fetch data from db
        fetched_ids = None
        
        if inputfile is not None:
            ids = get_id_list_from_file(inputfile)
            INclause = str(ids).replace("]",")").replace("[","(")
            where_clause = " " + key_name + " IN " + INclause
            data = get_db_training_data(allcols, table_name, key_name, key_name, where_clause, get_ids = True)
        else:
            data = get_object_description(ticket_id, table_name, key_name, allcols)
            data[IDS_COLUMN_SUFFIX] = [ticket_id]

        fetched_ids = data[IDS_COLUMN_SUFFIX]
                    
        conformal_pred = dict()
        semantic_cnn_conformal_pred = dict()
        pred = dict()
        names = dict()
        results = dict()
        error_counters = dict()
        correct_counters = dict()
        reports = dict()
        confmats = dict()
        metrics = dict()
        print_check_predictions = dict()
        
        recursive_tree_classifier_evaluate_fast(data, target_cols, data_cols, 0, ROOT_MODEL_LABEL, model_prefix, tree_dictionary, parameters, pred, conformal_pred, semantic_cnn_conformal_pred, names, do_text_preprocess = True)
        
        results["id"] = fetched_ids
        
        ### PRINT RESULTS
        for k in target_cols:
            current_true_val_key = k + TRUE_VALUE_SUFFIX
            if check_predictions is True and current_true_val_key in pred.keys():
                print_check_predictions[k] = True
                reports[k], confmats[k], metrics[k] = report_and_confmat(pred[current_true_val_key], pred[k], model_prefix + "_RecursiveTreeClassifier_TEST_PREDICT_" + str(k), save_data=False)
                correct_counters[k] = np.sum(np.array(pred[current_true_val_key]) == np.array(pred[k]))
                results[k] = list(map(lambda x: str(x)[-tree_classifier_result_table_col_width:], pred[current_true_val_key]))
                
                if metrics[k] is None or reports[k] is None or confmats[k] is None:
                    print_check_predictions[k] = False
 
            results[k + "_PRED"] = list(map(lambda x: ERROR_MESSAGE_MODEL_MISSING if pd.isnull(x) else str(x)[-tree_classifier_result_table_col_width:], pred[k]))
            results[k + "_MODEL"] = names[k]
            results[k + IDS_COLUMN_SUFFIX] = pred[k + IDS_COLUMN_SUFFIX]
            
            error_counters[k] = len([n for n in names[k] if n == ERROR_MESSAGE_MODEL_MISSING])
        
        if Tree_config["tree_classifier_print_result_table"] is True:
            print(tabulate(results,headers="keys",showindex="never"))
            print("\n")

        predictions_count = len(pred[target_cols[0]])
        
        if CONFORMAL_PREDICTION and check_predictions is True:
            for k in target_cols:
                print("\n")
                print(" --- LAYER: " + k)
                print(" --- |--> CONFORMAL PREDICTION: ")
                print(" --- |--> ACCURACY: ", compute_model_accuracy(conformal_pred[k], np.array(pred[k + TRUE_VALUE_SUFFIX])))
                print("\n")
        
        if ENABLE_SEMANTIC_CNN and check_predictions is True:
            for k in target_cols:
                print("\n")
                print(" --- LAYER: " + k)
                print(" --- |--> SEMANTIC CNN CONFORMAL PREDICTION: ")
                print(" --- |--> ACCURACY: ", compute_model_accuracy(semantic_cnn_conformal_pred[k], np.array(pred[k + TRUE_VALUE_SUFFIX])))
                print("\n")

        
        for k in target_cols:
            print(" --- LAYER: " + k)
            print(" --- |--> ERRORS - MODEL NOT CREATED: " + str(error_counters[k]) + "/" + str(predictions_count))
            if check_predictions is True:
                if print_check_predictions[k] is True:
                    print(" --- |--> CORRECT PREDICTIONS: " + str(correct_counters[k]) + "/" + str(predictions_count))
                    print(" --- |--> ACCURACY: " + str(metrics[k]["accuracy"]))
                else: 
                    print(" --- |--> CORRECT PREDICTIONS: True Values not found in table " + str(table_name))
                    print(" --- |--> ACCURACY: True Values not found in table " + str(table_name))

        #initialize the final output dictionary
        final_output = dict()                
        for id in fetched_ids:
            final_output[id] = dict()
        
        #collects all the predictions made in  a json format to be saved in db 
        for j in range(predictions_count):
            for t in target_cols:
                final_output[ pred[t+IDS_COLUMN_SUFFIX][j] ][t] = pred[t][j] 
                if CONFORMAL_PREDICTION:
                    final_output[ pred[t+IDS_COLUMN_SUFFIX][j] ][t + CONFORMAL_SUFFIX] = np.array(conformal_pred[t][j]).tolist()
                if ENABLE_SEMANTIC_CNN:
                    final_output[ pred[t+IDS_COLUMN_SUFFIX][j] ][t + SEMANTIC_CNN_CONFORMAL_SUFFIX] = np.array(semantic_cnn_conformal_pred[t][j]).tolist()
               
        save_predictions_in_db_batch(cron_id, list(final_output.keys()), list(map(lambda x: json.dumps(x),final_output.values())))
        
        print(" --- RECURSIVE TREE CLASSIFIER --- PREDICT END at: " + str(datetime.datetime.now()))
        
    except Exception as e:
        print("Error when predicting with RecursiveTreeClassifier")
        print(e)
        traceback.print_exc()


    return






        
#########################################################################################################        
#####################                CRAP CLASSIFIER METHODS                        #####################        
#########################################################################################################        

CRAP_LABEL = "CRAP"
CRAP_SUFFIX = "CRAP_"
CRAP_COL = "CRAP_COL"
###CRAP_EXCLUDED = "CRAP_EXCLUDED"

def crap_classifier_train(column_list, table, table_key, target_col, where_clause = "", save_prefix = "", parameters = None):
    
    """
    Train the model of CrapClassifier
    """
    
    # Check if Model folder exists
    check_for_folders()
    
    # Get tickets ids, title and description
    print("Fetching data from db...")
    data = get_db_training_data(column_list, table, table_key, target_col, where_clause)
    #data.dropna(inplace=True)
    
    # # Fetch parameters from service table json
    cc_config = get_cc_config_params(parameters)
    # fetch the crap classifier metric threshold (min score value to consider the classification usable)
    cc_min_score = cc_config["cc_min_score"]
    # fetch the crap classifier min cardinality filter threshold (min cardinality of a class to be classified)
    cc_min_cardinality = cc_config["cc_min_cardinality"]
    # fetch the crap classifier min cardinality ratio (between the current class and the largest class of the dataset)
    cc_min_cardinality_ratio = cc_config["cc_min_cardinality_ratio"]
    
    
    # Filters out labels with cardinality < ", "min_cardinality" parameter value if set in "parameters" column in services table
    # filters based on the last layer that is the actual classification target 
    MIN_CARDINALITY = get_min_cardinality_parameter(parameters)
    filtered_data = filter_by_cardinality(data, target_col, MIN_CARDINALITY)
    print("Dataframe shape:")
    print(filtered_data.shape)

    # pull out a subset of the dataset to be used as global test for the CrapClassifier
    # stratify parameters makes the split proportional between labels (percentage of ticket per label will be the same in the splitted datasets)
    # dataframe_global_test is used to test the CrapClassifier global performance
    # split proportions based on the last layer that is the actual classification target 
    proportional = True
    stratify_param = get_service_parameter_col(parameters,"stratify_split")
    if stratify_param is not None and stratify_param is False:
        proportional = False

    stratify = None
    if proportional:
        stratify = filtered_data[target_col]
    dataframe, dataframe_global_test = train_test_split(filtered_data, stratify=stratify,test_size=0.10)
    
    #store the model to use for every level/label
    cc_dictionary = dict()
    cc_dictionary = recursive_train(column_list, table, table_key, target_col, where_clause, save_prefix, parameters, dataframe, cc_dictionary, 0)
    
    # EVALUATE
    # evaluate the global performance of the CrapClassifier by predicting on the global test dataset dataframe_global_test
    print("--- CRAP CLASSIFIER --- EVALUATE START at: " + str(datetime.datetime.now()))
    print("--- Evaluation running on: "+ str(len(dataframe_global_test.index)) +" samples")
    final_pred = pd.DataFrame(columns=["true_val","pred"])
    
    ###pdb.set_trace()
    ###dft = pd.DataFrame(columns=dataframe_global_test.columns)
    ###f = open("logs/" + model_prefix + "_" + model_name + "_train_test_data", "r")
    ###for x in f:
    ###    dft = dft.append([{"ticket_title":" ","description":x,"issue_liv_2":}])
    
    
    ###predictions = crap_classifier_evaluate_multiclass(dft, target_col, save_prefix, cc_dictionary, final_pred, parameters)
    predictions = crap_classifier_evaluate(dataframe_global_test, target_col, save_prefix, cc_dictionary, final_pred, parameters)
    #dump_var_to_file(predictions, save_prefix + "_predictions_FINAL", "logs")
    print("--- CRAP CLASSIFIER --- EVALUATE END at: " + str(datetime.datetime.now()))

    final_output = dict()
    reports, confmat, metrics = report_and_confmat(predictions.loc[:,"true_val"].values, predictions.loc[:,"pred"].values, save_prefix + "_CrapClassifier - " + str(target_col), save_data=True)
    final_output["Model"] = "CrapClassifier"
    final_output["Score"] = metrics["accuracy"]
    
    #saving results to file
    save_path = os.path.join(os.getcwd(), "Models")
    filename = save_prefix + "_CrapClassifier_results.pkl"
    full_filename = os.path.join(save_path, filename)
    with open(full_filename, "wb") as pklfile:
        pickle.dump(cc_dictionary, pklfile)

    print("=================================================")
    print("============ CRAP CLASSIFIER RESULTS ============")
    print("=================================================")
    pprint.pprint(cc_dictionary)
    save_dir = os.path.join(os.getcwd(), "Results")
    file = os.path.join(save_dir, save_prefix + "_CrapClassifier_results.txt")
    with open(file, 'wt') as f:
        pprint.pprint(cc_dictionary, f) #saves to file
    
    return final_output


def recursive_train(column_list, table, table_key, target_col, where_clause, save_prefix, parameters, dataframe, cc_dictionary, index):
    
    """
    Implements the recursive train the model of CrapClassifier
    """
    
    print("=====================================================================")
    print("====== RECURSIVE TRAIN ===== round: " + str(index))
    print("=====================================================================")
    if DEBUG_MODE: pdb.set_trace()
    if dataframe is None: 
        print("--- WARNING --- TRAIN STOPPED")
        return cc_dictionary
    
    model_id = str(index) #set_modelid(target_col,index)
    prefix = save_prefix + "_" + model_id
    
    target = target_col
    ###df_temp = pd.DataFrame(columns = dataframe.columns)
    if CRAP_COL in dataframe.columns:
        target = CRAP_COL
        #drop all the rows with label CRAP_EXCLUDED
        ###df_temp = dataframe.drop(dataframe[dataframe[CRAP_COL] == CRAP_EXCLUDED].index,inplace=False)
        ###model = train_classifiers(column_list, table, table_key, target, where_clause, prefix, parameters, df_temp)
    ###else:
    model = train_classifiers(column_list, table, table_key, target, where_clause, prefix, parameters, dataframe, data_columns = column_list, disable_min_cardinality_filter = True)
    
    
    # # Fetch parameters from service table json
    cc_config = get_cc_config_params(parameters)
    # fetch the crap classifier metric threshold (min score value to consider the classification usable)
    cc_min_score = cc_config["cc_min_score"]
    # fetch the crap classifier min cardinality filter threshold (min cardinality of a class to be classified)
    cc_min_cardinality = cc_config["cc_min_cardinality"]
    # fetch the crap classifier min cardinality ratio (between the current class and the largest class of the dataset)
    cc_min_cardinality_ratio = cc_config["cc_min_cardinality_ratio"]
    
    score = model["metric_value"]
    # if the first train has score higher than cc_min_score it just stops
    if ((score is None or (score is not None and float(score) > cc_min_score)) and index == 0):
        print("--- Standard classification got a score of " + str(score) + " > cc_min_score parameter(" + str(cc_min_score) + ")")
        print("--- saving model " + model["model_name"] + " in cc_dictionary with index " + model_id + " ---")
        cc_dictionary[model_id] = model
        return cc_dictionary
    
    if CRAP_COL not in dataframe.columns:
        if DEBUG_MODE: pdb.set_trace()
        print("--- Model "+ model["model_name"] + " at id " + str(model_id) + " train returned score " + str(score))
        #print("--- Model "+ model["model_name"] + " at id " + str(model_id) + " train returned score " + str(score) + " < cc_min_score parameter (" + str(cc_min_score) + ")")
        # Delete the model file
        try:
            save_path = os.path.join(os.getcwd(), "Models")
            
            if model["model_name"] == "CNNClassifier":
                filename = os.path.join(save_path, prefix + CNN_MODEL_FILENAME_SUFFIX)
                if os.path.exists(filename):
                    os.remove(filename)
                    print("Successfully deleted unused CNN model file: " + filename)
                
            dict_filename = os.path.join(save_path, prefix + MODEL_FILENAME_SUFFIX)
            if os.path.exists(dict_filename):
                os.remove(dict_filename)
                print("Successfully deleted unused dict model file: " + dict_filename)
        except Exception as e:
            print("Error when deleting the model file " + dict_filename + "for score lower than cc_min_score ")
        
        #Merge in a single crap_class all the texts labeled in a class with cardinality ratio over max_cardinality <= cc_min_cardinality_ratio
        dataframe = crap_class_create(dataframe,target_col, cc_min_cardinality_ratio)
        
        if dataframe is None: 
            print("--- WARNING --- TRAIN STOPPED , try a lower value for cc_min_cardinality_ratio parameter")
            return cc_dictionary
        
        #recursive call with the new subset made of:
        # - the "meaningful" classes (max_cardinality <= cc_min_cardinality_ratio) enough to train the model
        # - the CRAP class containin all the remaining classes
        cc_dictionary = recursive_train(column_list, table, table_key, target_col, where_clause, save_prefix, parameters, dataframe, cc_dictionary, index+1)
    
    else:# the trained model returned a good score so it ends
        if DEBUG_MODE: pdb.set_trace()
        #print("--- Model "+ model["model_name"] + " at id " + str(model_id) + " got a score of " + str(score) + " > cc_min_score parameter(" + str(cc_min_score) + ")")
        print("--- Model "+ model["model_name"] + " at id " + str(model_id) + " got a score of " + str(score))
        print("--- saving model " + model["model_name"] + " in cc_dictionary with index " + model_id + " ---")
        cc_dictionary[model_id] = model
        
        if CRAP_COL in dataframe.columns:
            if DEBUG_MODE: pdb.set_trace()
            dataframe = crap_splitter(dataframe, target_col, cc_min_cardinality)
            
            if dataframe is None:
                print("--- WARNING --- TRAIN STOPPED , try a lower value for cc_min_cardinality parameter")
                return cc_dictionary
            
            #recursive call with the new subset made of ONLY:
            # - the classes previously selected as CRAP classes 
            cc_dictionary = recursive_train(column_list, table, table_key, target_col, where_clause, save_prefix, parameters, dataframe, cc_dictionary, index+1)
    
        
    return cc_dictionary

    
def crap_classifier_evaluate(data, target_col, model_prefix, cc_dictionary, evaluate_output, parameters, index = 0):
    
    """
    Performs the model evaluation of CrapClassifier
    """
    
    print("=====================================================================")
    print("====== RECURSIVE EVALUATE ===== round: " + str(index))
    print("=====================================================================")
    
    #in case the first standard train (no CRAP classes) did not return an high enough score, model ids starts with index = 1
    last_index = int(max(cc_dictionary, key=int))
    while str(index) not in cc_dictionary and index < last_index:
        index += 1
    
    try:
        #means there are no further models
        if index > last_index:
            print("--- WARNING --- evaluation finished, no further models")
            return pd.DataFrame(columns=evaluate_output.columns)
        
        #no further entries to be classified
        if len(data.index) == 0:
            print("--- WARNING --- evaluation finished, no further data")
            return pd.DataFrame(columns=evaluate_output.columns)
        
        model_id = str(index) #set_modelid(target_col,index)
            
        #load model data
        model_folder = os.path.join(os.getcwd(), "Models")
        model_name = cc_dictionary[model_id]["model_name"]
        model_filename = model_prefix + "_" + model_id + MODEL_FILENAME_SUFFIX
        to_load = os.path.join(model_folder, model_filename)
        
        with open(to_load, "rb") as file:
            model_dict = pickle.load(file)
        model_date = str(model_dict["createdtime"])
        vectorizer = model_dict["vectorizer"]
        column_list = model_dict["columns"]
        if target_col in column_list: column_list.remove(target_col)
        
        model = model_dict["model"]
        data = text_preprocess(data, "text", parameters, column_list)
        # if predicting with CNN Classifier it needs to load the model and weights from h5 file
        CNNmodel = None
        predictions = None

        print("--- Loading model " + model_id + " : " + model_name + " to evaluate " + str(len(data.index)) + " entries")
    
        if model_name == "CNNClassifier":
            #CNNmodel = load_model(os.path.join(model_folder,model_dict["CNN_model_file"]))
            CNNmodel = load_model(os.path.join(model_folder,model_prefix + "_" + model_id + CNN_MODEL_FILENAME_SUFFIX))
            CNN_config = get_CNN_config_params(parameters)
            pred = get_CNN_predictions_confidence(CNNmodel, data["text"], CNN_config, model_dict["CNN_tokenizer"], model_dict["CNN_labelencoder"])
            predictions = pred[0]
        else:
            data_bow = vectorizer.transform(data["text"])
            predictions = model.predict(data_bow)
    
        #dump_var_to_file(predictions, model_prefix + "_" + str(index) + "_" + model_name + "_predictions", "logs")
       
        #save the not-crap predictions 
        jj = 0
        crap_subset = pd.DataFrame(columns=data.columns)
        for idx,trueval in data[target_col].iteritems():
            if predictions[jj] == CRAP_LABEL:
                jj += 1
                crap_subset = crap_subset.append(data.loc[idx],ignore_index=True)
                continue
            evaluate_output = evaluate_output.append({"true_val":trueval,"pred":str(predictions[jj])},ignore_index=True)
            jj += 1
        
        print("--- Evaluated in total " + str(len(evaluate_output.index)) + " entries")
        
        #dump_var_to_file(evaluate_output, model_prefix + "_" + str(index) + "_" + model_name + "_evaluate_output", "logs")
        
        if CRAP_LABEL in predictions:
            print("--- Recursive call to evaluate the remaining crap data (" + str(len(crap_subset.index)) + " entries)")
            evaluate_output = crap_classifier_evaluate(crap_subset, target_col, model_prefix, cc_dictionary, evaluate_output, parameters, index+1)            
            #dump_var_to_file(evaluate_output, model_prefix + "_" + str(index) + "_" + model_name + "_evaluate_output_APPEND", "logs")
            
    except Exception as e:
        print("ERROR in CrapClassifier evaluate")
        print(e)
        traceback.print_exc()
    
    return evaluate_output


def crap_splitter(dataframe, target_col, cc_min_cardinality):
    """
    remove good classes and returns the new "clean" subset made of only the previous crap classes 
    """
    
    #gets label count for each label 
    label_counts = dataframe[CRAP_COL].value_counts(sort=True,ascending=False)
    if label_counts.index[0] == CRAP_LABEL: #doesn't count the previous CRAP class
        max_cardinality = label_counts[1] 
    else:
        max_cardinality = label_counts[0] 
    crap_unique_classes_count = len(dataframe.loc[dataframe[CRAP_COL] == CRAP_LABEL, target_col].drop_duplicates().index)
    
    if max_cardinality < cc_min_cardinality or crap_unique_classes_count < 2:
        return None

    dataframe = dataframe.loc[dataframe[CRAP_COL] == CRAP_LABEL]
    dataframe.drop(CRAP_COL, axis=1, inplace=True)
    
    return dataframe


def crap_class_create(dataframe,target_col, cc_min_cardinality_ratio):
    """
    merge in one "crap" class all the classes with cardinality ratio over the largest class cardinality < cc_min_cardinality_ratio
    """
    
    #if the dataframe contains already a "crap" class it has to return None to avoid infinite loops
    if CRAP_COL in dataframe.columns:
        return None
    
    #create a new target column to set the create the crap class
    dataframe[CRAP_COL] = dataframe[target_col]

    #gets label count for each label 
    label_counts = dataframe[CRAP_COL].value_counts(sort=True,ascending=False)
    max_cardinality = label_counts[0] 
    min_cardinality_by_ratio = max_cardinality * cc_min_cardinality_ratio
    
    label_counts_dataframe = label_counts.to_frame()
        
    #gets the labels with cardinality <= min_cardinality_by_ratio
    crap_labels = label_counts_dataframe[label_counts_dataframe[CRAP_COL] <= min_cardinality_by_ratio].index
    
    #assign the "CRAP" label to all the 
    dataframe.loc[dataframe[target_col].isin(crap_labels), CRAP_COL] = CRAP_LABEL
    
    return dataframe


def crap_classifier_predict(cron_id, ticket_id, table_name, key_name, model_prefix, train_target_column, parameters = None):
    
    """
    predicts a single entry with CrapClassifier using the recursive evaluation method
    """
    
    try:
        #load the CrapClassifier dictionary from file
        model_folder = os.path.join(os.getcwd(), "Models")
        fname = model_prefix + "_CrapClassifier_results.pkl"
        to_load = os.path.join(model_folder, fname)
        with open(to_load, "rb") as file:
            cc_dictionary = pickle.load(file)
        print("CrapClassifier dictionary loaded from file: " + to_load)
        
    except Exception as e:
        print("ERROR when loading CrapClassifier dictionary from file: " + to_load)
        traceback.print_exc()
    
    try:

        index = 0
        last_index = int(max(cc_dictionary, key=int))
        prediction = CRAP_LABEL
        
        while prediction == CRAP_LABEL:
        
            #get the next model index
            while str(index) not in cc_dictionary and index < last_index:
                index += 1

            model_id = str(index)
            
            #load model data
            model_folder = os.path.join(os.getcwd(), "Models")
            model_name = cc_dictionary[model_id]["model_name"]
            model_filename = model_prefix + "_" + model_idx +  MODEL_FILENAME_SUFFIX
            to_load = os.path.join(model_folder, model_filename)
            
            with open(to_load, "rb") as file:
                model_dict = pickle.load(file)
            model_date = str(model_dict["createdtime"])
            vectorizer = model_dict["vectorizer"]
            column_list = model_dict["columns"]
            if train_target_column in column_list: column_list.remove(train_target_column)
            model = model_dict["model"]

            print("Classifying with " + model_name + " created " + model_date)                
            # Fecth texts from db
            data = get_object_description(ticket_id, table_name, key_name, column_list)
            data = text_preprocess(data, "text", parameters, column_list)
            
            print("--- Input text: ")
            print(data.loc[0,"text"])
                    
            # if predicting with CNN Classifier it needs to load the model and weights from h5 file
            CNNmodel = None

            print("--- Loading model " + model_id + " : " + model_name)
            
            if model_name == "CNNClassifier":
                #print("--- Filename: " + model_dict["CNN_model_file"])
                #CNNmodel = load_model(os.path.join(model_folder,model_dict["CNN_model_file"]))
                print("--- Filename: " + model_prefix + "_" + model_id + CNN_MODEL_FILENAME_SUFFIX)
                CNNmodel = load_model(os.path.join(model_folder,model_prefix + "_" + model_id + CNN_MODEL_FILENAME_SUFFIX))
                CNN_config = get_CNN_config_params(parameters)
                pred = get_CNN_predictions_confidence(CNNmodel, data["text"], CNN_config, model_dict["CNN_tokenizer"], model_dict["CNN_labelencoder"])
                prediction = pred[0][0]
            else:
                print("--- Filename: " + model_filename)
                data_bow = vectorizer.transform(data["text"])
                prediction = (model.predict(data_bow))[0]
            
            print("--- Predicted value: " + prediction)
            
            index += 1

        print("CrapClassifier prediction: ")
        print(prediction)
        
        save_predictions_in_db(cron_id, ticket_id, prediction)
    
    
    except Exception as e:
        print("ERROR in CrapClassifier predict") 
        print(e)
        traceback.print_exc()
    
    return




    


#########################################################################################################        
#####################            CRAP CLASSIFIER MULTICLASS METHODS                 #####################        
#########################################################################################################        


def crap_classifier_train_multiclass(column_list, table, table_key, target_col, where_clause = "", save_prefix = "", parameters = None):
    
    """
    Train the model of CrapClassifier
    """
    
    # Check if Model folder exists
    check_for_folders()
    
    # Get tickets ids, title and description
    print("Fetching data from db...")
    data = get_db_training_data(column_list, table, table_key, target_col, where_clause)
    #data.dropna(inplace=True)
    
    # # Fetch parameters from service table json
    cc_config = get_cc_config_params(parameters)
    # fetch the crap classifier metric threshold (min score value to consider the classification usable)
    cc_min_score = cc_config["cc_min_score"]
    # fetch the crap classifier min cardinality filter threshold (min cardinality of a class to be classified)
    cc_min_cardinality = cc_config["cc_min_cardinality"]
    # fetch the crap classifier min cardinality ratio (between the current class and the largest class of the dataset)
    cc_min_cardinality_ratio = cc_config["cc_min_cardinality_ratio"]
    
    
    # Filters out labels with cardinality < ", "min_cardinality" parameter value if set in "parameters" column in services table
    # filters based on the last layer that is the actual classification target 
    MIN_CARDINALITY = get_min_cardinality_parameter(parameters)
    filtered_data = filter_by_cardinality(data, target_col, MIN_CARDINALITY)
    print("Dataframe shape:")
    print(filtered_data.shape)

    # pull out a subset of the dataset to be used as global test for the CrapClassifier
    # stratify parameters makes the split proportional between labels (percentage of ticket per label will be the same in the splitted datasets)
    # dataframe_global_test is used to test the CrapClassifier global performance
    # split proportions based on the last layer that is the actual classification target 
    proportional = True
    stratify_param = get_service_parameter_col(parameters,"stratify_split")
    if stratify_param is not None and stratify_param is False:
        proportional = False

    stratify = None
    if proportional:
        stratify = filtered_data[target_col]
    dataframe, dataframe_global_test = train_test_split(filtered_data, stratify=stratify,test_size=0.10)
    
    #store the model to use for every level/label
    cc_dictionary = dict()
    cc_dictionary = recursive_train_multiclass(column_list, table, table_key, target_col, where_clause, save_prefix, parameters, dataframe, cc_dictionary, 0)
    
    # EVALUATE
    # evaluate the global performance of the CrapClassifier by predicting on the global test dataset dataframe_global_test
    print("--- CRAP CLASSIFIER --- EVALUATE START at: " + str(datetime.datetime.now()))
    print("--- Evaluation running on: "+ str(len(dataframe_global_test.index)) +" samples")
    final_pred = pd.DataFrame(columns=["true_val","pred"])
    
    ###pdb.set_trace()
    ###dft = pd.DataFrame(columns=dataframe_global_test.columns)
    ###f = open("logs/" + model_prefix + "_" + model_name + "_train_test_data", "r")
    ###for x in f:
    ###    dft = dft.append([{"ticket_title":" ","description":x,"issue_liv_2":}])
    
    
    ###predictions = crap_classifier_evaluate_multiclass(dft, target_col, save_prefix, cc_dictionary, final_pred, parameters)
    predictions = crap_classifier_evaluate_multiclass(dataframe_global_test, target_col, save_prefix, cc_dictionary, final_pred, parameters)
    #dump_var_to_file(predictions, save_prefix + "_predictions_FINAL", "logs")
    print("--- CRAP CLASSIFIER --- EVALUATE END at: " + str(datetime.datetime.now()))

    final_output = dict()
    reports, confmat, metrics = report_and_confmat(predictions.loc[:,"true_val"].values, predictions.loc[:,"pred"].values, save_prefix + "_CrapClassifier - " + str(target_col), save_data=True)
    final_output["Model"] = "CrapClassifier"
    final_output["Score"] = metrics["accuracy"]
    
    #saving results to file
    save_path = os.path.join(os.getcwd(), "Models")
    filename = save_prefix + "_CrapClassifier_results.pkl"
    full_filename = os.path.join(save_path, filename)
    with open(full_filename, "wb") as pklfile:
        pickle.dump(cc_dictionary, pklfile)

    print("=================================================")
    print("============ CRAP CLASSIFIER RESULTS ============")
    print("=================================================")
    pprint.pprint(cc_dictionary)
    save_dir = os.path.join(os.getcwd(), "Results")
    file = os.path.join(save_dir, save_prefix + "_CrapClassifier_results.txt")
    with open(file, 'wt') as f:
        pprint.pprint(cc_dictionary, f) #saves to file
    
    return final_output
    

def recursive_train_multiclass(column_list, table, table_key, target_col, where_clause, save_prefix, parameters, dataframe, cc_dictionary, index):
    
    """
    Implements the recursive train the model of CrapClassifier
    """
    
    print("=====================================================================")
    print("====== RECURSIVE TRAIN ===== round: " + str(index))
    print("=====================================================================")
    if DEBUG_MODE: pdb.set_trace()
    if dataframe is None: 
        print("--- WARNING --- TRAIN STOPPED")
        return cc_dictionary
    
    model_id = str(index) #set_modelid(target_col,index)
    prefix = save_prefix + "_" + model_id
    
    target = target_col
    ###df_temp = pd.DataFrame(columns = dataframe.columns)
    if CRAP_COL in dataframe.columns:
        target = CRAP_COL
        #drop all the rows with label CRAP_EXCLUDED
        ###df_temp = dataframe.drop(dataframe[dataframe[CRAP_COL] == CRAP_EXCLUDED].index,inplace=False)
        ###model = train_classifiers(column_list, table, table_key, target, where_clause, prefix, parameters, df_temp)
    ###else:
    model = train_classifiers(column_list, table, table_key, target, where_clause, prefix, parameters, dataframe, data_columns = column_list, disable_min_cardinality_filter = True)
    
    
    # # Fetch parameters from service table json
    cc_config = get_cc_config_params(parameters)
    # fetch the crap classifier metric threshold (min score value to consider the classification usable)
    cc_min_score = cc_config["cc_min_score"]
    # fetch the crap classifier min cardinality filter threshold (min cardinality of a class to be classified)
    cc_min_cardinality = cc_config["cc_min_cardinality"]
    # fetch the crap classifier min cardinality ratio (between the current class and the largest class of the dataset)
    cc_min_cardinality_ratio = cc_config["cc_min_cardinality_ratio"]
    # fetch the crap classifier crap_classes_size_multiplier (multiplies the cardinality of the smallest non-crap class to obtain the crap classes cardinality)
    crap_classes_size_multiplier = cc_config["crap_classes_size_multiplier"]
    
    score = model["metric_value"]
    # if the first train has score higher than cc_min_score it just stops
    if ((score is None or (score is not None and float(score) > cc_min_score)) and index == 0):
        print("--- Standard classification got a score of " + str(score) + " > cc_min_score parameter(" + str(cc_min_score) + ")")
        print("--- saving model " + model["model_name"] + " in cc_dictionary with index " + model_id + " ---")
        cc_dictionary[model_id] = model
        return cc_dictionary
    
    if CRAP_COL not in dataframe.columns:
        if DEBUG_MODE: pdb.set_trace()
        print("--- Model "+ model["model_name"] + " at id " + str(model_id) + " train returned score " + str(score))
        #print("--- Model "+ model["model_name"] + " at id " + str(model_id) + " train returned score " + str(score) + " < cc_min_score parameter (" + str(cc_min_score) + ")")
        # Delete the model file
        try:
            save_path = os.path.join(os.getcwd(), "Models")
            
            if model["model_name"] == "CNNClassifier":
                filename = os.path.join(save_path, prefix + CNN_MODEL_FILENAME_SUFFIX)
                if os.path.exists(filename):
                    os.remove(filename)
                    print("Successfully deleted unused CNN model file: " + filename)
                
            #dict_filename = os.path.join(save_path, model["model_filename"])
            dict_filename = prefix + MODEL_FILENAME_SUFFIX
            if os.path.exists(dict_filename):
                os.remove(dict_filename)
                print("Successfully deleted unused dict model file: " + dict_filename)
        except Exception as e:
            print("Error when deleting the model file " + dict_filename + "for score lower than cc_min_score ")
        
        #Merge in a single crap_class all the texts labeled in a class with cardinality ratio over max_cardinality <= cc_min_cardinality_ratio
        dataframe = crap_class_create_multiclass(dataframe,target_col, cc_min_cardinality_ratio, crap_classes_size_multiplier)
        
        if dataframe is None: 
            print("--- WARNING --- TRAIN STOPPED , try a lower value for cc_min_cardinality_ratio parameter")
            return cc_dictionary
        
        #recursive call with the new subset made of:
        # - the classes "meaningful" (max_cardinality <= cc_min_cardinality_ratio) enough to train the model
        # - the CRAP class containinall the remaining classes
        cc_dictionary = recursive_train_multiclass(column_list, table, table_key, target_col, where_clause, save_prefix, parameters, dataframe, cc_dictionary, index+1)
    
    else:# the trained model returned a good score so it ends
        if DEBUG_MODE: pdb.set_trace()
        #print("--- Model "+ model["model_name"] + " at id " + str(model_id) + " got a score of " + str(score) + " > cc_min_score parameter(" + str(cc_min_score) + ")")
        print("--- Model "+ model["model_name"] + " at id " + str(model_id) + " got a score of " + str(score))
        print("--- saving model " + model["model_name"] + " in cc_dictionary with index " + model_id + " ---")
        cc_dictionary[model_id] = model
        
        if CRAP_COL in dataframe.columns:
            if DEBUG_MODE: pdb.set_trace()
            dataframe = crap_splitter_multiclass(dataframe, target_col, cc_min_cardinality)
            
            if dataframe is None:
                print("--- WARNING --- TRAIN STOPPED , try a lower value for cc_min_cardinality parameter")
                return cc_dictionary
            
            #recursive call with the new subset made of ONLY:
            # - the classes previously selected as CRAP classes 
            cc_dictionary = recursive_train_multiclass(column_list, table, table_key, target_col, where_clause, save_prefix, parameters, dataframe, cc_dictionary, index+1)
    
        
    return cc_dictionary
    

def crap_classifier_evaluate_multiclass(data, target_col, model_prefix, cc_dictionary, evaluate_output, parameters, index = 0):
    
    """
    Performs the model evaluation of CrapClassifier
    """
    
    print("=====================================================================")
    print("====== RECURSIVE EVALUATE ===== round: " + str(index))
    print("=====================================================================")
    
    
    #in case the first standard train (no CRAP classes) did not return an high enough score, model ids starts with index = 1
    last_index = int(max(cc_dictionary, key=int))
    while str(index) not in cc_dictionary and index < last_index:
        index += 1
    
    try:
        #means there are no further models
        if index > last_index:
            print("--- WARNING --- evaluation finished, no further models")
            return pd.DataFrame(columns=evaluate_output.columns)
        
        #no further entries to be classified
        if len(data.index) == 0:
            print("--- WARNING --- evaluation finished, no further data")
            return pd.DataFrame(columns=evaluate_output.columns)
        
        model_id = str(index) #set_modelid(target_col,index)
            
        #load model data
        model_folder = os.path.join(os.getcwd(), "Models")
        model_name = cc_dictionary[model_id]["model_name"]
        #model_filename = cc_dictionary[model_id]["model_filename"]
        model_filename = model_prefix + "_" + model_id + MODEL_FILENAME_SUFFIX
        to_load = os.path.join(model_folder, model_filename)
        
        with open(to_load, "rb") as file:
            model_dict = pickle.load(file)
        model_date = str(model_dict["createdtime"])
        vectorizer = model_dict["vectorizer"]
        column_list = model_dict["columns"]
        if target_col in column_list: column_list.remove(target_col)
        
        model = model_dict["model"]
       
        data = text_preprocess(data, "text", parameters, column_list)

        # if predicting with CNN Classifier it needs to load the model and weights from h5 file
        CNNmodel = None
        predictions = None

        print("--- Loading model " + model_id + " : " + model_name + " to evaluate " + str(len(data.index)) + " entries")
    
        if model_name == "CNNClassifier":
            #CNNmodel = load_model(os.path.join(model_folder,model_dict["CNN_model_file"]))
            CNNmodel = load_model(os.path.join(model_folder,model_prefix + "_" + model_id + CNN_MODEL_FILENAME_SUFFIX))
            CNN_config = get_CNN_config_params(parameters)
            pred = get_CNN_predictions_confidence(CNNmodel, data["text"], CNN_config, model_dict["CNN_tokenizer"], model_dict["CNN_labelencoder"])
            predictions = pred[0]
        else:
            data_bow = vectorizer.transform(data["text"])
            predictions = model.predict(data_bow)
    
    
        #dump_var_to_file(predictions, model_prefix + "_" + str(index) + "_" + model_name + "_predictions", "logs")
       
        #save the not-crap predictions 
        jj = 0
        crap_subset = pd.DataFrame(columns=data.columns)
        for idx,trueval in data[target_col].iteritems():
            if is_crap_label(predictions[jj],CRAP_SUFFIX):
                jj += 1
                crap_subset = crap_subset.append(data.loc[idx],ignore_index=True)
                continue
            evaluate_output = evaluate_output.append({"true_val":trueval,"pred":str(predictions[jj])},ignore_index=True)
            jj += 1
        
        print("--- Evaluated in total " + str(len(evaluate_output.index)) + " entries")
        
        #dump_var_to_file(evaluate_output, model_prefix + "_" + str(index) + "_" + model_name + "_evaluate_output", "logs")
        
        still_crap = False
        for p in predictions:
            if is_crap_label(p,CRAP_SUFFIX): still_crap = True
                    
        if still_crap:
            print("--- Recursive call to evaluate the remaining crap data (" + str(len(crap_subset.index)) + " entries)")
            evaluate_output = crap_classifier_evaluate_multiclass(crap_subset, target_col, model_prefix, cc_dictionary, evaluate_output, parameters, index+1)            
            #dump_var_to_file(evaluate_output, model_prefix + "_" + str(index) + "_" + model_name + "_evaluate_output_APPEND", "logs")
            
    except Exception as e:
        print("ERROR in CrapClassifier evaluate")
        print(e)
        traceback.print_exc()
    
    return evaluate_output


def crap_splitter_multiclass(dataframe, target_col, cc_min_cardinality):
    """
    remove good classes and returns the new "clean" subset made of only the previous crap classes 
    """
    
    #gets label count for each label 
    label_counts = dataframe[CRAP_COL].value_counts(sort=True,ascending=False)
    if is_crap_label(label_counts.index[0], CRAP_SUFFIX): #doesn't count the previous CRAP class
        max_cardinality = label_counts[1] 
    else:
        max_cardinality = label_counts[0]
        
    crap_unique_classes_count = len(dataframe.loc[dataframe[CRAP_COL].str.startswith(CRAP_SUFFIX), target_col].drop_duplicates().index)
    
    if max_cardinality < cc_min_cardinality or crap_unique_classes_count < 2:
        return None

    dataframe = dataframe.loc[dataframe[CRAP_COL].str.startswith(CRAP_SUFFIX)]
    dataframe.drop(CRAP_COL, axis=1, inplace=True)
    
    return dataframe
    

def crap_class_create_multiclass(dataframe,target_col, cc_min_cardinality_ratio, crap_classes_size_multiplier):
    """
    merge in N "crap" classes all the classes with cardinality ratio over the largest class cardinality < cc_min_cardinality_ratio
    where N = sum(all crap-classes cardinality) / mean_value(all not-crap classes cardinality)
    """
    
    #if the dataframe contains already a "crap" class it has to return None to avoid infinite loops
    if CRAP_COL in dataframe.columns:
        return None
    
    #create a new target column to set the create the crap class
    dataframe[CRAP_COL] = dataframe[target_col]

    #gets label count for each label 
    label_counts = dataframe[CRAP_COL].value_counts(sort=True,ascending=False)
    max_cardinality = label_counts[0]
    min_cardinality_by_ratio = max_cardinality * cc_min_cardinality_ratio
    
    label_counts_dataframe = label_counts.to_frame()
        
    #gets the labels with cardinality <= min_cardinality_by_ratio and the remaining good ones 
    good_labels_counts = label_counts_dataframe[label_counts_dataframe[CRAP_COL] > min_cardinality_by_ratio]
    crap_labels_counts = label_counts_dataframe[label_counts_dataframe[CRAP_COL] <= min_cardinality_by_ratio]
    crap_labels = crap_labels_counts.index
    
    #crap_class_max_cardinality = good_labels_counts[CRAP_COL].mean() // 1.5
    crap_class_max_cardinality = math.modf(good_labels_counts[CRAP_COL].min() * crap_classes_size_multiplier)[1] 
    #crap_classes_cardinality_total = crap_labels_counts[crap_labels_counts[CRAP_COL]].sum()
    
    #calc the number of crap classes to be created (with cardinality = crap_class_max_cardinality)
    #total_crap_classes = crap_classes_cardinality_total // crap_class_max_cardinality #integer division
    current_count = 0
    crap_label_idx = 0
    for cl in crap_labels:
        label_count = label_counts_dataframe.loc[label_counts_dataframe.index == cl,CRAP_COL][0]
        if current_count + label_count > crap_class_max_cardinality:
            current_count = 0
            crap_label_idx += 1
        current_count += label_count # sum up the rows associated to the current crap label
        #assign the "CRAP_idx" label to all the entries marked with a crap label 
        dataframe.loc[dataframe[target_col] == cl, CRAP_COL] = CRAP_SUFFIX + str(crap_label_idx)
        ###new crap label
        ###craplabel = CRAP_SUFFIX + str(crap_label_idx)
        ###dataframe.loc[dataframe[target_col] == cl, CRAP_COL] = craplabel
        ###keep only the "underfit_ratio" percentage from each crap class for the next train
        ###underfit_ratio = 0.3
        ###edx = int(label_count * underfit_ratio)
        ###dataframe.loc[dataframe.loc[dataframe[target_col] == cl, CRAP_COL].index[0:edx], CRAP_COL] = CRAP_EXCLUDED     
        
    return dataframe
    

def crap_classifier_predict_multiclass(cron_id, ticket_id, table_name, key_name, model_prefix, train_target_column, parameters = None):
    
    """
    predicts a single entry with CrapClassifier using the recursive evaluation method
    """
    
    try:
        #load the CrapClassifier dictionary from file
        model_folder = os.path.join(os.getcwd(), "Models")
        fname = model_prefix + "_CrapClassifier_results.pkl"
        to_load = os.path.join(model_folder, fname)
        with open(to_load, "rb") as file:
            cc_dictionary = pickle.load(file)
        print("CrapClassifier dictionary loaded from file: " + to_load)
        
    except Exception as e:
        print("ERROR when loading CrapClassifier dictionary from file: " + to_load)
        traceback.print_exc()
    
    try:

        index = 0
        last_index = int(max(cc_dictionary, key=int))
        prediction = CRAP_SUFFIX
        
        while is_crap_label(prediction,CRAP_SUFFIX):
        
            #get the next model index
            while str(index) not in cc_dictionary and index < last_index:
                index += 1

            model_id = str(index)
            
            #load model data
            model_folder = os.path.join(os.getcwd(), "Models")
            model_name = cc_dictionary[model_id]["model_name"]
            model_filename = model_prefix + "_" + model_id + MODEL_FILENAME_SUFFIX
            #model_filename = cc_dictionary[model_id]["model_filename"]
            to_load = os.path.join(model_folder, model_filename)
            
            with open(to_load, "rb") as file:
                model_dict = pickle.load(file)
            model_date = str(model_dict["createdtime"])
            vectorizer = model_dict["vectorizer"]
            column_list = model_dict["columns"]
            if train_target_column in column_list: column_list.remove(train_target_column)
            model = model_dict["model"]

            print("Classifying with " + model_name + " created " + model_date)                
            # Fecth texts from db
            data = get_object_description(ticket_id, table_name, key_name, column_list)
            data = text_preprocess(data, "text", parameters, column_list)
            
            print("--- Input text: ")
            print(data.loc[0,"text"])
                    
            # if predicting with CNN Classifier it needs to load the model and weights from h5 file
            CNNmodel = None

            print("--- Loading model " + model_id + " : " + model_name)
            
            if model_name == "CNNClassifier":
                #print("--- Filename: " + model_dict["CNN_model_file"])
                print("--- Filename: " + model_prefix + "_" + model_id + CNN_MODEL_FILENAME_SUFFIX)
                #CNNmodel = load_model(os.path.join(model_folder,model_dict["CNN_model_file"]))
                CNNmodel = load_model(os.path.join(model_folder,model_prefix + "_" + model_id + CNN_MODEL_FILENAME_SUFFIX))
                CNN_config = get_CNN_config_params(parameters)
                pred = get_CNN_predictions_confidence(CNNmodel, data["text"], CNN_config, model_dict["CNN_tokenizer"], model_dict["CNN_labelencoder"])
                prediction = pred[0][0]
            else:
                print("--- Filename: " + model_filename)
                data_bow = vectorizer.transform(data["text"])
                prediction = (model.predict(data_bow))[0]
            
            print("--- Predicted value: " + prediction)
            
            index += 1

        print("CrapClassifier prediction: ")
        print(prediction)
        
        save_predictions_in_db(cron_id, ticket_id, prediction)
    
    
    except Exception as e:
        print("ERROR in CrapClassifier predict") 
        print(e)
        traceback.print_exc()
    
    return



    
    
    
#########################################################################################################        
#####################                THRESHOLD FINDER METHODS                       #####################        
#########################################################################################################        


def threshold_find(column_list, table, table_key, target_col, where_clause = "", save_prefix = "", parameters = None):
    
    """
    Sorts the clsses by cardinality (descending) and trains the first N (see param threshold_finder - tf_train_classes_subset_size) classes 
    each time until the accuracy on the current subset is lower than min_accuracy (see param threshold_finder - tf_min_accuracy)
    
    """
    # Check if Model folder exists
    check_for_folders()
    
    # Get tickets ids, title and description
    print("Fetching data from db...")
    data = get_db_training_data(column_list, table, table_key, target_col, where_clause)
    #data.dropna(inplace=True)
    
    # # Fetch parameters from service table json
    tf_config = get_tf_config_params(parameters)
    # fetch the threshold_finder accuracy threshold (min accuracy value)
    tf_min_accuracy = tf_config["tf_min_accuracy"]
    # fetch the threshold_finder tf_train_classes_subset_size parameter (number of classes for each train step)
    tf_train_classes_subset_size = tf_config["tf_train_classes_subset_size"]
    
    #output results file
    save_dir = os.path.join(os.getcwd(), "Results")
    filename = os.path.join(save_dir, save_prefix + "_ThresholdFinder_results.txt")
    results_file = open(filename, "a")
    
    # Filters out labels with cardinality < ", "min_cardinality" parameter value if set in "parameters" column in services table
    # filters based on the last layer that is the actual classification target 
    filtered_data = filter_by_cardinality(data, target_col, get_min_cardinality_parameter(parameters))
    print("Dataframe shape:")
    print(filtered_data.shape)

    total_classes = len(filtered_data[target_col].unique())
    
    #splits the first <tf_train_classes_subset_size> classes and the remaining ones into two dfferent datasets
    dft, dfr, threshold_class = split_dataset(filtered_data, target_col, tf_train_classes_subset_size)
    score = 0
    
    while True: 
        model = train_classifiers(column_list, table, table_key, target_col, where_clause, save_prefix, parameters, dft)
        
        score = model["metric_value"]
        if (score is None or (score is not None and float(score) < tf_min_accuracy)):
            print_std_and_file("--- WARNING --- Classification got a score of " + str(score) + " < " + str(tf_min_accuracy) + " (tf_min_accuracy param)", results_file)
            print_std_and_file("--- WARNING --- Threshold reached at class: " + threshold_class, results_file)
            break
        
        print_std_and_file("--- Classification with: " + model["model_name"] + " got a score of " + str(score) + " > " + str(tf_min_accuracy) + " (tf_min_accuracy param)", results_file)
        print_std_and_file("--- Run training over " + str(tf_train_classes_subset_size) + " classes each training step", results_file)
        print_std_and_file("--- Trained classes: " + str((total_classes - len(dfr[target_col].unique()))) + " / " + str(total_classes), results_file)
        dft, dfr, threshold_class = split_dataset(dfr, target_col, tf_train_classes_subset_size)
        
    final_output = dict()
    final_output["Model"] = "Threshold_finder"
    final_output["Score"] = score
    final_output["Threshold_class"] = threshold_class
    
    results_file.close()
    
    return final_output
        
    
    
    
    
    
    

#########################################################################################################        
##############             OLD/UNSUED RECURSIVE TREE CLASSIFIER METHODS               ###################
#########################################################################################################  

def recursive_tree_classifier_train(column_list, table, table_key, target_col, where_clause = "", save_prefix = "", parameters = None):
    
    # Check if Model folder exists
    check_for_folders()
    
    classification_layers = target_col.split(',') 
    print("Classifying layers: ")
    print(classification_layers)

    # classification final target is the last element of the classification_layers as is the goal of the TreeClassifier
    final_target = classification_layers[-1]
    
    # Get tickets ids, title and description
    print(" --- COLLECTING DATA FROM DB - START at: " + str(datetime.datetime.now()))
    data = get_db_training_data(column_list, table, table_key, classification_layers, where_clause)
    print(" --- COLLECTING DATA FROM DB - END at: " + str(datetime.datetime.now()))
    #data.dropna(inplace=True)

    # # Fetch parameters from service table json
    Tree_config = get_Tree_config_params(parameters)
    #column width in case of verbose output -> param tree_classifier_print_result_table = True
    tree_classifier_result_table_col_width = Tree_config["tree_classifier_result_table_col_width"] 
    
        
    # Filters out labels with cardinality < ", "min_cardinality" parameter value if set in "parameters" column in services table
    # filters based on the last layer that is the actual classification target 
    filtered_data = filter_by_cardinality(data, final_target, Tree_config["tree_classifier_min_cardinality"])

    print("Dataframe shape:")
    print(filtered_data.shape)
    
    # Fetch min_max_cardinality_ratio parameter from parameters column
    min_max_cardinality_ratio = get_min_max_cardinality_ratio_parameter(parameters)
    
    #balancing dataset classes if imbalanced
    #balanced_data = balance_dataset_by_undersampling(filtered_data, final_target,min_max_cardinality_ratio=0.2)

    # pull out a subset of the dataset to be used as global test for the RecursiveTreeClassifier
    # stratify parameters makes the split proportional between labels (percentage of ticket per label will be the same in the splitted datasets)
    # dataframe_global_test is used to test the RecursiveTreeClassifier global performance
    # split proportions based on the last layer that is the actual classification target 
    proportional = True
    stratify_param = get_service_parameter_col(parameters,"stratify_split")
    if stratify_param is not None and stratify_param is False:
        proportional = False

    stratify = None
    if proportional:
        stratify = filtered_data[final_target]
    dataframe, dataframe_global_test = train_test_split(filtered_data, stratify=stratify,test_size=0.10)
    
    #BALANCING DATASET BY UNDERSAMPLING
    balanced_data = balance_dataset_by_undersampling(dataframe, final_target, min_max_cardinality_ratio)
    
    #store the model to use for every level/label
    tree_dictionary = dict()
    recursive_tree_recursive_train(balanced_data, classification_layers, 0, ROOT_MODEL_LABEL, save_prefix, column_list, table, table_key, parameters, tree_dictionary, Tree_config)
    
    # evaluate the global performance of the RecursiveTreeClassifier by predicting on the global test dataset dataframe_global_test
    print(" --- RECURSIVE TREE CLASSIFIER --- EVALUATE START at: " + str(datetime.datetime.now()))
    prediction_results = pd.DataFrame(columns=classification_layers)
    model_names = pd.DataFrame(columns=classification_layers)
    
    jj = 1
    tot = len(dataframe_global_test.index)
    errors = dict()
    correct_predictions = dict()
    data_cols = column_list
        
    for l in classification_layers:
        correct_predictions[l] = 0
        errors[l] = 0
        if l in data_cols:
            data_cols.remove(l)

    for datarow in dataframe_global_test.iterrows():

        print("\n")
        print("---- RecursiveTreeClassifier --- evaluating " + str(jj) + " of " + str(tot) + " ... " + str(datetime.datetime.now()))

        jj += 1
        pred = dict()
        names = dict()
        text = datarow[1]
        data = text_preprocess(text, "text", parameters, data_cols)
                
        recursive_tree_classifier_evaluate(text, classification_layers, 0, ROOT_MODEL_LABEL, tree_dictionary, parameters, pred, names)

        results = pd.DataFrame(columns=["Label","Correct/Predicted","Accuracy","TrueValue","Predicted Value","Model Name"])
        for k in classification_layers:
            if "ERROR" in pred.keys() and k not in pred.keys(): 
                errors[k] += 1
                a = [{"Label": k,
                      "Correct/Predicted": "-",
                      "Accuracy": "-",
                      "TrueValue": str(datarow[1][k])[-tree_classifier_result_table_col_width:],
                      "Predicted Value": str(pred["ERROR"])[-tree_classifier_result_table_col_width:],
                      "Model Name": "ERROR"                  
                      }]
            else: 
                if pred[k] == datarow[1][k]: correct_predictions[k] += 1
                partial_accuracy = float("{:.2f}".format(correct_predictions[k] * 100 / jj))
                a = [{"Label": k,
                      "Correct/Predicted": str(correct_predictions[k]) + "/" + str(jj),
                      "Accuracy": str(partial_accuracy),
                      "TrueValue": str(datarow[1][k])[-tree_classifier_result_table_col_width:],
                      "Predicted Value": str(pred[k])[-tree_classifier_result_table_col_width:],
                      "Model Name": names[k]                  
                      }]
                      
            results = results.append(a)
        
        print(tabulate(results,headers="keys",showindex="never"))
        
        prediction_results = prediction_results.append([pred],ignore_index=True)
        model_names = model_names.append([names],ignore_index=True)
        reset_keras()
    
    print("\n")
    print(" --- CORRECT PREDICTIONS: ")
    for l in classification_layers:
        print(" --- " + str(l) + ": " + str(correct_predictions[l]) + " / " + str(jj))
        
    print(" --- ERRORS / MODEL NOT CREATED (try to decrease the tree_classifier_min_score parameter!): ")
    for l in classification_layers:
        print(" --- " + str(l) + ": " + str(errors[l]) + " / " + str(jj))

        
    print(" --- RECURSIVE TREE CLASSIFIER --- EVALUATE END at: " + str(datetime.datetime.now()))
    
    reports = dict()
    confmats = dict()
    metrics = dict()
    final_output = dict()
    for layer in classification_layers:
        #remove NaN values for the labels not predicted because no model was created (excluded by tree_classifier_min_cardinality or tree_classifier_min_score)
        cleaned_predictions = []
        cleaned_global_test = []
        ii = 0
        for g in dataframe_global_test[layer]:
            p = prediction_results[layer][ii]
            if pd.isnull(p): 
                p = "ERROR - NO MODEL CREATED"
            cleaned_predictions.append(str(p))
            cleaned_global_test.append(str(g))
            ii += 1

        if len(cleaned_predictions) == 0 or len(cleaned_global_test) == 0:
            print("--- WARNING --- no model created for " + str(layer) + " -- try change tree_classifier_min_cardinality or tree_classifier_min_score parameters")
            final_output[layer] = "0"
            continue
            
        reports[layer], confmats[layer], metrics[layer] = report_and_confmat(cleaned_global_test, cleaned_predictions, save_prefix + "_RecursiveTreeClassifier - " + str(layer), save_data=True)
        final_output[layer] = metrics[layer]["accuracy"]
    
    
    #saving result tree to file
    save_path = os.path.join(os.getcwd(), "Models")
    filename = save_prefix + "_RecursiveTreeClassifier_results.pkl"
    full_filename = os.path.join(save_path, filename)
    with open(full_filename, "wb") as pklfile:
        pickle.dump(tree_dictionary, pklfile)

    #nicely print the tree dictionary 
    print("                                                     ")
    print("      *   =================================   *      ")
    print("     ***  === RECURSIVE TREE CLASSIFIER ===  ***     ")
    print("    ***** ===          RESULTS          === *****    ")
    print("      |   =================================   |      ")
    print("                                                     ")
    #pprint.pprint(tree_dictionary)
    save_dir = os.path.join(os.getcwd(), "Results")
    file = os.path.join(save_dir, save_prefix + "_RecursiveTreeClassifier_results.txt")
    with open(file, 'wt') as f:
        pprint.pprint(tree_dictionary, f) #saves to file
    
    return final_output 


def recursive_tree_classifier_evaluate(data, levels, target_index, model_idx, tree_dict, parameters, prediction_results, model_names):
    
    """
    Evaluates the performance of the RecursiveTreeClassifier over the given test dataset 
    Itarates over all the entries in the dataset and generates the confusion matrix for each layer of the classification hierarchy 
    :return: a dictionary like {"target_label_liv1":"predicted_label", "target_label_liv2":"predicted_label",....}
    """

    try:
        if target_index >= len(levels): 
            return prediction_results, model_names
        
        current_level = levels[target_index]
                                      
        #load the model from file       
        model_folder = os.path.join(os.getcwd(), "Models")
        model_name = tree_dict[model_idx]["model_name"]
        model_filename = tree_dict[model_idx]["model_filename"] 
        to_load = os.path.join(model_folder, model_filename)
        
        with open(to_load, "rb") as file:
            model_dict = pickle.load(file)
        model_date = str(model_dict["createdtime"])
        vectorizer = model_dict["vectorizer"]
        
        model = model_dict["model"]
        
        # if predicting with CNN Classifier it needs to load the model and weights from h5 file
        CNNmodel = None
        prediction = None
        if type(data["text"]) is str:
            text = pd.Series(data["text"])
        else:
            text = data["text"]
        model_names[current_level] = model_name
        
        if model_name == "CNNClassifier":
            CNNmodel = load_model(os.path.join(model_folder,model_dict["CNN_model_file"]))
            CNN_config = get_CNN_config_params(parameters)
            prediction = get_CNN_predictions_confidence(CNNmodel, text, CNN_config, model_dict["CNN_tokenizer"], model_dict["CNN_labelencoder"])
            prediction_results[current_level] = str(prediction[0][0])
        else:
            data_bow = vectorizer.transform(text)
            prediction = model.predict(data_bow)
            prediction_results[current_level] = prediction[0]
        
        mindex = set_modelid(model_idx,prediction_results[current_level])
        
        #recursive call to go further down along the tree
        recursive_tree_classifier_evaluate(data, levels, target_index+1, mindex, tree_dict, parameters, prediction_results, model_names)
        
    except Exception as e:
        prediction_results["ERROR"] = str(e)
        model_names["ERROR"] = str(e)
    
    return prediction_results, model_names


def recursive_tree_classifier_predict(data_columns, cron_id, ticket_id, table_name, key_name, model_prefix, train_target_columns, parameters = None):
    
    """
    Predicts the label of a given ticket using the RecursiveTreeClassifier. 
    Given the ticket_id it gets the text from the database and then it saves the result in the database
    :param ticket_id: integer corresponding to the ticket to classify
    :param parameters: contains a json with all the additional parameters (lemming,stemming,...)
    :return: /
    """

    ### LOADING MODEL FROM FILE
    try:
        #load the tree dictionary from file
        model_folder = os.path.join(os.getcwd(), "Models")
        fname = model_prefix + "_RecursiveTreeClassifier_results.pkl"
        to_load = os.path.join(model_folder, fname)
        with open(to_load, "rb") as file:
            tree_dictionary = pickle.load(file)
        print("RecursiveTreeClassifier dictionary loaded from file: " + to_load)
        
                
    except Exception as e:
        print("ERROR when loading RecursiveTreeClassifier dictionary from file: " + to_load)
        traceback.print_exc()
        
    ### PREDICTING VALUE
    try:
        
        # # Fetch parameters from service table json
        Tree_config = get_Tree_config_params(parameters)
        tree_classifier_result_table_col_width = Tree_config["tree_classifier_result_table_col_width"] 
        
        data_cols = data_columns.split(',')
        target_cols = train_target_columns.split(',')
        
        # Fecth texts from db
        allcols = data_cols + target_cols
        data = get_object_description(ticket_id, table_name, key_name, allcols)
        data = text_preprocess(data, "text", parameters, data_cols)
                    
        print("--- Input text: ")
        print(data.loc[0,"text"])
        
        pred = dict()
        names = dict()
       
        recursive_tree_classifier_evaluate(data, target_cols, 0, ROOT_MODEL_LABEL, tree_dictionary, parameters, pred, names)
    
        ### PRINT RESULTS 
        results = pd.DataFrame(columns=["Label","TrueValue","Predicted Value","Model Name"])
        for k in target_cols:
            if "ERROR" in pred.keys() and k not in pred.keys(): 
                a = [{"Label": k,
                      "TrueValue": str(data[k][0])[-tree_classifier_result_table_col_width:],
                      "Predicted Value": "ERROR - " + str(pred["ERROR"])[-tree_classifier_result_table_col_width:],
                      "Model Name": "ERROR"                  
                      }]
            else: 
                a = [{"Label": k,
                      "TrueValue": str(data[k][0])[-tree_classifier_result_table_col_width:],
                      "Predicted Value": str(pred[k])[-tree_classifier_result_table_col_width:],
                      "Model Name": names[k]                  
                      }]
                      
            results = results.append(a)
        
        print(tabulate(results,headers="keys",showindex="never"))
    
        save_predictions_in_db(cron_id, ticket_id, pred)
    
        
    except Exception as e:
        print("Error when predicting with RecursiveTreeClassifier")
        print(e)
        traceback.print_exc()


    return




    







#########################################################################################################        
#####################             OLD/UNSUED TREE CLASSIFIER METHODS                #####################        
#########################################################################################################  

def tree_classifier_train(column_list, table, table_key, target_col, where_clause = "", save_prefix = "", parameters = None):

    """
    Run a hierarchical tree classification.
    :param parameters: contains a json with all the additional parameters (lemming,stemming,...)
    """

    # Check if Model folder exists
    check_for_folders()
    
    classification_layers = target_col.split(',') 
    print("Classifying layers: ")
    print(classification_layers)

    # Get tickets ids, title and description
    print("Fetching data from db...")
    data = get_db_training_data(column_list, table, table_key, classification_layers, where_clause)
    #data.dropna(inplace=True)

    # # Fetch parameters from service table json
    Tree_config = get_Tree_config_params(parameters)
    # fetch the tree classifier metric threshold (min score value to consider the classification usable)
    tree_classifier_min_score =  Tree_config["tree_classifier_min_score"]
    # fetch the tree classifier min cardinality threshold (min cardinality of a class to be classified)
    tree_classifier_min_cardinality =  Tree_config["tree_classifier_min_cardinality"]
    
    # Filters out labels with cardinality < ", "min_cardinality" parameter value if set in "parameters" column in services table
    # filters based on the last layer that is the actual classification target 
    filtered_data = filter_by_cardinality(data, classification_layers[0], tree_classifier_min_cardinality)

    print("Dataframe shape:")
    print(filtered_data.shape)

    # pull out a subset of the dataset to be used as global test for the TreeClassifier
    # stratify parameters makes the split proportional between labels (percentage of ticket per label will be the same in the splitted datasets)
    # dataframe_global_test is used to test the TreeClassifier global performance
    # split proportions based on the last layer that is the actual classification target 
    proportional = True
    stratify_param = get_service_parameter_col(parameters,"stratify_split")
    if stratify_param is not None and stratify_param is False:
        proportional = False

    stratify = None
    if proportional:
        stratify = filtered_data[classification_layers[0]]
    dataframe, dataframe_global_test = train_test_split(filtered_data, stratify=stratify,test_size=0.10)
    
    #store the model to use for every level/label
    tree_dictionary = dict()
    
    # save the model file with classification_layer(the target column for this train) and 
    # label(in this case there is no label as it's the tree root)
    current_label = ROOT_MODEL_LABEL
    model_id = set_modelid(classification_layers[0],current_label)
    root_model = train_classifiers(column_list, table, table_key, classification_layers[0], where_clause, save_prefix + "_" + model_id, parameters, dataframe)
    #store the result from training
    tree_dictionary[model_id] = root_model
    
    subset_concatenation = False
    unique_label_subset_concat_param = get_service_parameter_col(parameters,"unique_label_subset_concat")
    if unique_label_subset_concat_param is not None and unique_label_subset_concat_param is True:
        subset_concatenation = True
    
    j = 0
    for layer in classification_layers:
        layer_labels = np.unique(np.array(list(dataframe[layer])))
        if j+1 >= len(classification_layers): break # if it's the deepest layer stop
        
        bad_df = None # dataframe eventually containing the previous subset if the train_classifiers return None (min_score or just one unique label in the whole subset)
        bad_label = ""  # label related to the bad_df training
                
        for label in layer_labels:
            if bad_label != "":
                model_id = set_modelid(classification_layers[j+1],label + "_" + bad_label)
            else:
                model_id = set_modelid(classification_layers[j+1],label)
            
            #gets the subset for the next train
            df = dataframe.loc[dataframe[layer] == label] 
            temp_df = filter_by_cardinality(df, classification_layers[j+1], tree_classifier_min_cardinality)
                        
            if temp_df is None or temp_df is not None and len(temp_df.index) == 0:
                print("In layer: " + str(layer) + " excluding label: " + str(label) + 
                " for label with cardinality < tree_classifier_min_cardinality parameter (" + str(tree_classifier_min_cardinality) + ")")
                if bad_label != "": model_id = set_modelid(classification_layers[j+1],label)
                tree_dictionary[model_id] = None
                bad_df = None
                bad_label = ""
                continue
            
            #checkfor subsets with only one unique label
            if len(temp_df[classification_layers[j+1]].unique()) == 1:
                if subset_concatenation:
                    #concatenates the current subset with the next one
                    if bad_df is not None:
                        bad_df = pd.concat([bad_df,temp_df])
                        bad_label += "_" + label
                    else:
                        bad_df = temp_df
                        bad_label = label
                    continue
                else:
                    #no classification possible on a subset with one unique label
                    tree_dictionary[model_id] = None
                    bad_df = None
                    bad_label = ""
                    continue
            
            #if the bad_df dataframe contains the previous subset concatenate it to the current one
            if bad_df is not None:
                temp_df = pd.concat([temp_df,bad_df])
                bad_df = None
                bad_label = ""
                print("--- WARNING --- previous subset train returned None, CONCATENATING previous subset with current: " + model_id)
    
            prefix = save_prefix + "_" + model_id
            where_condition = layer + " = '" + label + "'"
            cols = column_list.copy()
            
            #logfile = os.path.join(os.path.join(os.getcwd(), "logs"), prefix + "_DUMP_TRAINING_SUBSET.txt")
            #with open(logfile, "w") as file:
            #    np.set_printoptions(threshold=sys.maxsize)
            #    pd.set_option('display.max_rows', None)
            #    pd.set_option('display.max_columns', None)
            #    pd.set_option('display.width', None)
            #    pd.set_option('display.max_colwidth', -1)
            #    pprint.pprint(classification_layers[j+1],file,width=sys.maxsize)
            #    pprint.pprint("------------------------------------------------------------------------------------" ,file,width=sys.maxsize)
            #    pprint.pprint(temp_df[classification_layers[j+1]].unique(),file,width=sys.maxsize)
            
            
            current_model = train_classifiers(cols, table, table_key, classification_layers[j+1], where_condition, prefix, parameters, temp_df, disable_min_cardinality_filter = True, tree_classifier_layers = classification_layers)
            
            if current_model["metric_value"] is None or (current_model["metric_value"] is not None and float(current_model["metric_value"]) < tree_classifier_min_score):
                print("In layer: " + str(layer) + " excluding label: " + str(label) + " for model classification score: "
                                   + str(current_model["metric_value"]) + " < tree_classifier_min_score parameter (" + str(tree_classifier_min_score) + ")")
                tree_dictionary[model_id] = None
                bad_df = None
                bad_label = ""
                # Delete the model file
                try:
                    save_path = os.path.join(os.getcwd(), "Models")
                    
                    if current_model["model_name"] == "CNNClassifier":
                        filename = os.path.join(save_path, prefix + CNN_MODEL_FILENAME_SUFFIX)
                        if os.path.exists(filename):
                            os.remove(filename)
                            print("Successfully deleted unused CNN model file: " + filename)
                        
                    dict_filename = os.path.join(save_path, current_model["model_filename"])
                    if os.path.exists(dict_filename):
                        os.remove(dict_filename)
                        print("Successfully deleted unused dict model file: " + dict_filename)
                except Exception as e:
                    print("Error when deleting the model file " + dict_filename + "for score lower than tree_classifier_min_score ")
                continue
                
            tree_dictionary[model_id] = current_model
        
        j += 1
    
    # evaluate the global performance of the TreeClassifier by predicting on the global test dataset dataframe_global_test
    print(" --- TREE CLASSIFIER --- EVALUATE START at: " + str(datetime.datetime.now()))
    prediction_results = pd.DataFrame(columns=classification_layers)
    model_names = pd.DataFrame(columns=classification_layers)
    
    jj = 1
    tot = len(dataframe_global_test.index)
    
    if DEBUG_MODE: pdb.set_trace()
    
    for datarow in dataframe_global_test.iterrows():
        if DEBUG_MODE: tracemalloc.start()    

        print(" --- TreeClassifier --- evaluating " + str(jj) + " of " + str(tot) + " ... " + str(datetime.datetime.now()))
        jj += 1
        pred, names = tree_classifier_evaluate(datarow[1], target_col, save_prefix, tree_dictionary, parameters)
        prediction_results = prediction_results.append([pred],ignore_index=True)
        model_names = model_names.append([names],ignore_index=True)
        print(" ".join(pd.Series.to_string(datarow[1][target_col.split(',')]).split()) + " |~~~~~| " + str(pred))
        print("using models: " + str(names))
        reset_keras()
        
        if DEBUG_MODE: current, peak = tracemalloc.get_traced_memory()
        if DEBUG_MODE: print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        if DEBUG_MODE: tracemalloc.stop()

    print(" --- TREE CLASSIFIER --- EVALUATE END at: " + str(datetime.datetime.now()))
    
    
    reports = dict()
    confmats = dict()
    metrics = dict()
    final_output = dict()
    for layer in classification_layers:
        #remove NaN values for the labels not predicted because no model was created (excluded by tree_classifier_min_cardinality or tree_classifier_min_score)
        cleaned_predictions = []
        cleaned_global_test = []
        ii = 0
        for g in dataframe_global_test[layer]:
            p = prediction_results[layer][ii]
            if pd.isnull(p): 
                ii += 1
                continue
            else:
                cleaned_predictions.append(str(p))
                cleaned_global_test.append(str(g))
                ii += 1
        
        if len(cleaned_predictions) == 0 or len(cleaned_global_test) == 0:
            print("--- WARNING --- no model created for " + str(layer) + " -- try change tree_classifier_min_cardinality or tree_classifier_min_score parameters")
            final_output[layer] = "0"
            continue
            
        reports[layer], confmats[layer], metrics[layer] = report_and_confmat(cleaned_global_test, cleaned_predictions, save_prefix + "_TreeClassifier - " + str(layer), save_data=True)
        final_output[layer] = metrics[layer]["accuracy"]
    
    
    #saving result tree to file
    save_path = os.path.join(os.getcwd(), "Models")
    filename = save_prefix + "_TreeClassifier_results.pkl"
    full_filename = os.path.join(save_path, filename)
    with open(full_filename, "wb") as pklfile:
        pickle.dump(tree_dictionary, pklfile)

    #nicely print the tree dictionary 
    print("                                                     ")
    print("      *   =================================   *      ")
    print("     ***  ===      TREE CLASSIFIER      ===  ***     ")
    print("    ***** ===          RESULTS          === *****    ")
    print("      |   =================================   |      ")
    print("                                                     ")
    #pprint.pprint(tree_dictionary)
    save_dir = os.path.join(os.getcwd(), "Results")
    file = os.path.join(save_dir, save_prefix + "_TreeClassifier_results.txt")
    with open(file, 'wt') as f:
        pprint.pprint(tree_dictionary, f) #saves to file
    
    return final_output 


def tree_classifier_evaluate(data, target_columns, model_prefix, tree_dict, parameters):
    
    """
    Evaluates the performance of the TreeClassifier over the given test dataset 
    Itarates over all the entries in the dataset and generates the confusion matrix for each layer of the classification hierarchy 
    :param data: the test dataset
    :param target_columns: the hierarchy of targets (tree layers for classification)
    :param model_prefix: the model filename prefix (cronID + serviceID)
    :param tree_dict: the TreeClassifier dictionary containing the models data
    :param parameters: contains a json with all the additional parameters (lemming,stemming,...)
    :return: a dictionary like {"target_label_liv1":"predicted_label", "target_label_liv2":"predicted_label",....}
    """
    
    #go through layers to hierarchically load models to classify the text
    j = 0
    label = ROOT_MODEL_LABEL #the first layer is the root of the tree
    prediction_results = dict()
    model_names = dict()
    
    try:
        tree_layers = target_columns.split(',') 
        
        for layer in tree_layers:
            model_id = set_modelid(layer,label)
            concatenated_key = ""
            try:
                #concatenated_key = next(k for (k,v) in tree_dict.items() if k.startswith(model_id))
                lb = "_" + slugify(label)
                concatenated_key = next(k for (k,v) in tree_dict.items() if lb in k)
            except StopIteration:
                concatenated_key = ""
            if model_id not in tree_dict.keys() and concatenated_key != "": # try to check if the subset has been concatenated with another subset
                model_id = concatenated_key
                print("--- FOUND CONCATENATED SUBSETS --- " + str(model_id))
                if DEBUG_MODE: pdb.set_trace()

                # if the model is None means it hasn't been created in training (score or cardinality too low)
            if (model_id not in tree_dict.keys() and concatenated_key == "") or (tree_dict[model_id] is None):
                break
            model_folder = os.path.join(os.getcwd(), "Models")
            model_name = tree_dict[model_id]["model_name"]
            model_filename = tree_dict[model_id]["model_filename"] 
            to_load = os.path.join(model_folder, model_filename)
            
            with open(to_load, "rb") as file:
                model_dict = pickle.load(file)
            model_date = str(model_dict["createdtime"])
            vectorizer = model_dict["vectorizer"]
            column_list = model_dict["columns"]
            for t in tree_layers:
                if t in column_list:
                    column_list.remove(t)
            
            model = model_dict["model"]
                        
            # Preprocess the text before classification (stopwords removal, lemming, stemming)
            try:
                # fetch the text language from db
                language = get_language_parameter(parameters)

                data["text"] = ""
                for col in column_list:
                    data["text"] = data["text"] + ' ' + data[col]
                data["text"] = clean_text(data["text"], language)
                
                # Lemmatize text 
                if get_lemming_parameter(parameters):
                    data["text"] = word_lemming(data["text"], language)
                    
                # Stemmatize text 
                if get_stemming_parameter(parameters):
                    data["text"] = word_stemming(data["text"], language)

            except ValueError as e:
                print("Error when processing the text.")
                print(e)
                traceback.print_exc()
                        
            # if predicting with CNN Classifier it needs to load the model and weights from h5 file
            CNNmodel = None
            prediction = None
            text = pd.Series([data["text"]])
            model_names[layer] = model_name

            if model_name == "CNNClassifier":
                CNNmodel = load_model(os.path.join(model_folder,model_dict["CNN_model_file"]))
                CNN_config = get_CNN_config_params(parameters)
                prediction = get_CNN_predictions_confidence(CNNmodel, text, CNN_config, model_dict["CNN_tokenizer"], model_dict["CNN_labelencoder"])
                prediction_results[layer] = str(prediction[0][0])
            else:
                data_bow = vectorizer.transform(text)
                prediction = model.predict(data_bow)
                prediction_results[layer] = prediction[0]
            
            label = prediction_results[layer]
        #END FOR LOOP ON LAYERS
        
    
    except Exception as e:
        print("ERROR in TreeClassifier evaluate") 
        print(e)
        traceback.print_exc()
        prediction_results["ERROR"] = str(e)
        model_names["ERROR"] = str(e)

    return prediction_results, model_names


def tree_classifier_predict(cron_id, ticket_id, table_name, key_name, model_prefix, train_target_columns, parameters = None):
    
    """
    Predicts the label of a given ticket using the TreeClassifier. 
    Given the ticket_id it gets the text from the database and then it saves the result in the database
    :param ticket_id: integer corresponding to the ticket to classify
    :param parameters: contains a json with all the additional parameters (lemming,stemming,...)
    :return: /
    """
    
    try:
        #load the tree dictionary from file
        model_folder = os.path.join(os.getcwd(), "Models")
        fname = model_prefix + "_TreeClassifier_results.pkl"
        to_load = os.path.join(model_folder, fname)
        with open(to_load, "rb") as file:
            tree_dict = pickle.load(file)
        print("TreeClassifier dictionary loaded from file: " + to_load)
        
    except Exception as e:
        print("ERROR when loading TreeClassifier dictionary from file: " + to_load)
        traceback.print_exc()
        
        
    #go through layers to hierarchically load models to classify the text
    j = 0
    label = ROOT_MODEL_LABEL #the first layer is the root of the tree
    prediction_results = dict()
    model_names = dict()
    
    try:
        tree_layers = train_target_columns.split(',') 

        for layer in tree_layers:
            #pprint.pprint(tree_dict)
            model_id = set_modelid(layer,label)
            concatenated_key = ""
            try:
                #concatenated_key = next(k for (k,v) in tree_dict.items() if k.startswith(model_id))
                lb = "_" + slugify(label)
                concatenated_key = next(k for (k,v) in tree_dict.items() if lb in k)
            except StopIteration:
                concatenated_key = ""
            if model_id not in tree_dict.keys() and concatenated_key != "": # try to check if the subset has been concatenated with another subset
                model_id = concatenated_key
                print("--- FOUND CONCATENATED SUBSETS --- " + str(model_id))

            # if the model is None means it hasn't been created in training (score or cardinality too low)
            if (model_id not in tree_dict.keys() and concatenated_key == "") or (tree_dict[model_id] is None):
                print("--- PREDICTION STOPPED --- at layer " + layer + " for label " + label + " because the model hasn't been created in training" +
                      "(training score or cardinality too low).")
                break
            model_name = tree_dict[model_id]["model_name"]
            model_filename = tree_dict[model_id]["model_filename"] 
            to_load = os.path.join(model_folder, model_filename)
            
            with open(to_load, "rb") as file:
                model_dict = pickle.load(file)
            model_date = str(model_dict["createdtime"])
            vectorizer = model_dict["vectorizer"]
            column_list = model_dict["columns"]
            for t in tree_layers:
                if t in column_list:
                    column_list.remove(t)
            
            
            model = model_dict["model"]
            
            print("Classifying with " + model_name + " created " + model_date + " for " + model_prefix + " in layer " + layer)                
            
            # Fecth texts from db
            data = get_object_description(ticket_id, table_name, key_name, column_list)
            
            # Preprocess the text before classification (stopwords removal, lemming, stemming)
            data = text_preprocess(data, "text", parameters)
                        
            # if predicting with CNN Classifier it needs to load the model and weights from h5 file
            CNNmodel = None
            prediction = None
            model_names[layer] = model_name

            if model_name == "CNNClassifier":
                #CNNmodel = load_model(os.path.join(model_folder,model_dict["CNN_model_file"]))
                CNNmodel = load_model(os.path.join(model_folder,model_prefix + CNN_MODEL_FILENAME_SUFFIX))
                CNN_config = get_CNN_config_params(parameters)
                prediction = get_CNN_predictions_confidence(CNNmodel, data["text"], CNN_config, model_dict["CNN_tokenizer"], model_dict["CNN_labelencoder"])
                prediction_results[layer] = str(prediction[0][0])
            else:
                data_bow = vectorizer.transform(data["text"])
                prediction = model.predict(data_bow)
                prediction_results[layer] = prediction[0]

            label = prediction_results[layer]
        #END FOR LOOP ON LAYERS
    
        print("TreeClassifier prediction: ")
        print("TEXT: " + data.loc[0, "text"])
        print("RESULTS:")
        pprint.pprint(prediction_results)
        print("USING MODELS:")
        pprint.pprint(model_names)
        
        save_predictions_in_db(cron_id, ticket_id, prediction_results)    
    
    
    except Exception as e:
        print("ERROR in TreeClassifier prediction") 
        print(e)
        traceback.print_exc()
        

    