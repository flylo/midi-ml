import os
import logging
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from midi_ml.pipelines.midi_reads import LabeledCorpusSet
from midi_ml.models.decomposition import PrincipalComponents
from midi_ml.tools.util import download_from_gcs, copy_file_to_gcs
from midi_ml.models.linear_decision_rules import PenalizedLogisticRegression, LinearDiscriminantAnalysis, \
    NaiveBayesClassifier

#########################################
# Things to train:                      #
#   - Un-penalized Logistic Regression  #
#   - Penalized Logistic Regression     #
#   - LDA                               #
#   - Multinomial Naive Bayes           #
#   - Bernoulli Naive Bayes             #
#   - PCA Gaussian Naive Bayes          #
#   - PCA Logistic Regression           #
# All models will be trained on:        #
#   - 2 tuples of notes                 #
#   - 3 tuples of notes                 #
# All models will be evaluated with:    #
#   - Leave-k out evaluation            #
#########################################

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


def train_regression(X: np.array, y: np.array):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    plr = PenalizedLogisticRegression(X_train, y_train, l2_penalty=10, num_iter=3)
    plr.fit()
    preds = plr.predict_probabilities(y_test) > 0.5
    print(confusion_matrix(y_train, preds))


def train_lda(X: np.array, y: np.array):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    lda = LinearDiscriminantAnalysis(X_train, y_train)
    lda.fit()
    preds = lda.predict(y_test)
    print(confusion_matrix(y_train, preds))

def train_multinomial_nb(X: np.array, y: np.array):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    mnb = NaiveBayesClassifier(X_train, y_train, parametric_form="multinomial")
    mnb.fit()
    preds = mnb.predict(y_test)
    print(confusion_matrix(y_train, preds))


def main():
    download_from_gcs(bucket_name="midi-ml",
                      prefix="parsed_corpus/",
                      local_fs_loc=os.environ["DATA_IN_LOC"])
    corpus_matrix = joblib.load(os.environ["DATA_IN_LOC"] + "/sparse_matrix")
    labels = joblib.load(os.environ["DATA_IN_LOC"] + "/labels")
    print(labels)
    print(corpus_matrix.shape)
    # bach_labels = [k for k in range(len(labels)) if labels[k] == "bach-js"]
    # beethoven_labels = [k for k in range(len(labels)) if labels[k] == "beethoven"]
    # X = corpus_matrix[bach_labels + beethoven_labels]
    # y = np.array([1 for i in range(len(bach_labels))] + [0 for i in range(len(beethoven_labels))])
    # # y = y.reshape(y.shape[0], 1, )
    # logger.info("training model")
    # plr = PenalizedLogisticRegression(X, y, l2_penalty=10, num_iter=3)
    # plr.fit()
    # print(plr.beta_)
