import numpy as np
from sklearn.externals import joblib
from midi_ml.pipelines.midi_reads import LabeledCorpusSet
from midi_ml.models.decomposition import PrincipalComponents
from midi_ml.models.linear_decision_rules import PenalizedLogisticRegression, LinearDiscriminantAnalysis, \
    NaiveBayesClassifier


#########################################
# Things to train:                      #
#   - Un-penalized Logistic Regression   #
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


def main(labeled_corpus: LabeledCorpusSet):
    corpus_matrix = labeled_corpus.sparse_matrix.todense()
    labels = labeled_corpus.corpus_labels
    bach_labels = [k for k in range(len(labels)) if labels[k] == "bach-js"]
    beethoven_labels = [k for k in range(len(labels)) if labels[k] == "beethoven"]
    X = corpus_matrix[bach_labels + beethoven_labels]
    y = np.array([1 for i in range(len(bach_labels))] + [0 for i in range(len(beethoven_labels))])
    # y = y.reshape(y.shape[0], 1, )
    plr = PenalizedLogisticRegression(X, y, l2_penalty=10)
    plr.fit()
