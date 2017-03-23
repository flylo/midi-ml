from sklearn.externals import joblib
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


def main():
    pass


if __name__ == "__main__":
    main()
