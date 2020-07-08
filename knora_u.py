import numpy as np
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import DistanceMetric


class KNORA_U(BaseEstimator, ClassifierMixin):
    """
    This method selects all classifiers that correctly classified at least one sample belonging to the
    region of competence of the query sample. Each selected classifier has a number of votes equals
    to the number of samples in the region of competence that it predicts the correct label. The
    votes obtained by all base classifiers are aggregated to obtain the final ensemble decision.

    Parameters:
        pool_classifiers -- list of classifiers
        k {int} -- number of neighbors used to estimate the competence of the base classifiers
        random_state {int} -- random_state to pass to base_estimator
    """

    # Initializer
    def __init__(self, pool_classifiers=None, k=7, random_state=66):

        self.pool_classifiers = pool_classifiers
        self.k = k
        self.random_state = random_state

        np.random.seed(self.random_state)

    # Fitting the model to the data
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.X_, self.y_ = X, y

        return self

    # finding a region of competence of smaple (xquery) by selecting KNN of sample in the validation set
    def region_of_competence(self, xquery, vali_set):
        region = []

        for i in vali_set:
            score = np.array(
                DistanceMetric.get_metric("euclidean").pairwise([xquery, i[0]])
            ).max()
            region.append([i, score])

        region = sorted(region, key=lambda t: t[1])[: self.k]

        return region

    # selection of all classifires that are able to correctly recognise at least one sample it the region of competence
    def selection(self, clf, region):
        value = 0
        for i in region:
            pred = clf.predict(i[0][0].reshape(1, -1))
            if pred != i[1]:
                value += 1

        if value == len(region):
            return False
        else:
            return True

    def predict(self, samples):
        check_is_fitted(self)
        samples = check_array(samples)

        y_pred = []

        for query in samples:
            region = self.region_of_competence(query, zip(self.X_, self.y_))
            ensemble = []

            for clf in self.pool_classifiers:
                if self.selection(clf, region):
                    ensemble.append(clf)

            # majority voting
            forecast = 0
            for clf in ensemble:
                value = clf.predict(query.reshape(1, -1))
                forecast += value

            if forecast <= (len(ensemble) / 2):
                y_pred.append(0)
            else:
                y_pred.append(1)

        return y_pred
