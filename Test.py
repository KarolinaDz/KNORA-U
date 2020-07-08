from sklearn.ensemble import BaggingClassifier
from deslib.des import DESKNN
from deslib.des.knora_u import KNORAU
from deslib.des.knora_e import KNORAE
from knora_u import *
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from scipy.stats import rankdata
from scipy.stats import ranksums
from tabulate import tabulate
from sklearn.tree import DecisionTreeClassifier

rs = 66
# pool = []
# for i in range(10):
    # pool.append(DecisionTreeClassifier(random_state=rs))

clfs = {
    "Bagging": BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=rs), random_state=rs),
    "DES-KNN": DESKNN(random_state=rs),
    "KNORA-E": KNORAE(random_state=rs),
    "KNORA-U": KNORAU(random_state=rs),
    "My KNORA-U": KNORA_U(random_state=rs),
}


datasets = [
    "ecoli-0-1_vs_5",
    "ecoli4",
    "german",
    "glass2",
    "glass4",
    "monkone",
    "shuttle-c0-vs-c4",
    "vowel0",
    "yeast-2_vs_4",
    "yeast4",
]

# experiment
n_datasets = len(datasets)
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=1234
)
scores = np.zeros((len(clfs), n_datasets, n_splits * n_repeats))

for data_id, dataset in enumerate(datasets):
    dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])
            if clf_name == "Bagging":
                bag = clf
            else:
                clf.pool_classifiers = bag.estimators_

            clf.fit(X[train], y[train])

            y_pred = clf.predict(X[test])
            scores[clf_id, data_id, fold_id] = accuracy_score(y[test], y_pred)

np.save("results", scores)

# Results analysis
scores = np.load("results.npy")
print("\nScores:\n", scores.shape)

mean_scores = np.mean(scores, axis=2).T
print("\nMean scores:\n", mean_scores)

# Ranks
ranks = []
for ms in mean_scores:
    ranks.append(rankdata(ms).tolist())
ranks = np.array(ranks)
print("\nRanks:\n", ranks)

# Mean ranks
mean_ranks = np.mean(ranks, axis=0)
print("\nMean ranks:\n", mean_ranks)


# w-statistic and p-value
alfa = 0.05
w_statistic = np.zeros((len(clfs), len(clfs)))
p_value = np.zeros((len(clfs), len(clfs)))

for i in range(len(clfs)):
    for j in range(len(clfs)):
        w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])
# print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)


headers = list(clfs.keys())
names_column = np.expand_dims(np.array(list(clfs.keys())), axis=1)
w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("w-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

# Advantage
advantage = np.zeros((len(clfs), len(clfs)))
advantage[w_statistic > 0] = 1
advantage_table = tabulate(np.concatenate((names_column, advantage), axis=1), headers)
print("Advantage:\n", advantage_table)

# Statistical significance
significance = np.zeros((len(clfs), len(clfs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(
    np.concatenate((names_column, significance), axis=1), headers
)
print("\nStatistical significance (alpha = 0.05):\n", significance_table)

# Statistical significance better
stat_better = significance * advantage
stat_better_table = tabulate(
    np.concatenate((names_column, stat_better), axis=1), headers
)
print("Statistically significantly better:\n", stat_better_table)
