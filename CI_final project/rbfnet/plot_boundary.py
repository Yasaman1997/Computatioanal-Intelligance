# x = np.random.uniform(2.5, 3.75, 100000)
# y = np.random.uniform(1, 5.5, 100000)
# plt.scatter(x, y)
# plt.show()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.plotting import plot_decision_regions
from sklearn.svm import SVC


def svd():
    data = pd.read_csv('2clstrain1200.csv', header=None)

    # this will create a variable x which has the feature values
    x = data.iloc[:, 0:2].values
    x = x.astype(np.integer)
    # y = data.iloc[:, 1].values
    label = data.iloc[:, 2].values
    label = label.astype(np.integer)
    # Training a classifier
    svm = SVC(C=0.5, kernel='rbf')
    svm.fit(x, label)

    # Plotting decision regions
    plot_decision_regions(x, label, clf=svm, legend=2)

    # Adding axes annotations
    plt.xlabel('data')
    plt.ylabel('label')
    plt.title('boundary')
    plt.show()


def Decision_Region_Grids():
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    import numpy as np

    # Initializing Classifiers
    clf1 = LogisticRegression(random_state=1,
                              solver='newton-cg',
                              multi_class='multinomial')
    clf2 = RandomForestClassifier(random_state=1, n_estimators=100)
    clf3 = GaussianNB()
    clf4 = SVC(gamma='auto')

    # Loading some example data
    data = pd.read_csv('2clstrain1200.csv', header=None)

    # this will create a variable x which has the feature values
    x = data.iloc[:, 0:2].values
    x = x.astype(np.integer)
    # y = data.iloc[:, 1].values
    label = data.iloc[:, 2].values
    label = label.astype(np.integer)

    import matplotlib.pyplot as plt
    from mlxtend.plotting import plot_decision_regions
    import matplotlib.gridspec as gridspec
    import itertools
    gs = gridspec.GridSpec(2, 2)

    fig = plt.figure(figsize=(10, 8))

    labels = ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'SVM']
    for clf, lab, grd in zip([clf1, clf2, clf3, clf4],
                             labels,
                             itertools.product([0, 1], repeat=2)):
        clf.fit(x, label)
        ax = plt.subplot(gs[grd[0], grd[1]])
        fig = plot_decision_regions(X=x, y=label, clf=clf, legend=2)
        plt.title(lab)

    plt.show()


def Highlighting_Test_Data_Points():
    from mlxtend.plotting import plot_decision_regions
    from mlxtend.preprocessing import shuffle_arrays_unison
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn.svm import SVC

    # Loading some example data
    iris = datasets.load_iris()
    data = pd.read_csv('2clstrain1200.csv', header=None)

    X, y = data.iloc[:, 0:2].values, data.iloc[:, 2].values
    X = X.astype(np.integer)
    y = y.astype(np.integer)
    X, y = shuffle_arrays_unison(arrays=[X, y], random_seed=3)

    X_train, y_train = X[:700], y[:700]
    X_test, y_test = X[700:], y[700:]

    # Training a classifier
    svm = SVC(C=0.5, kernel='linear')
    svm.fit(X_train, y_train)

    # Plotting decision regions
    plot_decision_regions(X, y, clf=svm, legend=2,
                          X_highlight=X_test)

    # Adding axes annotations
    plt.xlabel('')
    plt.ylabel('')
    plt.title('SVM on Iris')
    plt.show()


# XOR
def Evaluating_Classifier_Behavior_on_Non_Linear_Problems():
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    # Initializing Classifiers
    clf1 = LogisticRegression(random_state=1, solver='lbfgs')
    clf2 = RandomForestClassifier(n_estimators=100,
                                  random_state=1)
    clf3 = GaussianNB()
    clf4 = SVC(gamma='auto')

    # Loading Plotting Utilities
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import itertools
    from mlxtend.plotting import plot_decision_regions
    import numpy as np
    xx, yy = np.meshgrid(np.linspace(-3, 3, 50),
                         np.linspace(-3, 3, 50))
    rng = np.random.RandomState(0)
    X = rng.randn(300, 2)
    y = np.array(np.logical_xor(X[:, 0] > 0, X[:, 1] > 0),
                 dtype=int)
    gs = gridspec.GridSpec(2, 2)

    fig = plt.figure(figsize=(10, 8))

    labels = ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'SVM']
    for clf, lab, grd in zip([clf1, clf2, clf3, clf4],
                             labels,
                             itertools.product([0, 1], repeat=2)):
        clf.fit(X, y)
        ax = plt.subplot(gs[grd[0], grd[1]])
        fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
        plt.title(lab)

    plt.show()


def Decision_regions_with_more_than_two_training_features():
    from mlxtend.plotting import plot_decision_regions
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn.svm import SVC

    # Loading some example data
    X, y = datasets.make_blobs(n_samples=600, n_features=3,
                               centers=[[2, 2, -2], [-2, -2, 2]],
                               cluster_std=[2, 2], random_state=2)

    # Training a classifier
    svm = SVC(gamma='auto')
    svm.fit(X, y)

    # Plotting decision regions
    fig, ax = plt.subplots()
    # Decision region for feature 3 = 1.5
    value = 1.5
    # Plot training sample with feature 3 = 1.5 +/- 0.75
    width = 0.75
    plot_decision_regions(X, y, clf=svm,
                          filler_feature_values={2: value},
                          filler_feature_ranges={2: width},
                          legend=2, ax=ax)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Feature 3 = {}'.format(value))

    # Adding axes annotations
    fig.suptitle('SVM on make_blobs')
    plt.show()


def Grid_of_decision_region_slices():
    from mlxtend.plotting import plot_decision_regions
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn.svm import SVC

    # Loading some example data
    X, y = datasets.make_blobs(n_samples=500, n_features=3, centers=[[2, 2, -2], [-2, -2, 2]],
                               cluster_std=[2, 2], random_state=2)

    # Training a classifier
    svm = SVC(gamma='auto')
    svm.fit(X, y)

    # Plotting decision regions
    fig, axarr = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    values = [-4.0, -1.0, 1.0, 4.0]
    width = 0.75
    for value, ax in zip(values, axarr.flat):
        plot_decision_regions(X, y, clf=svm,
                              filler_feature_values={2: value},
                              filler_feature_ranges={2: width},
                              legend=2, ax=ax)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('Feature 3 = {}'.format(value))

    # Adding axes annotations
    fig.suptitle('SVM on make_blobs')
    plt.show()


def Customizing_the_plotting_style():
    from mlxtend.plotting import plot_decision_regions
    from mlxtend.preprocessing import shuffle_arrays_unison
    import matplotlib.pyplot as plt
    from sklearn.svm import SVC

    # Loading some example data
    data = pd.read_csv('2clstrain1200.csv', header=None)

    X, y = data.iloc[:, 0:2].values, data.iloc[:, 2].values
    X = X.astype(np.integer)
    y = y.astype(np.integer)
    X, y = shuffle_arrays_unison(arrays=[X, y], random_seed=3)

    X_train, y_train = X[:700], y[:700]
    X_test, y_test = X[700:], y[700:]

    # Training a classifier
    svm = SVC(C=0.5, kernel='linear')
    svm.fit(X_train, y_train)

    # Specify keyword arguments to be passed to underlying plotting functions
    scatter_kwargs = {'s': 120, 'edgecolor': None, 'alpha': 0.7}
    contourf_kwargs = {'alpha': 0.2}
    scatter_highlight_kwargs = {'s': 120, 'label': 'Test data', 'alpha': 0.7}
    # Plotting decision regions
    plot_decision_regions(X, y, clf=svm, legend=2,
                          X_highlight=X_test,
                          scatter_kwargs=scatter_kwargs,
                          contourf_kwargs=contourf_kwargs,
                          scatter_highlight_kwargs=scatter_highlight_kwargs)

    # Adding axes annotations
    plt.xlabel('')
    plt.ylabel('')
    plt.title('SVM on Iris')
    plt.show()


Customizing_the_plotting_style()
