import configparser
import logging
import openai
import pandas as pd
import sys

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC


logger = logging.getLogger()


class ModelTrainer:
    """

    """
    def __init__(self, df):
        self.df = df.copy()


    @staticmethod
    def split(df):
        """
        Create train and test sets with an 80/20 split.
        Args:
            df:

        Returns:

        """
        # split data into train and test
        x_train, x_test, y_train, y_test = train_test_split(
            list(df['embedding'].values), df['specialty'], test_size=0.2, random_state=42
        )
        return x_train, x_test, y_train, y_test

    @staticmethod
    def train_and_predict(x_train, x_test, y_train, y_test, classifier):
        """

        ... train a classifier using the default parameters. This is handy if you want to do a quick survey of multiple
        classifiers before settling on one to tune the hyperparameters.

        Args:
            x_train:
            x_test:
            y_train:
            y_test:
            classifier:

        Returns:

        """
        clf = classifier()
        clf.fit(x_train, y_train)
        preds = clf.predict(x_test)

        report = classification_report(y_test, preds)
        print(report)

        with open(f'{classifier.__class__.__name__}_report.txt', mode='a') as f:
            f.write(report)

    @staticmethod
    def tune_linearsvc_hyperparameters(x_train, x_test, y_train, y_test):
        """

        Args:
            x_train:
            x_test:
            y_train:
            y_test:

        Returns:

        """
        # Define the parameter grid to search over
        param_grid = {
            'C': [0.1, 1, 10],  # Regularization parameter
            'loss': ['hinge', 'squared_hinge'],  # Loss function
            'max_iter': [1000, 2000, 3000]  # Maximum number of iterations
        }

        # Create the LinearSVC classifier
        clf_svc = LinearSVC()

        # Perform grid search with cross-validation
        grid_search = GridSearchCV(clf_svc, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(x_train, y_train)

        # Get the best model
        best_model = grid_search.best_estimator_

        # Evaluate the best model on the test set
        preds = best_model.predict(x_test)
        report = classification_report(y_test, preds)
        print(report)
        with open('tuned_linearsvc_report.txt', mode='a') as f:
            f.write(report)

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, value):
        if isinstance(value, pd.DataFrame):
            self._df = value
        else:
            logger.error(f'Attempted to set df to {value}. df must be a pandas DataFrame.')
