import configparser
import logging
import openai
import pandas as pd
import sys

from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC

from model import Model

logger = logging.getLogger()


class ModelTrainer:
    """

    """
    def __init__(self, df, path_for_results=Path()):
        self.df = df.copy()
        # A directory in which to save results. Defaults to the current working directory.
        self.path_for_results = path_for_results
        # Lists of the training and test data for the model.
        self._x_train, self._x_test, self._y_train, self._y_test = self.split()
        self.models = []

    @property
    def path_for_results(self):
        return self._path_for_results

    @path_for_results.setter
    def path_for_results(self, value):
        if not isinstance(value, Path):
            logger.error('"%s" is not a Path. Please provide a Path for saving model results.' % value)
            self._path_for_results = None
        else:
            # If the Path doesn't currently exist, create it.
            if not value.exists():
                logger.info('Output directory "%s" doesn\'t exist. Creating it now.' % value)
                value.mkdir(parents=True)
            self._path_for_results = value

    def split(self, independent_var_column='embedding', dependent_var_column='specialty'):
        """
        Create train and test sets with an 80/20 split.
        Args:
            df:
            independent_var_column (str): Column name in `df` for the independent variable (x).
            dependent_var_column (str): Column name in `df` for the dependent variable (y).

        Returns:

        """
        # split data into train and test
        x_train, x_test, y_train, y_test = train_test_split(
            list(self.df[independent_var_column].values), self.df[dependent_var_column], test_size=0.2, random_state=42
        )
        return x_train, x_test, y_train, y_test

    def train_a_model(self, classifier):
        """

        ... train a classifier using the default parameters. This is handy if you want to do a quick survey of multiple
        classifiers before settling on one to tune the hyperparameters.

        Args:

            classifier:

        Returns:

        """
        # Instantiate a Model with a new classifier.
        model = Model(classifier())
        model.classifier.fit(self._x_train, self._y_train)
        model.predictions = model.classifier.predict(self._x_test)

        model.report = classification_report(self._y_test, model.predictions)
        model.write_report(self.path_for_results)
        return model

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
            logger.error('Attempted to set df to %s. df must be a pandas DataFrame.' % value)
