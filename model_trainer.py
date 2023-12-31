import logging
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC

from model import Model

logger = logging.getLogger()


class ModelTrainer:
    """
    Class for handling model training. This class takes in a clean data frame that has an 'embedding' column and
    a 'specialty' column and splits the data into train and test data sets. It can then use those data to train
    various classifier models. It can determine which is the best one by seeing which model has the highest weighted
    f1 score, then write that model to a pickle file.
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
            df: A DataFrame containing columns `independent_var_column` and `dependent_var_column`.
            independent_var_column (str): Column name in `df` for the independent variable (x).
            dependent_var_column (str): Column name in `df` for the dependent variable (y).

        Returns:
            x_train, x_test, y_train, y_test
        """
        # split data into train and test
        x_train, x_test, y_train, y_test = train_test_split(
            list(self.df[independent_var_column].values), self.df[dependent_var_column], test_size=0.2, random_state=42
        )
        return x_train, x_test, y_train, y_test

    def train_a_model(self, classifier_class):
        """
        Create a Model object based around a `classifier_class` classifier. Train a classifier using the default
        parameters. Then save the classification report (precision, recall, accuracy, and f1) to file. This is handy if
        you want to do a quick survey of multiple classifiers before settling on one to tune the hyperparameters.

        Args:
            classifier_class: A sklearn classifier class, e.g., LinearSVC or RandomForestClassifier

        Returns:
            Nothing. After creating, training, and using the model for predictions, append it to the self.models list.
        """
        # Instantiate a Model with a new classifier.
        model = Model(classifier_class())
        logger.info('Training a %s model' % model.name)
        model.classifier.fit(self._x_train, self._y_train)
        model.predictions = model.classifier.predict(self._x_test)

        model.classification_report_text = classification_report(self._y_test, model.predictions)
        model.classification_report_dict = classification_report(self._y_test, model.predictions, output_dict=True)
        model.write_report(self.path_for_results)
        self.models.append(model)

    def train_a_model_with_hyperparameter_tuning(self, classifier_class):
        """
        Similar to train_a_model, but this time it does hyperparameter tuning.
        Note: right now this only accounts for the hyperparameters for LinearSVC models. Future improvements would
        include modifying this to allow for hyperparameters for various kinds of classifiers, perhaps by using storing
        the parameter grid in a config file.

        Args:
            classifier_class: A sklearn classifier class, e.g., LinearSVC or RandomForestClassifier

        Returns:
            Nothing. After creating, training, and using the model for predictions, append it to the self.models list.
        """
        classifier = classifier_class()
        logger.info('Training a %s model with hyperparameter tuning.' % classifier.__class__.__name__)

        # Define the parameter grid to search over
        param_grid = {
            'C': [0.1, 1, 10],  # Regularization parameter
            'loss': ['hinge', 'squared_hinge'],  # Loss function
            'max_iter': [3000]  # Maximum number of iterations
        }
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self._x_train, self._y_train)

        # Get the best model
        model = Model(grid_search.best_estimator_)

        # Evaluate the best model on the test set
        model.predictions = model.classifier.predict(self._x_test)
        model.classification_report_text = classification_report(self._y_test, model.predictions)
        model.classification_report_dict = classification_report(self._y_test, model.predictions, output_dict=True)
        model.write_report(self.path_for_results, '_hyperparameter')
        self.models.append(model)

    def make_several_example_models(self):
        """
        Make and save several example models.
        """
        self.train_a_model(LinearSVC)
        self.train_a_model(RandomForestClassifier)
        self.train_a_model_with_hyperparameter_tuning(LinearSVC)
        best_model, best_score = self.get_model_with_best_f1()
        if best_model:
            logger.info('The model with the highest weighted f1 score is %s with a score of %s' % (best_model.name,
                                                                                                   best_score))
            logger.info('Writing the best model to a pickle file.')
            best_model.write_classifier_to_pickle(self.path_for_results)
        else:
            logger.info('Wow. All of these models are awful. None of them have an f1 score > 0.0.')

    def get_model_with_best_f1(self):
        """
        Scan through all the models in the self.models list to find the model with the best weighted f1 score.

        Note: Future work would change this to let the user pick which statistic to favor.
        Returns:
            best_model (Model): The Model object that contains the best classifier.
            best_f1_score (float): The f1 score of the best model.
        """
        best_model = None
        best_f1_score = 0.0

        for model in self.models:
            weighted_avg_f1_score = model.classification_report_dict['weighted avg']['f1-score']
            if weighted_avg_f1_score > best_f1_score:
                best_model = model
                best_f1_score = weighted_avg_f1_score

        return best_model, best_f1_score

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, df):
        if not isinstance(df, pd.DataFrame):
            logger.error('Attempted to set df to %s. df must be a pandas DataFrame.' % df)
        if 'embedding' not in df.columns:
            logger.error('This DataFrame doesn\'t have an "embedding" column. Please provide ModelTrainer a df with'
                         ' a text embedding column called "embedding".')
        # Make sure the embedding column is full of numpy arrays (it might be full of strings if we read the df in from
        # a file).
        df['embedding'] = df['embedding'].apply(self.convert_to_array)
        self._df = df

    @staticmethod
    def convert_to_array(value):
        """
        Check if a value is a numpy array. If it isn't, turn it into one.
        Args:
            value (np.array or str): A value that might be a numpy array in disguise.
                E.g. the string "[0.3123, .213., 1232]"

        Returns:
            value (np.array): The value which is now definitely a numpy array.
        """

        if isinstance(value, np.ndarray):
            return value
        else:
            return np.array(eval(value))
