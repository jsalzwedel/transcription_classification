import logging
import pickle

from pathlib import Path

logger = logging.getLogger()


class Model:
    """
    Class for holding information about a classifier model, such as the model itself, any predictions it has made on
    training data, and the classification report that describes its accuracy.
    """
    def __init__(self, classifier=None):
        self.classifier = classifier
        self.predictions = None
        self.classification_report_text = None
        self.classification_report_dict = None

    def write_report(self, output_path, filename_modifier=''):
        output_filename = Path(output_path, f'{self.name}{filename_modifier}_report.txt')
        logger.info(f'Writing classification report to {output_filename}.')
        with open(output_filename, mode='a') as f:
            f.write(self.classification_report_text)

    def write_classifier_to_pickle(self, output_path):
        output_filename = Path(output_path, f'{self.name}_model.pickle')
        logger.info(f'Writing classifier model to pickle file at {output_filename}.')
        with open(output_filename, mode='wb') as f:
            pickle.dump(self.classifier, f)

    def read_classifier_from_pickle(self, input_path):
        input_filename = Path(input_path)
        logger.info(f'Reading classifier model from pickle file: {input_filename}.')
        with open(input_filename, 'rb') as f:
            classifier = pickle.load(f)
        self.classifier = classifier

    @property
    def name(self):
        return self.classifier.__class__.__name__

