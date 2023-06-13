import logging

from pathlib import Path

logger = logging.getLogger()


class Model:
    """
    Class for holding information about a classifier model, such as the model itself, any predictions it has made on
    training data, and the classification report that describes its accuracy.
    """
    def __init__(self, classifier):
        self.classifier = classifier
        self.predictions = None
        self.classification_report = None

    def write_report(self, output_path):
        output_filename = Path(output_path, f'{self.classifier.__class__.__name__}_report.txt')
        logger.info(f'Writing classification report to {output_filename}')
        with open(output_filename, mode='a') as f:
            f.write(self.classification_report)

    @property
    def name(self):
        return self.classifier.__class__.__name__
