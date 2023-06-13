import argparse
import logging
import sys

from colorama import Fore
from data_processor import DataProcessor
from os import linesep
from pathlib import Path
from model_trainer import ModelTrainer

logger = logging.getLogger()
# Set the logging level to INFO to that the logger pays attention to INFO-level messages and above.
logger.setLevel(logging.INFO)

# Create a console handler and set its level to INFO. This will make INFO-level messages and above be displayed
# in the terminal.
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter and add it to the console handler. This formatter shows the file and function names where the
# logging message is called.
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(console_handler)


if __name__ == '__main__':
    """
    
    """
    # I like to include usage examples with my argparser
    example_command = (Fore.CYAN +
                       f'**Usage**{linesep}'
                       'To get started with the data processor and/or model trainer, try one of the following example'
                       ' commands:'
                       + Fore.RESET +
                       f'{linesep}'
                       'The following command will run the data processor to clean the data and query the OpenAI'
                       ' API to get word embeddings for all the transcription_notes.'
                       f'{linesep}'
                       + Fore.GREEN +
                       'python main.py --filepath_to_raw_data "data/mtsamples_transcription_data.xlsx"'
                       + Fore.RESET + f'{linesep}{linesep}'
                       'The following command will run the model trainer over existing clean data'
                       ' API to get word embeddings for all the transcription_notes.'
                       f'{linesep}'
                       + Fore.GREEN +
                       'python main.py --filepath_to_processed_data "data/mtsamples_transcription_data_clean.csv"'
                       ' --train_model'
                       + Fore.RESET + f'{linesep}{linesep}'
                       )

    # Use RawDescriptionHelpFormatter to disable the automatic word wrapping and whitespace manipulation that argparser
    # does to epilog text.
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, epilog=example_command)
    parser.add_argument('--filepath_to_raw_data',
                        type=str,
                        help='If you want to clean and process data, provide the relative or absolute path to the csv'
                             ' or xlsx that you want to process. E.g., data/mtsamples_transcription_data.xlsx. This'
                             ' uses the OpenAI API (about $1.5 for the mtsamples data) and takes about 20 minutes, so'
                             ' you may want to avoid running it more than necessary.',
                        default=None)

    parser.add_argument('--filepath_to_processed_data',
                        type=str,
                        help='If you want read in data that has already been cleaned and processed, provide the'
                             ' relative or absolute path to the csv or xlsx.'
                             ' E.g., data/mtsamples_transcription_data_clean.csv.',
                        default=None)

    parser.add_argument('--train_model',
                        action='store_true',
                        help='Set to True if you want to train a model using the processed data. This must be used in'
                             ' conjunction with either --filepath_to_raw_data or --filepath_to_processed_data.')

    options = parser.parse_args()
    data_processor = None
    if options.filepath_to_raw_data:
        data_processor = DataProcessor.process_data(options.filepath_to_raw_data)

    if options.filepath_to_processed_data:
        if not DataProcessor.validate_data_path(options.filepath_to_processed_data):
            sys.exit(-1)
        # No need to clean and process raw data.
        data_processor = DataProcessor()
        data_processor.data_path = options.filepath_to_processed_data
        if data_processor.data_path:
            data_processor.df_clean = data_processor.read_data(data_processor.data_path)

    if options.train_model:
        if not data_processor:
            logger.error('A data processor is required for model training. Please rerun with either the'
                         ' --filepath_to_raw_data or --filepath_to_processed_data options.')
            sys.exit(-1)
        # Get a clean data frame from the data processor.
        model_trainer = ModelTrainer(data_processor.df_clean, Path('results'))
        model_trainer.make_several_example_models()

