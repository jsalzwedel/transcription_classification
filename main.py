import argparse
import logging

from colorama import Fore
from data_processor import DataProcessor
from os import linesep
from model_trainer import ModelTrainer

logger = logging.getLogger()


if __name__ == '__main__':
    """
    
    """
    # I like to include usage examples with my argparser
    example_command = (Fore.CYAN +
                       f'**Usage**{linesep}'
                       'To get started with the data processor and/or model trainer, try one of the following example'
                       ' commands:'
                       + Fore.RESET +
                       f'{linesep} {linesep}'
                       'The following will command will run the data processor to clean the data and query the OpenAI'
                       ' API for to get word embeddings for all the transcription_notes.'
                       f'{linesep}'
                       + Fore.GREEN +
                       'python main.py --process "data/mtsamples_transcription_data.xlsx"'
                       + Fore.RESET
                       )

    # Use RawDescriptionHelpFormatter to disable the automatic word wrapping and whitespace manipulation that argparser
    # does to epilog text.
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, epilog=example_command)
    parser.add_argument('--filepath_to_raw_data',
                        type=str,
                        help='If you want to clean and process data, provide the relative or absolute path to the csv'
                             ' or xlsx that you want to process. E.g., data/mtsamples_transcription_data.xlsx. This'
                             ' uses the OpenAI API (about $1.5 for the mtsamples data), and takes a few minutes, so you'
                             ' may want to avoid running it more than necessary.',
                        default=None)

    parser.add_argument('--filepath_to_processed_data',
                        type=str,
                        help='If you want read in data that has already been cleaned and processed, provide the'
                             ' relative or absolute path to the csv or xlsx.'
                             ' E.g., data/mtsamples_transcription_data_clean.xlsx.',
                        default=None)

    parser.add_argument('--train_model',
                        type=bool,
                        help='Set to True if you want to train a model using the processed data. This must be used in'
                             ' conjunction with either --filepath_to_raw_data or --filepath_to_processed_data',
                        default=False)

    options = parser.parse_args()
    if options.filepath_to_raw_data:
        data_processor = DataProcessor.process_data(options.filepath_to_raw_data)

    if options.filepath_to_processed_data:
        # No need to clean and process raw data.
        data_processor = DataProcessor()
        data_processor.data_path = options.filepath_to_processed_data
        if data_processor.data_path:
            data_processor.df_clean = data_processor.read_data(data_processor.data_path)

    if options.train_model:
        # Get a clean data frame from the data processor.
        model_trainer = ModelTrainer(data_processor.df_clean)
        # ...

