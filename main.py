import argparse
import itertools
import logging

from colorama import Fore
from data_processor import DataProcessor
from os import linesep
from model_trainer import ModelTrainer

logger = logging.getLogger()


def process_data():
    logger.info('Running analysis.')

    # TODO argparser lets you skip preparing raw data and read in the processed data instead.

    raw_data_filepath = 'data/mtsamples_transcription_data.xlsx.xlsx'
    # Read in and clean the data
    data_processor = DataProcessor(raw_data_filepath)
    data_processor.get_embeddings_for_transcription_notes()
    data_processor.output_clean_data()


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
                       'python main.py --process "data/mtsamples_transcription_data.xlsx.xlsx"'
                       + Fore.RESET
                       )

    # Use RawDescriptionHelpFormatter to disable the automatic word wrapping and whitespace manipulation that argparser
    # does to epilog text.
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, epilog=example_command)

    # do_stuff()