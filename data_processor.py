import configparser
import logging
import openai
import pandas as pd
import sys
import tiktoken

from openai.embeddings_utils import get_embedding
from os import linesep
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger()


class DataProcessor:
    """
    Class for reading, cleaning, and processing data.
    """
    def __init__(self, data_path=None):
        """

        Args:
            data_path
        """

        # Path to where the input data is stored. # User doesn't have to supply a data path during DataProcessor
        # creation. They may choose to do so at a later time.
        self._data_path = None
        # If they do supply a path, make sure it is valid.
        if data_path:
            self.data_path = data_path
        self._df_raw = None
        self._df_clean = None
        if self.data_path:
            # Validate the data path and read in the data. The df_raw setter also takes care of calling the clean_data
            # method to update self._df_clean.
            self.df_raw = self.read_data(data_path)


    @staticmethod
    def read_data(data_path):
        """
        Takes a path to a data file (csv or xlsx), reads in that data as a data frame, then returns the
        data.
        Args:
            data_path (str):

        Returns:
            df
        """
        logger.info('Reading data.')

        if data_path.suffix == '.csv':
            df = pd.read_csv(data_path)
        else:
            df = pd.read_excel(data_path)
        return df

    def clean_data(self, df):
        """

        Args:
            df:

        Returns:
            df_clean
        """
        logger.info('Cleaning data.')
        # Clear out the rows that have no transcription_notes in them.
        df_clean = df.dropna(subset=['transcription_notes']).copy()

        # There is a leading whitespace in every row of the specialty column that offends my senses. Get rid of it.
        df_clean['specialty'] = df_clean['specialty'].apply(lambda x: x.strip())

        # We have a lot of categories, some of which are sparsely populated. Let's try to do some class reduction. We'll
        # keep the 10 most populated categories and re-bucket all the other reports under "Other".
        specialty_counts = df_clean['specialty'].value_counts()
        # Rename all the categories to 'Other' except for the categories with the 10 largest row counts.
        classes_to_rename = specialty_counts.nsmallest(len(specialty_counts) - 10).index
        # Rename the classes to 'Other'
        df_clean.loc[df_clean['specialty'].isin(classes_to_rename), 'specialty'] = 'Other'

        return df_clean

    def get_embeddings_for_transcription_notes(self):
        """

        Returns:

        """
        logger.info('Creating text embeddings for the transcription notes.')
        df = self.df_clean
        config = configparser.ConfigParser()
        config.read('config.ini')
        openai.api_key = config.get('Credentials', 'api_key')

        # embedding model parameters
        embedding_model = "text-embedding-ada-002"
        embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
        max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

        encoding = tiktoken.get_encoding(embedding_encoding)

        # omit reviews that are too long to embed
        df["n_tokens"] = df['transcription_notes'].apply(lambda x: len(encoding.encode(x)))
        df = df[df.n_tokens <= max_tokens]

        # Estimate how much it's going to cost to get embeddings for this dataset. Verify that the user is okay
        # with that cost.
        total_tokens = df["n_tokens"].sum()
        response = input(f'Using the Ada 2 model, embeddings cost $0.0004 per 1k tokens. For this dataset with'
                         f' {total_tokens} tokens, that overall cost will be ${total_tokens * 0.0004 / 1000}. Is that'
                         f' okay? y/n{linesep}')
        if response.lower() not in ['y', 'yes']:
            logger.info('Embeddings will not be generated as the price is too high. Please rerun this program after you'
                        ' receive your weekly allowance.')
            sys.exit(-1)

        # Now get the text embeddings for each transcription_notes. Normally I'd just use .apply to loop over the
        # rows of a data frame, but if we manually loop over all the rows we can use tqdm to see a progress bar.
        df['embedding'] = None
        for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc='Processing rows'):
            df.at[idx, 'embedding'] = get_embedding(row['transcription_notes'], engine=embedding_model)

        return df

    @property
    def data_path(self):
        return self._data_path

    @data_path.setter
    def data_path(self, value):
        """
        Set data_path after validating that it is a csv/xlsx file.
        """
        if self.validate_data_path(value):
            self._data_path = value
        else:
            logger.error('Setting data_path to None.')

    @staticmethod
    def validate_data_path(data_path):
        """
        Validate that there is a csv/xlsx at the supplied path.
        Args:
            data_path (str): Absolute or relative path to a csv/xlsx.

        Returns:
            Bool: Whether the path points to a valid file.
        """
        if not data_path or not Path(data_path).exists():
            logger.error('Data file could not be found at %s.' % data_path)
            return False

        data_path = Path(data_path)
        if not any([data_path.suffix == extension for extension in ['.xlsx', '.csv']]):
            logger.error('File "%s" is not valid. Please provide a csv or xlsx.' % data_path)
            return False
        return True

    @property
    def df_raw(self):
        return self._df_raw

    @df_raw.setter
    def df_raw(self, value):
        """
        Set df_raw to a new value and update df_clean accordingly.

        If `value` is not actually a data frame, send a logger error. In that case, df_raw will remain as whatever it
        was previously set as
        """
        if isinstance(value, pd.DataFrame):
            self._df_raw = value
            self.df_clean = self.clean_data(self._df_raw)
        else:
            logger.error(f'Attempted to set df_raw to {value}. df_raw must be a pandas DataFrame.')

    @property
    def df_clean(self):
        return self._df_clean

    @df_clean.setter
    def df_clean(self, value):
        if isinstance(value, pd.DataFrame):
            self._df_clean = value
        else:
            logger.error(f'Attempted to set df_clean to {value}. df_clean must be a pandas DataFrame.')

    def output_clean_data(self):
        # Save the clean data alongside the raw data file in case someone wants to look at it.
        directory = self._data_path.parent
        filename_stem = self._data_path.stem
        output_path = Path(directory, f'{filename_stem}_clean.xlsx')
        logger.info('Saving processed data to %s' % str(output_path))
        self.df_clean.to_excel(output_path, index=False)

    @staticmethod
    def process_data(raw_data_filepath):
        """
        Read in the data from raw_data_filepath, clean it, get word embeddings for the transcription_notes
        column, write out the results, and return the data_processor in case someone wants to use any of
        the information in it.

        Args:
            raw_data_filepath (str):

        Returns:
            data_processor (DataProcessor): A data processor that contains the raw and clean data.
        """
        # Read in and clean the data
        data_processor = DataProcessor(raw_data_filepath)
        data_processor.get_embeddings_for_transcription_notes()
        data_processor.output_clean_data()
        return data_processor
