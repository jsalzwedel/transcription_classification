import pandas as pd
import unittest

from io import StringIO
from pathlib import Path
from unittest.mock import patch
from unittest import TestCase

from data_processor import DataProcessor


class TestGetEmbeddingsForTranscriptionNotes(TestCase):
    """
    Test the get_embeddings_for_transcription_notes function.
    """

    def setUp(self):
        """
        Prepare a tiny data set that we can use for testing our ability to get embeddings without spending a lot of time
        or breaking the bank.
        """

        self._simple_df = pd.DataFrame(
            {'specialty': ['Surgery', 'Homeopathy'],
             'transcription_notes': ['Patient was prepped for surgery. We cracked his ribs and repaired his broken'
                                     ' heart. He is much better now.',
                                     '1 mL of cyanide was mixed with 500 L of water. Patient drank 1 mL of this'
                                     ' concoction. Patient did not die. Hooray!']
             }
        )

    @patch('builtins.input', return_value='n')
    def test_exit_without_getting_embeddings_if_user_says_no(self, mock_input):
        """
        Getting embeddings takes time and money. Test that the code exits without getting embeddings if the user
        opts out.
        """
        processor = DataProcessor()
        processor.df_raw = self._simple_df
        with self.assertRaises(SystemExit) as cm:
            processor.get_embeddings_for_transcription_notes()
            # Check that sys.exit() was called with -1
            self.assertEqual(cm.exception.code, -1)

    @patch('builtins.input', return_value='y')
    def test_get_embeddings_when_user_says_yes(self, mock_input):
        """
        Test that an embeddings column is created if the user agrees to generate the embeddings.
        """
        with patch('sys.stdout', new=StringIO()):
            processor = DataProcessor()
            processor.df_raw = self._simple_df
            df = processor.get_embeddings_for_transcription_notes()
            self.assertIn('embedding', df.columns)


class TestOutputCleanData(TestCase):
    """
    Test the output_clean_data function.
    """

    def setUp(self):
        # Create a data processor that we can use for testing.
        self.processor = DataProcessor()
        self.processor._data_path = Path('data/mock_data.xlsx')
        self.processor.df_clean = pd.DataFrame({'foo': [1, 2, 3]})

    def test_output_clean_data(self):
        """
        Test that when processor.output_clean_data() is called, it writes a clean data file to the same folder as the
        input data file.
        """
        # Call the output_clean_data method
        self.processor.output_clean_data()

        # Verify that the clean data file has been saved in the correct location
        expected_output_path = Path('data/mock_data_clean.xlsx')
        self.assertTrue(expected_output_path.exists())
        self.assertTrue(expected_output_path.is_file())

    def tearDown(self):
        # Delete the saved clean data file after the test
        expected_output_path = Path('data/mock_data_clean.xlsx')
        if expected_output_path.exists():
            expected_output_path.unlink()


if __name__ == '__main__':
    unittest.main()
