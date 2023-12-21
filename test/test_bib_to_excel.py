import unittest
import os
import pandas as pd
import os,sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path
from data_processing.bib_to_excel import BibTeXConverter

class TestBibTeXConverter(unittest.TestCase):

    def setUp(self):
        self.document_name = '*IEEE Xplore Citation BibTeX Download*'
        self.files_from_path = "test/data"
        self.file_path = "test/data/test_output.xlsx"
        self.bibtex_converter = BibTeXConverter(document_name=self.document_name, files_from_path=self.files_from_path, file_path=self.file_path)

    def tearDown(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def test_convert_to_excel(self):
        self.bibtex_converter.convert_to_excel()
        self.assertTrue(os.path.exists(self.file_path))

        df = pd.read_excel(self.file_path)
        self.assertEqual(len(df), 23)  # Assuming there are no BibTeX files matching the pattern

        # You can add more assertions to validate the content of the Excel file

if __name__ == '__main__':
    unittest.main()