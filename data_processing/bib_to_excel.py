import glob
import os
from pybtex.database import parse_file
import pandas as pd

class BibTeXConverter:
    """
    A class that converts BibTeX files to Excel format.

    Args:
        document_name (str): The name pattern of the BibTeX files to be converted.
        files_from_path (str): The directory path where the BibTeX files are located.
        file_path (str): The file path where the converted Excel file will be saved.

    Attributes:
        document_name (str): The name pattern of the BibTeX files to be converted.
        files_from_path (str): The directory path where the BibTeX files are located.
        file_path (str): The file path where the converted Excel file will be saved.
        data (dict): A dictionary to store the extracted data from the BibTeX files.

    Methods:
        _parse_bib_file(bib_file): Parses a single BibTeX file and extracts the required data.
        convert_to_excel(): Converts the BibTeX files to Excel format and saves the file.

    """

    def __init__(self, document_name, files_from_path, file_path):
        self.document_name = document_name
        self.files_from_path = files_from_path
        self.file_path = file_path
        self.data = {'Booktitle': [], 'Title': [], 'Year': [], 'Pages': [], 'Abstract': [], 'Keywords': [], 'DOI': [], 'Authors': []}

    def _parse_bib_file(self, bib_file):
        """
        Parses a single BibTeX file and extracts the required data.

        Args:
            bib_file (str): The path of the BibTeX file to be parsed.

        """
        bib_data = parse_file(bib_file)
        for entry_key, entry in bib_data.entries.items():
            self.data['Booktitle'].append(entry.fields.get('booktitle', ''))
            self.data['Title'].append(entry.fields.get('title', ''))
            self.data['Year'].append(entry.fields.get('year', ''))
            self.data['Pages'].append(entry.fields.get('pages', ''))
            self.data['Abstract'].append(entry.fields.get('abstract', ''))
            self.data['Keywords'].append(entry.fields.get('keywords', ''))
            self.data['DOI'].append(entry.fields.get('doi', ''))
            authors = entry.persons.get('author', [])
            self.data['Authors'].append(', '.join(str(author) for author in authors))

    def convert_to_excel(self):
        """
        Converts the BibTeX files to Excel format and saves the file.

        """
        patron = f'{self.document_name}*'
        archivos_coincidentes = glob.glob(os.path.join(self.files_from_path, patron))

        for nombre_coincidente in archivos_coincidentes:
            self._parse_bib_file(nombre_coincidente)

        df = pd.DataFrame(self.data)
        df.to_excel(self.file_path, index=False)
        print(f"Excel file saved successfully at {self.file_path}")
