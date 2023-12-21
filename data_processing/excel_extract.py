import pandas as pd
import glob
from copy import deepcopy

class ExcelFilesProcessor:
    def __init__(self):
        """
        Initializes the ExcelFilesProcessor object.

        The ExcelFilesProcessor object is used for extracting and processing data from Excel files.
        It initializes the `multiple_dataframes` list to store multiple dataframes and sets the `dataframe` attribute to None.
        """
        self.multiple_dataframes = []
        self.dataframe = None

    def load_files(self, folder, prefix):
        """
        Load multiple Excel files from a specified folder with a given prefix.

        Args:
            folder (str): The folder path where the files are located.
            prefix (str): The prefix of the files to be loaded.

        Returns:
            None
        """
        files = glob.glob(f'{folder}/{prefix}*')
        dataframes = [pd.read_excel(file) for file in files]
        self.dataframe = pd.concat(dataframes, ignore_index=True)

    def eliminate_duplicates_normalize(self, output_file, article_title):
        """
        Eliminate duplicates from the loaded dataframe based on the specified article title column,
        normalize the data, and save the result to an Excel file.

        Args:
            output_file (str): The path to save the processed data.
            article_title (str): The column name to identify duplicate articles.

        Returns:
            None
        """
        if self.dataframe is None:
            raise ValueError("No files have been loaded. Use load_files first.")

        df_no_duplicates = self.dataframe.drop_duplicates(subset=article_title, keep='first')
        df_no_duplicates.rename(columns={article_title: 'title'}, inplace=True)

        df_no_duplicates.to_excel(output_file, index=False)
        print(f'Duplicates have been eliminated and saved to {output_file}')

    def multiple_dataframes_append(self, dataframe):
        """
        Append a dataframe to the list of multiple dataframes.

        Args:
            dataframe (pd.DataFrame): The dataframe to append.

        Returns:
            None
        """
        self.multiple_dataframes.append(dataframe)

    def multiple_dataframes_clear(self):
        """
        Clear the list of multiple dataframes.

        Returns:
            None
        """
        self.multiple_dataframes.clear()

    def multiple_dataframes_concat(self):
        """
        Concatenate the list of multiple dataframes into a single dataframe.

        Returns:
            None
        """
        self.dataframe = pd.concat(self.multiple_dataframes, ignore_index=True)

    def joint_articles(self, output_file):
        """
        Process the loaded dataframe, remove duplicates, and save the result to an Excel file.

        Args:
            output_file (str): The path to save the processed data.

        Returns:
            None
        """
        if self.dataframe is None:
            raise ValueError("No files have been loaded. Use load_files first.")
        self.dataframe.drop_duplicates(subset='title', keep='first', inplace=True)
        self.clean_wow_acm_dblp()
        self.dataframe.to_excel(output_file, index=False)

    def clean_wow_acm_dblp(self):
        """
        Clean and normalize specific columns in the dataframe.

        Returns:
            None
        """
        self.dataframe = self.dataframe.drop(columns=['Abstract - Foreign', 'Abstract - English Transliteration',
                                                      'Abstract - Foreign.1', 'Abstract - Korean', 'Article Title - SciELO',
                                                      'Article Title - SciELO.1', 'Article Title - Chinese', 'Article Title - Russian',
                                                      'Book Authors', 'Group Authors', 'Cited References', 'Language', 'Advisor',
                                                      'Committee Member', 'Copyright', 'Degree Name', 'No of References', 'Cited References',
                                                      'Since 2013 Usage Count', '180 Day Usage Count', 'Conference Date', 'Supplement',
                                                      "Early Access Date", "License URI", 'License Name', 'Article Number', 'Version',
                                                      'Version History', 'Meeting Abstract', 'Special Issue', 'Book Series Title',
                                                      'Book Series Subtitle', 'Publication Type', "Conference location", 'Author - Arabic',
                                                      'Source Title', 'Source Title - Korean', 'Patent Number', 'Patent Assignee',
                                                      'Source Title - Arabic', 'Publication Year', 'Institution Address', 'Institution',
                                                      'Dissertation and Thesis Subjects', 'Author Keywords', 'Conference Sponsor',
                                                      'Conference Location', 'UT (Unique ID)', 'Pubmed Id', 'Unnamed: 75',
                                                      'Book Group Authors', 'Book Editors', 'License Description', 'Eprint ID', "URLs",
                                                      'Indexed Date', 'Editors', 'Times Cited, WoS Core', 'Times Cited, CSCD',
                                                      'Times Cited, RSCI', 'Times Cited, ARCI', 'Times Cited, BCI', 'Times Cited, SCIELO',
                                                      'Times Cited, All Databases', 'Publication year', 'Series', 'Address'])
        self.dataframe['Pages'] = self.dataframe.apply(
            lambda row: f"{row['Start Page']}-{row['End Page']}" if pd.notna(row['Start Page']) and pd.notna(
                row['End Page']) else '',
            axis=1
        )

        # Eliminate the 'Start Page' and 'End Page' columns
        self.dataframe.drop(['Start Page', 'End Page'], axis=1, inplace=True)
        self.dataframe['Publisher'] = self.dataframe['Publisher'].fillna('')
        self.dataframe['Journal'] = self.dataframe['Journal'].fillna('')
        self.dataframe['Publisher'] = self.dataframe['Publisher'].fillna('') + ' ' + self.dataframe['Journal'].fillna('')
        self.dataframe.drop(['Journal'], axis=1, inplace=True)

        # Concatenate 'DOI' and 'Book DOI'
        self.dataframe['DOI'] = self.dataframe['DOI'].fillna('') + ' ' + self.dataframe['Book DOI'].fillna('')
        self.dataframe.drop(['Book DOI'], axis=1, inplace=True)
        self.dataframe['Document Type'] = self.dataframe['Document Type'].fillna('') + ' ' + self.dataframe['Item type'].fillna('')
        self.dataframe.drop(['Item type'], axis=1, inplace=True)
        self.dataframe['Publication Date'] = self.dataframe['Publication Date'].fillna('')
        self.dataframe['Date published'] = self.dataframe['Date published'].fillna('')
        self.dataframe['Year'] = self.dataframe['Year'].fillna('')

        self.dataframe['Publication Date'] = self.dataframe.apply(
            lambda row: f"{row['Publication Date']}{row['Date published']}" if pd.notna(
                row['Publication Date']) or pd.notna(row['Date published']) else '',
            axis=1
        )
        self.dataframe['Publication Date'] = self.dataframe.apply(
            lambda row: f"{row['Publication Date']}{row['Year']}" if pd.notna(
                row['Publication Date']) or pd.notna(row['Year']) else '',
            axis=1
        )
        self.dataframe['Publication Date'] = self.dataframe['Publication Date'].str.extract(r'(\d{4})', expand=False)
        # Eliminate the 'Date published' column
        self.dataframe.drop(['Date published'], axis=1, inplace=True)
        self.dataframe['Proceedings title'] = self.dataframe['Proceedings title'].fillna('')
        self.dataframe['Conference Title'] = self.dataframe['Conference Title'].fillna('')
        self.dataframe['Booktitle'] = self.dataframe['Booktitle'].fillna('')
        self.dataframe['Proceedings title'] = self.dataframe['Proceedings title'] + ' ' + self.dataframe['Conference Title']
        self.dataframe['Proceedings title'] = self.dataframe['Proceedings title'] + ' ' + self.dataframe['Booktitle']
        self.dataframe.drop(['Conference Title'], axis=1, inplace=True)
        self.dataframe = self.dataframe[['Authors', 'Researcher Ids', 'ORCIDs', 'title', 'Volume', 'Issue',
                                         'DOI', 'Document Type', 'Publication Date', 'Abstract', 'ISSN', 'eISSN',
                                         'ISBN', 'Pages', 'Publisher', 'Proceedings title', 'Keywords']]

if __name__ == "__main__":
    processor = ExcelFilesProcessor()
    dict_titles = {"WOW": "Article Title", "ACM": "Title", "DBLP": "Title", "IEEE": "Title"}
    for element in ["WOW", "DBLP", "ACM", "IEEE"]:
        processor.load_files(folder="systematic review/data/input_excel",
                              prefix=element)
        processor.eliminate_duplicates_normalize(
            output_file="systematic review/data/ouput_excel/" + element + "_joint.xlsx",
            article_title=dict_titles[element])
        processor.load_files(folder="systematic review/data/ouput_excel",
                              prefix=element)
        processor.multiple_dataframes_append(deepcopy(processor.dataframe))
    processor.multiple_dataframes_concat()
    # processor.joint_articles(
    #     output_file="systematic review/data/ouput_excel/result_WOW_ACM_DBLP_IEEE.xlsx")
    processor.joint_articles(
        output_file="systematic review/data/ouput_excel/test.xlsx")
