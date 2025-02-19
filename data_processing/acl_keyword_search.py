import pandas as pd
import os
project_path =os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
import re

class KeywordSearchACL:
    """
    A class that searches for keywords in ACL Anthology files using AND/OR queries.

    Args:
        data_path (str): Path to the ACL Anthology Excel file.

    Attributes:
        df (pd.DataFrame): DataFrame containing the extracted data.

    Methods:
        search_keywords(query: str): Searches for keywords in the Title and Abstract.
    """

    def __init__(self, data_path: str):
        self.df = pd.read_excel(data_path)

    def parse_query(self, query: str):
        """
        Parses the query into separate keywords and logical operators (AND/OR).

        Args:
            query (str): The search query (e.g., "toxicity AND 'language models'").

        Returns:
            list: A list of terms and operators in sequence.
        """
        # Regular expression to extract words and phrases
        split_pattern = r'\s+(AND|OR)\s+'
        tokens = re.split(split_pattern, query)
        tokens = [t.strip('"') for t in tokens]  # Remove quotes
        return tokens

    def filter_dataframe(self,df, tokens_title=None, tokens_abstract=None):
        def evaluate_logical_expression(tokens, text):
            """Evaluates the logical condition using 'AND'/'OR' in tokens."""
            if not tokens or not text or type(text) != str:
                return False
            
            # Convert the text to lowercase for case-insensitive matching
            text = text.lower()
            
            # Convert tokens list into a Boolean expression
            expression = []
            for token in tokens:
                if token.upper() == "AND":
                    expression.append("and")
                elif token.upper() == "OR":
                    expression.append("or")
                else:
                    # Ensure proper string checking in text
                    expression.append(f'("{token.lower()}" in text)')

            # Join expression list into a valid Python condition
            expression_str = " ".join(expression)

            try:
                return eval(expression_str, {"text": text})
            except Exception as e:
                print(f"Error evaluating query: {expression_str} -> {e}")
                return False

        # Filter rows based on title
        if tokens_title and tokens_abstract:
            df = df[df["Title"].apply(lambda x: evaluate_logical_expression(tokens_title, x)) | df["Abstract"].apply(lambda x: evaluate_logical_expression(tokens_abstract, x))]
            
        if tokens_title and not tokens_abstract:    
            df = df[df["Title"].apply(lambda x: evaluate_logical_expression(tokens_title, x))]

        # Filter rows based on abstract
        if tokens_abstract and not tokens_title:
            df = df[df["Abstract"].apply(lambda x: evaluate_logical_expression(tokens_abstract, x))]

        return df

    def search_keywords(self, query_title: str=None, query_abstract: str=None):
        """
        Searches for keywords in the ACL Anthology files based on an AND/OR query.

        Args:
            query (str): Search query with AND/OR operators (e.g., "toxicity AND 'language models'").

        Returns:
            pd.DataFrame: Filtered results.
        """
        if query_title is not None:
            tokens_title = self.parse_query(query_title)
        else:
            tokens_title = None
        if query_abstract is not None:
            tokens_abstract = self.parse_query(query_abstract)
        else:
            tokens_abstract = None
        return self.filter_dataframe(df=self.df,tokens_title=tokens_title, tokens_abstract=tokens_abstract)

if __name__=="__main__":
    acl_filter = KeywordSearchACL(project_path+'/data/raw_acl_data/ACL_database.xlsx')
    df=acl_filter.search_keywords(query_title="toxic", query_abstract="toxic AND large language models")
    df.to_excel(project_path+'/data/input_excel/ACL_toxic_language_models.xlsx', index=False)
    df = acl_filter.search_keywords(query_title="toxicity", query_abstract="toxicity AND large language models")
    df.to_excel(project_path+'/data/input_excel/ACL_toxicity_language_models.xlsx', index=False)
    df = acl_filter.search_keywords(query_title="toxic AND transformers", query_abstract="toxic AND transformers")
    df.to_excel(project_path+'/data/input_excel/ACL_toxic_transformers.xlsx', index=False)
    df = acl_filter.search_keywords(query_title="toxic AND language detection", query_abstract="toxic AND language detection")
    df.to_excel(project_path+'/data/input_excel/ACL_toxic_language_detection.xlsx', index=False)
    df = acl_filter.search_keywords(query_title="toxic AND language", query_abstract="toxic AND language")
    df.to_excel(project_path+'/data/input_excel/ACL_toxic_language.xlsx', index=False)
    df = acl_filter.search_keywords(query_title="toxicity detection", query_abstract="toxicity detection")
    df.to_excel(project_path+'/data/input_excel/ACL_toxicity_detection.xlsx', index=False)
    df = acl_filter.search_keywords(query_title="toxicity detector", query_abstract="toxicity detector")
    df.to_excel(project_path+'/data/input_excel/ACL_toxicity_detector.xlsx', index=False)
    df = acl_filter.search_keywords(query_title="toxic AND span detection", query_abstract="toxic AND span detection")
    df.to_excel(project_path+'/data/input_excel/ACL_toxic_span_detection.xlsx', index=False)
    df = acl_filter.search_keywords(query_title="toxic dataset", query_abstract="toxic dataset")
    df.to_excel(project_path+'/data/input_excel/ACL_toxic_dataset.xlsx', index=False)
    df = acl_filter.search_keywords(query_title="toxicity AND dataset", query_abstract="toxicity AND dataset")
    df.to_excel(project_path+'/data/input_excel/ACL_toxicity_dataset.xlsx', index=False)
    df = acl_filter.search_keywords(query_title="mitigate AND toxicity", query_abstract="mitigate AND toxicity")
    df.to_excel(project_path+'/data/input_excel/ACL_mitigate_toxicity.xlsx', index=False)
    df = acl_filter.search_keywords(query_title="mitigating AND toxicity", query_abstract="mitigating AND toxicity")
    df.to_excel(project_path+'/data/input_excel/ACL_mitigating_toxicity.xlsx', index=False)
    df = acl_filter.search_keywords(query_title="self-correct language", query_abstract="self-correct language")
    df.to_excel(project_path+'/data/input_excel/ACL_self_correct_language.xlsx', index=False)
    df = acl_filter.search_keywords(query_title="detoxify", query_abstract="detoxify AND language model")
    df.to_excel(project_path+'/data/input_excel/ACL_detoxify_language_model.xlsx', index=False)
    df = acl_filter.search_keywords(query_title="controlled AND text generation AND language model", query_abstract="controlled AND text generation AND language model")
    df.to_excel(project_path+'/data/input_excel/ACL_controlled_text_generation_language_model.xlsx', index=False)
    df = acl_filter.search_keywords(query_title="controllable AND text generation", query_abstract="controllable AND text generation")
    df.to_excel(project_path+'/data/input_excel/ACL_controllable_text_generation.xlsx', index=False)
    df = acl_filter.search_keywords(query_title="detoxification", query_abstract="detoxification")
    df.to_excel(project_path+'/data/input_excel/ACL_detoxification.xlsx', index=False)
    df = acl_filter.search_keywords(query_title="counter narrative generation", query_abstract="counter narrative generation")
    df.to_excel(project_path+'/data/input_excel/ACL_counter_narrative_generation.xlsx', index=False)
    