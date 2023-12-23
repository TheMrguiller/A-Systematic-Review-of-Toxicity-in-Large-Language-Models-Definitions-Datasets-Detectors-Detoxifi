
import os,sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path
from utils.google_search import search,get_useragent
import requests
import pdfkit
import pdfplumber
from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from habanero import Crossref
from crossref.restful import Works
from bs4 import BeautifulSoup
import random 
from time import sleep
import pandas as pd
import re
from fuzzywuzzy import fuzz
import fitz
from utils.multi_column import column_boxes
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class AbstractExtractor:
    def __init__(self,dataframe) -> None:
        """
            Retrieves the DOI for a given query.

        populate_DOI():
            Populates the DOI column in the dataframe.

        abstract_from_DOI(DOI):
            Retrieves the abstract for a given DOI.

        download_html(url, save_path, download_html=False):
            Downloads the HTML content from a given URL.

        find_abstract_element(html_content):
            Finds the abstract element in the HTML content.

        check_pdf_in_link(link):
            Checks if a given link is a PDF link.

        store(store_path="test_.xlsx"):
            Stores the dataframe to an Excel file.

        similarity_score(query, result):
            Calculates the similarity score between a query and a result.

        filter_results_similarity(query, results, threshold=50):
            Filters the results based on the similarity score.

        sort_links(links, query):
            Sorts the links based on priority domains and similarity score.

        custom_by_domain(links):
            Custom sorting function for links based on priority domains.

        download_pdf(link, store_path):
            Downloads a PDF file from a given link.

        extract_abstract_section(text):
            Extracts the abstract section from a given text.

        remove_html_tags(text):
            Removes HTML tags from a given text.

        extract_abstract_from_pdf(store_path):
            Extracts the abstract from a PDF file.

        test(abstract_column_name='Abstract', title_column_name='title'):
            Performs the abstract extraction process for the dataframe.
    """
        self.dataframe=dataframe
        self.cr = Crossref()
        self.works = Works()
        
    def search_link(self, query, num_results=10, proxy=None, sleep_interval=0):
            """
            Search for links using a given query.

            Args:
                query (str): The search query.
                num_results (int, optional): The number of search results to retrieve. Defaults to 10.
                proxy (str, optional): The proxy server to use for the search. Defaults to None.
                sleep_interval (int, optional): The time interval to sleep between requests. Defaults to 0.

            Returns:
                list: A list of links retrieved from the search.
            """
            links = search(query, num_results=num_results, proxy=proxy, sleep_interval=sleep_interval, advanced=True)
            return links
        
    def get_DOI(self, query):
        """
        Retrieves the DOI (Digital Object Identifier) for a given query.

        Args:
            query (str): The query to search for.

        Returns:
            str or None: The DOI if found, or None if not found or an error occurred.
        """
        try:
            result = self.cr.works(query=query)
            if 'items' in result['message'] and result['message']['items']:
                return result['message']['items'][0]['DOI']
            else:
                return None
        except Exception as e:
            print(f"Error obtaining DOI in query:'{query}': {e}")
            return None
    def populate_DOI(self):
        """
        Populates the 'DOI' column of the dataframe with DOI values for rows that have blank abstracts.

        This method iterates over the rows of the dataframe and retrieves the DOI for each row's title using the
        'get_DOI' method. If a DOI is found, it is assigned to the 'DOI' column of the corresponding row in the dataframe.

        Note: This method introduces a random delay between 0.1 and 3.0 seconds using the 'sleep' function from the 'random' module.

        Returns:
            None
        """
        rows_with_blank_abstracts = self.dataframe[self.dataframe["DOI"].isnull() | (self.dataframe["DOI"] == '')| (self.dataframe["DOI"] == ' ')]
        for index, row in rows_with_blank_abstracts.iterrows():
            doi=self.get_DOI(row["title"])
            if doi is not None:
                self.dataframe.at[index,"DOI"]=doi
            sleep(random.uniform(0.1,3.0))

    def abstract_from_DOI(self, DOI):
        """
        Retrieves the abstract of a publication using its DOI.

        Args:
            DOI (str): The DOI (Digital Object Identifier) of the publication.

        Returns:
            str or None: The abstract of the publication if available, None otherwise.
        """
        result = self.works.doi(DOI)
        if result is not None:
            try:
                abstract = result['abstract']
                return abstract
            except KeyError:
                return None
        return None
    
    def download_html(self, url, save_path, download_html=False):
        """
        Downloads the HTML content from the specified URL and saves it to a file.

        Args:
            url (str): The URL to download the HTML content from.
            save_path (str): The path to save the downloaded HTML content.
            download_html (bool, optional): Whether to save the HTML content to a file. 
                Defaults to False.

        Returns:
            str: The downloaded HTML content as a string, or None if an error occurred.
        """
        try:
            # Send a GET request to the URL
            session = requests.Session()
            response = session.get(url, timeout=5, headers={"User-Agent": get_useragent()}, verify=False)
            response.raise_for_status()  # Raise an exception for HTTP errors
            if download_html:
                with open(save_path, 'w', encoding='utf-8') as file:
                    file.write(response.text)
            return response.text

        except requests.exceptions.HTTPError as errh:
            print(f"HTTP Error: {errh}")
        except requests.exceptions.ConnectionError as errc:
            print(f"Error Connecting: {errc}")
        except requests.exceptions.Timeout as errt:
            print(f"Timeout Error: {errt}")
        except requests.exceptions.RequestException as err:
            print(f"Request Exception: {err}")
        return None
    
    def find_abstract_element(self, html_content):
        """
        Finds and returns the abstract element from the given HTML content.

        Args:
            html_content (str): The HTML content to search for the abstract element.

        Returns:
            str or None: The full content of the abstract element, or None if not found.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        abstract_title = soup.find(lambda tag: tag.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'] and 'Abstract' in tag.text)
        if abstract_title:           
            abstract_container = abstract_title.find_parent()
            if abstract_container:                
                full_content = str(abstract_container.get_text().strip())
                return full_content
        abstract_elements = soup.find_all(lambda tag: tag.get('class') and 'abstract' in ' '.join(tag.get('class')).lower())
        if abstract_elements:           
            abstract = abstract_elements[0].get_text().strip()
            return abstract
        return None
 
    def check_pdf_in_link(self, link):
        """
        Check if the given link contains the string 'pdf'.

        Args:
            link (str): The link to be checked.

        Returns:
            bool: True if the link contains 'pdf', False otherwise.
        """
        if "pdf" in link:
            return True
        else:
            return False
        
    def store(self, store_path="/home/d4k/Documents/guillermo/doctorado/systematic_review/data/ouput_excel/test_.xlsx"):
        """
        Store the dataframe as an Excel file.

        Args:
            store_path (str): The path to save the Excel file. Defaults to "test_.xlsx".
        """
        self.dataframe.to_excel(store_path, index=False)
    
    def similarity_score(self, query, result):
        """
        Calculates the similarity score between a query and a result using the fuzz.ratio method.

        Parameters:
            query (str): The query string.
            result (str): The result string.

        Returns:
            int: The similarity score between the query and the result.
        """
        return fuzz.ratio(query.lower(), result.lower())
    
    def filter_results_similarity(self, query, results, threshold=50):
            """
            Filters the results based on the similarity score between the query and the title of each result.

            Args:
                query (str): The query string.
                results (list): A list of Result objects.
                threshold (int, optional): The minimum similarity score required for a result to be included in the filtered results. Defaults to 50.

            Returns:
                list: A list of URLs of the filtered results.
            """
            filtered_results = []
            for result in results:
                score = self.similarity_score(query, result.title)
                if score < 50 and result.title in query:
                    score = 51
                if score >= threshold:
                    filtered_results.append(result.url)
            return filtered_results
    
    def sort_links(self, links, query):
        """
        Sorts the given list of links based on their similarity to the query and the custom domain sorting.

        Args:
            links (list): The list of links to be sorted.
            query (str): The query used for similarity comparison.

        Returns:
            list: The sorted list of links.
        """
        sorted_links = sorted(self.filter_results_similarity(query=query, results=links), key=self.custom_by_domain)
        return sorted_links
    
    def custom_by_domain(self, links):
        """
        Determines the priority of a list of links based on their domain.

        Args:
            links (list): A list of links.

        Returns:
            tuple: A tuple containing the priority value and the original list of links.
                   The priority value is 0 if any of the links belong to the priority domains,
                   otherwise it is 1.
        """
        priority_domains = ['arxiv.org', 'aclanthology.org', 'dl.acm.org']

        for domain in priority_domains:
            if domain in links:
                return (0, links)
        return (1, links)
    
    def download_pdf(self, link, store_path):
        """
        Downloads a PDF file from the given link and stores it at the specified path.

        Args:
            link (str): The URL of the PDF file to download.
            store_path (str): The path where the downloaded PDF file will be stored.

        Returns:
            bool: True if the PDF is downloaded and stored successfully, False otherwise.
        """
        response = requests.get(link, timeout=5, headers={"User-Agent": get_useragent()}, verify=False)

        if response.status_code == 200:
            with open(store_path, 'wb') as archivo_local:
                archivo_local.write(response.content)
            print(f"PDF stored successfully at '{store_path}'")
            return True
        else:
            print(f"Error in PDF download. Code: {response.status_code}")
            return False
    def extract_abstract_section(self, text):
        """
        Extracts the abstract section from the given text.

        Args:
            text (str): The text to extract the abstract section from.

        Returns:
            str or None: The extracted abstract content, or None if no abstract section is found.
        """
        # Utiliza una expresión regular para capturar el contenido del abstract
        match = re.search(r'Abstract\s*(?::)?\s*(.*?)(?=1\sIntroduction|$)', text, re.DOTALL | re.IGNORECASE)
        if match:
            abstract_content = match.group(1).strip()
            return abstract_content
        else:
            return None
    def remove_html_tags(self, text):
        """
        Removes HTML tags from the given text.

        Args:
            text (str): The text containing HTML tags.

        Returns:
            str: The text with HTML tags removed.
        """
        clean_text = re.sub('<.*?>', '', text)
        return clean_text
    def extract_abstract_from_pdf(self, store_path):
        """
        Extracts the abstract from a PDF file.

        Args:
            store_path (str): The path to the PDF file.

        Returns:
            str or None: The extracted abstract, or None if no abstract is found.
        """
        try:
            doc = fitz.open(store_path)
        except fitz.fitz.FileDataError as e:
            print(f"Error opening the PDF file: {e}")
            return None

        for page in doc:
            text = ""
            block_result = ""
            page_result = ""
            bboxes = column_boxes(page, footer_margin=50, no_image_text=True)

            for rect in bboxes:
                block_text = page.get_text(clip=rect, sort=True)
                block_text = self.remove_html_tags(block_text)
                text += block_text
                block_result_ = self.extract_abstract_section(block_text)
                if block_result_ is not None:
                    block_result = block_result_

            page_result_ = self.extract_abstract_section(text)
            if page_result_ is not None:
                page_result = page_result_

            if page_result is not None and block_result is not None:
                if len(page_result) > len(block_result):
                    return page_result
                else:
                    return block_result
            elif page_result is not None:
                return page_result
            elif block_result is not None:
                return block_result

        return None
    def test(self, abstract_column_name='Abstract', title_column_name='title'):
            """
            This method extracts abstracts for rows with blank abstracts in the dataframe.
            
            Parameters:
            - abstract_column_name (str): The name of the column containing abstracts in the dataframe. Default is 'Abstract'.
            - title_column_name (str): The name of the column containing titles in the dataframe. Default is 'title'.
            """
        
            rows_with_blank_abstracts = self.dataframe[self.dataframe[abstract_column_name].isnull() | (self.dataframe[abstract_column_name] == '')]
            count=0
            for index,row in rows_with_blank_abstracts.iterrows():
                query = row['title']
                links=self.sort_links(self.search_link(query=query,sleep_interval=random.randint(a=5,b=10)),query)
                for link in links:
                    
                    if self.check_pdf_in_link(link):
                        #Descargar el pdf y utilizar mi modelo o otra tecnica.
                        self.download_pdf(link,store_path="/home/d4k/Documents/guillermo/doctorado/systematic_review/data/execution_time_data/out.pdf")
                        abstract=self.extract_abstract_from_pdf(store_path="/home/d4k/Documents/guillermo/doctorado/systematic_review/data/execution_time_data/out.pdf")
                        if abstract is not None:
                            self.dataframe.at[index,abstract_column_name]=abstract
                            break
                    else:
                        abstract=self.abstract_from_DOI(row["DOI"])
                        if abstract is None:
                            html_content=self.download_html(link,"/home/d4k/Documents/guillermo/doctorado/systematic_review/data/execution_time_data/out.html")
                            if html_content is not None:
                                abstract=self.find_abstract_element(html_content=html_content)
                                if abstract is not None:
                                    if len(abstract.strip().split(" "))>1:
                                        self.dataframe.at[index,abstract_column_name]=abstract
                                        break
                        else:
                            abstract=re.search(r'<jats:p>(.*?)<\/jats:p>', abstract, re.DOTALL).group(1)
                            self.dataframe.at[index,abstract_column_name]=abstract
                            break
                    sleep(random.uniform(0.1,3.0))
                if count%10==0:
                    self.store()
                count+=1
            self.store()

class abstractExtractorMistrall:
    #TODO
    def __init__(self,dataframe,max_token_length,instruction=None) -> None:
        self.llm = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.1-GGUF", model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf", model_type="mistral", gpu_layers=100,max_new_tokens = 4096,
                                           context_length = max_token_length)
        self.autotokenizer= AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        self.abstractExtractor= AbstractExtractor(dataframe=dataframe)
        if instruction is None:
            chunk=""
            instruction=f"""
                            <s>[INST] You are an expert reader assistant. Your task is to extract the abstract of a research from the given text, and you must provide the abstract exactly as it is formatted. If the provided text does not contain an abstract, the output format must be "No abstract provided."
                            Text: {chunk}
                            Abstract:[/INST]</s>
                            """
        self.max_token_length=max_token_length-self.instruction_length(instruction,max_token_length)
    def instruction_length(self,instruction,max_token_length):
        token_length=len(self.autotokenizer.tokenize(text=instruction,max_length=4096,truncation=False))
        if token_length>max_token_length:
            raise Exception("Instruction to long for the model token length size") 
        else:
            return token_length
    
    def extractabstract(self,store_path,pdf_name="out.pdf"):
        with pdfplumber.open(store_path+pdf_name) as pdf:
                pages = pdf.pages
                for page in pages:
                    text = page.extract_text()
                    for chunk in self.chunk_text(text):
                        instruction = f"""
                            <s>[INST] You are an expert reader assistant. Your task is to extract the abstract of a research from the given text, and you must provide the abstract exactly as it is formatted. If the provided text does not contain an abstract, the output format must be "No abstract provided."
                            Text: {chunk}
                            Abstract:[/INST]</s>
                            """
                        output=self.llm(instruction,temperature=1.0,seed=42)
                        if not "No abstract provided" in output:
                            return output
                            break
                return None
    def chunk_text(self,text):
        #TODO
        #Cojo el texto lo divido en parrafos. Cojo todos los parrafos lo convierto a tokens. Inicializo una variable que sea la longitud en tokens que llevo para un chunk
        # Inicializo una lista con los chunks que llevo y otra donde se iran guardando los chunks finales.
        #Voy añadiendo poco a poco sumando el tamaño de cada parrafo en tokens a mi variable de longitud. A la vez si la longitud no supera un maximo voy incluyendo los parrafos en el chunk actual.
        # Si se ha llegado al chunk actual se mete en la lista de chunks y se vacia el chunk actual. La idea es cada vez que se inicie uno nuevo quedarse con el chunk anterior siempre y cuando permita sumarle el siguiente chunk de tokens para mantener el contexto.
        paragraphs = text.split('\n')
        current_chunk=""

        all_chunks=[]
        current_token_length=[]
        for paragraph in paragraphs:
            
            # Tokeniza el párrafo completo
            split_delimeter='\n'
            paragraph_tokens_length = len(self.autotokenizer.tokenize(text=paragraph+split_delimeter,max_length=self.max_token_length,truncation=False))
            if paragraph_tokens_length <=self.max_token_length:
                if sum(current_token_length)+ paragraph_tokens_length >self.max_token_length:
                    all_chunks.append(current_chunk)
                    if current_token_length[-1]+paragraph_tokens_length >self.max_token_length:
                        current_token_length.clear()
                        current_token_length.append(paragraph_tokens_length)
                        current_chunk=paragraph+split_delimeter
                    else:
                        initial_length=paragraph_tokens_length+current_token_length[-1]
                        current_token_length.clear()
                        current_token_length.append(initial_length)
                        current_chunk=current_chunk.split(split_delimeter)[-1]+split_delimeter+paragraph+split_delimeter

                else:
                    
                    current_chunk+=paragraph+split_delimeter
                    current_token_length.append(paragraph_tokens_length)
                
            else:
                # Si ocurre un error de tokenización, divide el párrafo en oraciones y tokeniza cada oración
                split_delimeter='.'
                sentences = paragraph.split(split_delimeter)  # Puedes ajustar esto para manejar otros delimitadores de oraciones
                current_paragraph = ""
                for index,sentence in enumerate(sentences):
                    if index == len(sentences) - 1:
                        split_delimeter+='\n'
                    
                    sentence_tokens_length = len(self.autotokenizer.tokenize(sentence+split_delimeter,max_length=self.max_token_length,truncation=False))
                    if sentence_tokens_length <=self.max_token_length:
                        # Si agregar la oración actual excede el límite max_length, agrega el párrafo actual a la lista y comienza uno nuevo
                        if sum(current_token_length)+ sentence_tokens_length >self.max_token_length:
                            if current_paragraph=="":
                                all_chunks.append(current_chunk)
                                current_paragraph=sentence+split_delimeter
                                current_token_length.clear()
                                current_token_length.append(sentence_tokens_length)
                            else:
                                all_chunks.append(current_chunk+current_paragraph)
                                if current_token_length[-1]+sentence_tokens_length >self.max_token_length:
                                    current_paragraph=sentence+split_delimeter
                                    current_token_length.clear()
                                    current_token_length.append(sentence_tokens_length)
                                else:
                                    initial_length=sentence_tokens_length+current_token_length[-1]
                                    current_token_length.clear()
                                    current_token_length.append(initial_length)
                                    current_paragraph=current_paragraph.split(split_delimeter)[-1]+split_delimeter+sentence+split_delimeter
                                    
                            
                        else:
                            current_paragraph += sentence+split_delimeter
                            current_token_length+= sentence_tokens_length

                    else :
                        # Manejar errores adicionales de tokenización si es necesario
                        raise Exception(f"Sentence too long for the model: {sentence}")
                
                

        return all_chunks

    
    def obtain_pdf(self,link,store_path,pdf_name="out.pdf"):
        #True indicates correct download
        if self.abstractExtractor.check_pdf_in_link(link):
            return self.download_pdf(link=link,store_path=store_path+pdf_name)
        else:
            try:
                pdfkit.from_url(link, store_path+pdf_name)
                return True
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 403:
                    print(f"Error 403 Forbidden: Access to {link} is restricted.")
                else:
                    print(f"HTTP Error: {e}")
                return False

    def download_pdf(self, store_path,link):
        
        response = requests.get(link,timeout=5,headers={
            "User-Agent": get_useragent() },verify=False)

        
        if response.status_code == 200:
            
            with open(store_path, 'wb') as archivo_local:
                archivo_local.write(response.content)
            print(f"PDF store succesfully '{store_path}'")
            return True
        else:
            print(f"Error in PDF download. Code: {response.status_code}")
            return False
    def test(self,abstract_column_name='Abstract',title_column_name='title'):
        rows_with_blank_abstracts = self.abstractExtractor.dataframe[self.abstractExtractor.dataframe[abstract_column_name].isnull() | (self.abstractExtractor.dataframe[abstract_column_name] == '')]
        count=0
        for index,row in rows_with_blank_abstracts.iterrows():
            query = row['title']
            links=self.abstractExtractor.search_link(query=query,sleep_interval=random.randint(a=5,b=10))
            for link in links:
                if self.obtain_pdf(link,"./pdf/"):
                    #Descargar el pdf y utilizar mi modelo o otra tecnica.
                    abstract=self.extractabstract("./pdf/")
                    if abstract is not None:
                        self.abstractExtractor.dataframe.at[index,abstract_column_name]=abstract
                        break
                
                sleep(random.uniform(0.1,3.0))
            if count%10==0:
                self.abstractExtractor.store()
            count+=1
        self.abstractExtractor.store()
        pass
    def test_extract(self):
        abstract=self.extractabstract("./pdf/")
        if abstract is not None:
            pass
if __name__=="__main__":
    result=pd.read_excel("/home/d4k/Documents/guillermo/doctorado/systematic_review/data/ouput_excel/result_WOW_ACM_DBLP_IEEE_final_.xlsx")
    test=AbstractExtractor(result)
    # test.populate_DOI()
    # test.store()
    test.test()