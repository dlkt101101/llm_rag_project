import os, re, html, unicodedata, requests
import ollama, chromadb, tiktoken
from dotenv import load_dotenv
from datetime import datetime, timedelta, time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from llm_model import LLAMA_MODEL_ID

# Load environment variables
load_dotenv()

# getr api key from environment variable
api_key = os.getenv("API_KEY")

BASE_URL = "https://content.guardianapis.com"

DATE_NOW = datetime.now() # get the current date
TO_DATE_8AM = datetime.combine(DATE_NOW.date(), time(8, 0))
TO_DATE_8AM_ISO = TO_DATE_8AM.isoformat() # convert to ISO format

FROM_DATE_8AM = TO_DATE_8AM - timedelta(days=1) # get the date 24 hours ago at 8 AM
FROM_DATE_8AM_ISO = FROM_DATE_8AM.isoformat() # convert to ISO format

# create functions
# 1. fetch data from the endpoint
def fetch_endpoint_data(topic: str, endpoint: str, page_size: int=5):
    """
    Fetches data from the specified endpoint based on the provided topic and page size.

    Args:
        topic (str): The topic to search for.
        endpoint (str): The API endpoint to fetch data from.
        page_size (int): The number of results to return per page. Default is 10.
    
    Returns:
        dict: A dictionary containing the JSON response from the API.
    """
    parameters = {
        "q": topic,
        "page-size": page_size,
        "api-key": api_key,
        "from-date": FROM_DATE_8AM_ISO,
        "to-date": TO_DATE_8AM_ISO,
        "order-by": "newest",
        "type": "article",
        "show-fields": "headline,trailText,bodyText",
    }
    response = requests.get(url=f"{BASE_URL}/{endpoint}", params=parameters)
    return response.json()["response"]

# 2. normalize the data
def normalize_data(data:dict):
    results = data.get("results", [])
    data_list = []
    # get the relevant fields from the results
    for result in results:
        fields = result.get("fields", {})
        title = fields.get("headline", "")
        trail_text = fields.get("trailText", "")
        body = fields.get("bodyText", "")

        data_list.append({
            "text": "{title} \n {trail_text} \n\n {body}".format(title=title, trail_text=trail_text, body=body),
            "metadata": {
                "url": result.get("webUrl", ""),
                "id": result.get("id", ""),
                "title": title,
                "section_name": result.get("sectionName", "")
            }
        })
    return data_list

# 3. cleaning the data
def clean_text(text: list):
    """
    The function takes the normalized data and cleans it by removing any unwanted characters or formatting issues.
    """
    BOILERPLATE_PATTERNS = [
        r"^Sign up to .*$",
        r"^Photograph:.*$"]
    # remove unwanted characters or formatting issues from the text
    text = html.unescape(text)
    text = text.replace('\r\n', '\n')
    text = re.sub(r'<[^>]+>', ' ', text) # removing html tags
    text = unicodedata.normalize("NFKC", text)

    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines
            if not any(re.match(p, ln, re.IGNORECASE) for p in BOILERPLATE_PATTERNS)]
    text = "\n".join(lines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def clean_data(data_list: dict):
    """
    The function takes the normalized data and cleans it by using the function `clean_text`.
    It returns a list of cleaned data dictionaries containing the cleaned text and metadata.
    """
    cleaned_data = []
    for data in data_list:
        cleaned_text =  clean_text(data.get("text", ""))
        cleaned_data.append({
            "text": cleaned_text,
            "metadata": data.get("metadata", {})
        })
    return cleaned_data

# 3. chunking the data
# Define a Recursive Token Splitter
def ollama_token_count(cleaned_data: dict) -> int:
    """
    The function returns the number of tokens in a string using the Ollama tokenizer.
    """
    text = cleaned_data.get("text", "")
    encoding = AutoTokenizer.from_pretrained(LLAMA_MODEL_ID)
    num_tokens= len(encoding.encode(text))
    return num_tokens

recursiveSplitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=0.15*512,
    length_function=ollama_token_count)

# Chunking the data
def chunk_data(cleaned_data: list):
    """
    The function takes the cleanded data from the function `clean_data` and chunks it
    """
    chunked_data = []
    for data in cleaned_data:
        text = data.get("text", "")
        chunks = recursiveSplitter.split_text(text)
        chunked_data.append({
            "chunks": chunks,
            "metadata": data.get("metadata", {})
        })
    return chunked_data

# 4. embedding the data
def embed_data(chunked_data:list, embedding_model):
    """
    The function takes the chunked data from the function `chunk_data` and embeds it using the provided embedding model.
    """
    response = ollama.embeddings(model=LLAMA_MODEL_ID, prompt=chunk)
    embeddings = response.get("embeddings", [])

    pass

# 5. storing the data in a vector database using ChromaDB
# initialize the vectordb
client = chromadb.Client()
collection = client.create_collection(name="news_data")
def store_data_in_chromadb():
    pass

endpoint_data = fetch_endpoint_data(topic="technology", endpoint="search", page_size=2)
print(endpoint_data)

normalized_data = normalize_data(endpoint_data)
print(normalized_data)

cleaned_data = clean_data(normalized_data)
print("#"*30)
print("cleaned data: ")
print(cleaned_data)
