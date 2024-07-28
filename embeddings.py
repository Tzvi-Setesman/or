import openai
import numpy as np
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from credentials import OPENAI_API_KEY, EMBS_OPENAI_API_BASE
from retry import retry
import tiktoken
from tqdm import tqdm
import time

EMBED_DIM = 1536
MAX_TOKENS = 8192
EMNEDDING_MODEL = 'text-embedding-ada-002'

class Embedder:
    def __init__(self):    
        self.embedder = AzureOpenAIEmbeddings(
            azure_deployment=EMNEDDING_MODEL,
            openai_api_version="2023-05-15",
            azure_endpoint = EMBS_OPENAI_API_BASE,
            api_key = OPENAI_API_KEY
        )
        self.dim = EMBED_DIM
        self.max_tokens = MAX_TOKENS
        
    def trim_text_to_tokens(self, text):
        encoding = tiktoken.encoding_for_model(EMNEDDING_MODEL)
        tokens = encoding.encode(text)
        if len(tokens) > self.max_tokens:
            tokens = tokens[:self.max_tokens]
        encoding = tiktoken.encoding_for_model(EMNEDDING_MODEL)
        trimmed_text = encoding.decode(tokens)
        return trimmed_text
            
    @retry(openai.RateLimitError, tries=5, delay=1, backoff=2)        
    def embedd(self, text, normalize=True):
        text = self.trim_text_to_tokens(text)
        embedding = np.array(self.embedder.embed_query(text)) 
        if normalize:
            if not np.isclose(np.linalg.norm(embedding), 1, atol=1e-12):
                print ("embedding is not normalized: ", str(np.linalg.norm(embedding)), "normalizing...")
                embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def embed_df(self, df, text_col):
        print ("start embedding", time.ctime())
        tqdm.pandas()
        df["embeddings"] = df[text_col].progress_apply(lambda x: self.embedd(x))
        print ("finished embedding", time.ctime())
        return df

    

class SimilarityCalculator:
    
    def __init__(self, query_title, query, threshold=0.715):
        self.query_title = query_title
        self.query = query
        self.embedder = Embedder()
        self.query_embedding = self.embedder.embedd(self.query)
        self.threshold = threshold
    
    def calculate_similarity_on_df(self, df, query_title):
        tqdm.pandas()
        print ("start calculating similarity")
        similarity_col = query_title + "_similarity"
        df[similarity_col] = df['embeddings'].progress_apply(self.calculate_similarity)
        similarity_class_col = query_title + "_class_from_similarity"
        # embedding similarity threhold
        df[similarity_class_col] = (df[similarity_col] >= self.threshold).astype(int)
        print ("finished calculating similarity")
        return df

        
    def calculate_similarity(self, embedding):
        # assuming normalized vectors
        return np.dot(self.query_embedding, embedding) 

