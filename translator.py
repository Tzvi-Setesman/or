from llm import LLM
from langchain.schema import HumanMessage
from langchain_community.callbacks import get_openai_callback
from retry import retry
import openai
from tqdm import tqdm
import requests
import uuid
from credentials import AZURE_TRANSLATOR_KEY, AZURE_TRANLATOR_ENDPOINT

class OpenaiTranslator:
    def __init__(self, text_column, source_lang, target_lang="English"):
        self.llm = LLM.create_llm('gpt4')
        self.cost = 0
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.text_column = text_column
    
    @retry(openai.RateLimitError, tries=5, delay=1, backoff=2)        
    def translate(self, text):
      prompt = "Translate the following text from {} to {}: {}".format(self.source_lang, self.target_lang, text)
      message = HumanMessage(content=prompt)      
      with get_openai_callback() as cb:
        try:
          tranlation = self.llm([message]).content
        except Exception as e:
          tranlation = text
          print(" didn't translated due to Azure content filter")
        self.cost += cb.total_cost
      return tranlation
    
    def translate_df(self, df):
        tqdm.pandas()
        print ("start translation")
        trans_col = self.text_column + "_translated" 
        df[trans_col] = df[self.text_column].progress_apply(lambda x: self.translate(x))
        print ("Translation ended. Total cost for Translation: {0:.2f} $".format(self.cost))
        return df
    
class AzureTranslator:
  
    def __init__(self, text_column, source_lang, target_lang="en"):
        self.API_KEY = AZURE_TRANSLATOR_KEY
        self.API_ENDPOINT = AZURE_TRANLATOR_ENDPOINT
        self.location = "eastus"
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.text_column = text_column

        
    def translate(self, text):
        translated_text = []
        is_split = False
        if len(text) > 50000:
            text_list = text.split('. ')
            is_split = True
        else:
            text_list = [text]
            
        params = {
            'api-version': '3.0',
            'from': self.source_lang,
            'to': self.target_lang
        }
        
        headers = {
            'Ocp-Apim-Subscription-Key': self.API_KEY,
            'Ocp-Apim-Subscription-Region': self.location,
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }
        
        path = '/translate'
        constructed_url = self.API_ENDPOINT  + path
        for t in text_list:
            body = [{
                'text': f'{t}'
            }]
            request = requests.post(constructed_url, params=params, headers=headers, json=body)
            response = request.json()
            
            try:
                translated_text.append(response[0].get('translations')[0].get('text'))
            except Exception as e:
                raise e
        if is_split:
            return '. '.join(translated_text).strip()
        else:
            return translated_text[0].strip()

    def translate_df(self, df):
      tqdm.pandas()
      print ("start translation")
      trans_col = self.text_column + "_translated" 
      df[trans_col] = df[self.text_column].progress_apply(lambda x: self.translate(x))
      print ("Translation ended")
      return df

class LangugeDetector:
    def __init__(self, default_lang=None):
        from langdetect import detect
        self.detect = detect
        self.default_lang = "unknown" if default_lang is None else default_lang
    
    def detect_language(self, text):
        try:
            lang = self.detect(text)
            if lang not in ["en", "he", "ar", "fa", self.default_lang]:
                return self.default_lang
            else:
                return lang
        except:
            return self.default_lang
    
    def detetct_language_on_df(self, df, text_column):
        print ("start detecting language")
        tqdm.pandas()
        df["lang"] = df[text_column].progress_apply(lambda x: self.detect_language(x))
        print ("finished detecting language")
        return df
