import pandas as pd
from llm import LLM
import tiktoken
from langchain.schema import HumanMessage
from tqdm import tqdm
from langchain_community.callbacks import get_openai_callback


class ArticleSummarizer:
    def __init__(self):
        self.gpt4 = LLM.create_llm('gpt4')
        self.gpt432k = LLM.create_llm('gpt432k') 
        self.current_llm = self.gpt4
        self.cost = 0

    def choose_llm(self, text):
        encoding = tiktoken.encoding_for_model('gpt-4')
        num_tokens = len(encoding.encode(text))
        if num_tokens < 7000:
            self.current_llm =  self.gpt4
        else:
            print ("use gpt4-32k for summarizing")
            self.current_llm =  self.gpt432k

    def get_prompt(self, text):
        promt_template = """I am a researcher analyzing media coverage of the ongoing conflict
        between Israel and Gaza, with a focus on the war, the surrounding politics, 
        and aspects of anti-Semitism related to the conflict. 
        I would like you to read a specific press article and provide a summary. 
        Please pay special attention to the sentiment towards Israel's position. 
        Highlight if the article is overtly against Israel or if it presents a complex view that 
        acknowledges Israel's needs while also considering Palestinian humanitarian concerns. 
        Ensure the summary reflects whether the article is written from a neutral perspective 
        or is biased towards one side. Exclude any parts of the article that are not 
        related to the Israel-Gaza war or anti-Semitism. 
        If the entire article is irrelevant to these topics, or if from some reason the provided text seems to be not an article, please respond with onlt the following:
        'not relevant to the Israel-Gaza conflict'. \n\nArticle: {0} \n\nSummary:"""
        prompt = promt_template.format(text)
        return prompt
    
    def summarize(self, text):
        prompt = self.get_prompt(text)
        self.choose_llm(prompt)
        message = HumanMessage(content=prompt)
        try:
            with get_openai_callback() as cb:
                summary = self.current_llm([message]).content
                self.cost += cb.total_cost
        except Exception as e:
            summary = "Azure content filter not allowing to summarize this text"
            print("the following article wasm't summarize due to Azure content filter: {}".format(text))
        return summary

    def summarize_df(self, df, text_column):
        tqdm.pandas()
        print ("start summarizing {} articles".format(len(df)))
        df["summary"] = df[text_column].progress_apply(self.summarize)
        df["summary"] = df["summary"].apply(lambda x: '0' if 'not relevant to the Israel-Gaza conflict' in x else x)
        print ("finished summarizing articles. Total cost for gpt summarization: {0:.2f} $".format(self.cost))
        return df
        
