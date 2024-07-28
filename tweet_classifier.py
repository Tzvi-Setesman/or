import json
import re
import os
import time
import warnings
from enum import Enum
import pandas as pd
from langchain.chains import create_tagging_chain
from langchain.chains.llm import LLMChain
from langchain.output_parsers.list import CommaSeparatedListOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.callbacks import get_openai_callback
from tqdm import tqdm
import langchain_schemas as schemas
from llm import LLM, create_my_openai_chain
from paths import GPT_OUTPUTS_BACKUP_PATH

warnings.filterwarnings('ignore')

from credentials import MODEL_NAME

class TaskName(Enum):
    IsraelRelatedTweet = "Israel-related"
    IsraelSentimentTweet = "Israel-sentiment"
    IsraelSentimentTweetEurope = "Israel-sentiment-europe"
    IsraelMultiLabelTweet = "Israel-Multi-label"
    HouthisTweet = "Houthis"
    Unrwa = "Unrwa"


class ClassificationTask:

    def __init__(self, task_name:TaskName):
        self._task = task_name
        self.backup_name = self.name + "__" + time.strftime("%Y%m%d-%H%M%S") + ".json"

    @property
    def name(self) -> str:
        return self._task.name

    @property
    def task(self) -> TaskName:
        return self._task

    def get_properties(self):

        if self.task == TaskName.IsraelSentimentTweet:
            return schemas.ISRAEL_SENTIMENT_TASK_PROPERTIES
        if self.task == TaskName.IsraelSentimentTweetEurope:
            return schemas.ISRAEL_SENTIMENT_TASK_PROPERTIES
        if self.task == TaskName.IsraelMultiLabelTweet:
            return schemas.MULTI_LABEL_TASK_PROPERTIES
        if self.task == TaskName.HouthisTweet:
            return schemas.HOUTHIS_TASK_PROPERTIES
        if self.task == TaskName.Unrwa:
            return schemas.UNRWA_TASK_PROPERTIES
        else:
            raise Exception("no properties for task: {}".format(self.name))

    def get_error_handler_dict(self):
        return {col: "Azure content filter" for col in self.get_properties()}


    def get_schema(self):
        if self.task == TaskName.IsraelSentimentTweet:
            return schemas.ISRAEL_SENTIMENT_TASK_SCHEME
        if self.task == TaskName.IsraelSentimentTweetEurope:
            return schemas.ISRAEL_SENTIMENT_TASK_SCHEME_EUROPE
        if self.task == TaskName.IsraelMultiLabelTweet:
            return schemas.MULTI_LABEL_TASK_SCHEME
        if self.task == TaskName.HouthisTweet:
            return schemas.HOUTHIS_TASK_SCHEME
        if self.task == TaskName.Unrwa:
            return schemas.UNRWA_TASK_SCHEME
        else:
            raise Exception("no properties for task: {}".format(self.name))

    def get_model(self):
        if self.task == TaskName.IsraelRelatedTweet:
            return 'gpt432k'
        if (self.task in
                [TaskName.IsraelSentimentTweet,
                 TaskName.IsraelMultiLabelTweet,
                 TaskName.IsraelSentimentTweetEurope,
                 TaskName.HouthisTweet,
                 TaskName.Unrwa]):
            return MODEL_NAME
        else:
            raise Exception("no model for task: {}".format(self.name))

    def get_similarity_query_title(self):
        if self.task == TaskName.IsraelSentimentTweet:
            return schemas.ISRAEL_SENTIMENT_TASK_SIMILARITY_QUERY_TITLE
        if self.task == TaskName.HouthisTweet:
            return schemas.HOUTHIS_TASK_SIMILARITY_QUERY_TITLE
        if self.task == TaskName.Unrwa:
            return schemas.UNRWA_TASK_SIMILARITY_QUERY_TITLE
        else:
            raise Exception("no properties for task: {}".format(self.name))

    def get_similarity_query(self):
        if self.task == TaskName.IsraelSentimentTweet:
            return schemas.ISRAEL_SENTIMENT_TASK_SIMILARITY_QUERY
        if self.task == TaskName.HouthisTweet:
            return schemas.HOUTHIS_TASK_SIMILARITY_QUERY
        if self.task == TaskName.Unrwa:
            return schemas.UNRWA_TASK_SIMILARITY_QUERY
        else:
            raise Exception("no properties for task: {}".format(self.name))
    
    def get_keywords(self):
        if self.task == TaskName.HouthisTweet:
            return schemas.HOUTHIS_TASK_KEYWORDS
        else:
            return []
        
    def get_property_col_name(self):
        if self.task == TaskName.IsraelSentimentTweet:
            return schemas.SENTIMENT_PROPERTY
        if self.task == TaskName.HouthisTweet:
            return schemas.HOUTHIS_SENTIMENT
        if self.task == TaskName.Unrwa:
            return schemas.UNRWA_SENTIMENT
        else:
            raise Exception("no property col name for task: {}".format(self.name))

    def get_for_verification_col_name(self):
        if self.task == TaskName.IsraelSentimentTweet:
            return schemas.SENTIMENT_PROPERTY + '_v'
        if self.task == TaskName.HouthisTweet:
            return schemas.HOUTHIS_SENTIMENT + '_v'
        if self.task == TaskName.Unrwa:
            return schemas.UNRWA_SENTIMENT + '_v'
        else:
            raise Exception("no verification col name for task: {}".format(self.name))

class TweetClassifier:
    def __init__(self, cls_task: ClassificationTask):
        self.cls_task = cls_task
        self.llm = LLM.create_llm(self.cls_task.get_model())
        self.chain = self.create_chain()
        self.unpack_columns = self.cls_task.get_properties()
        self.backup_lst = []
        self.cost = 0

    def create_chain(self):
        schema = self.cls_task.get_schema()
        if False: # 'gpt4' in self.cls_task.get_model(): ## function calling doesn't work well in gpt-35-turbo and oss models
            chain = create_tagging_chain(schema, self.llm)
        else:
            # OSS (openchat) models
            class_title = list(schema['properties'].keys())[0]
            prompt_template = schema['properties'][class_title]['description']
            # chain = create_my_openai_chain(prompt_template) # Test: No langchain - it still hangs against vLLM openai server                        
            prompt = ChatPromptTemplate.from_template(prompt_template)
            output_parser = CommaSeparatedListOutputParser()
            chain = LLMChain(
                llm=self.llm,
                prompt=prompt,
                output_parser=output_parser,
            )
        return chain

    def classify_df(self, df, text_column):
        print("start GPT classification", time.ctime())
        tqdm.pandas()
        df[self.unpack_columns] = df[text_column].progress_apply(self.run_and_unpack)
        print("GPT classification ended", time.ctime())
        print("Total cost for gpt classification: {0:.2f} $".format(self.cost))
        return df

    def run_and_unpack(self, post):
        with get_openai_callback() as cb:
          try:
            result_dict = self.chain.run(post)
            self.cost += cb.total_cost
          except Exception as e:
            print(e)
            result_dict = self.cls_task.get_error_handler_dict()
        self.backup(result_dict)
        if False: # 'gpt4' in self.cls_task.get_model():
            try:
                numbers = [result_dict[key] for key in self.unpack_columns]
            except KeyError as ke:
                print(ke)
                result_dict = self.cls_task.get_error_handler_dict()
                numbers = [result_dict[key] for key in self.unpack_columns]
        else:
            if result_dict is None or type(result_dict)!=list or len(result_dict) == 0:                
                numbers = ["Azure content filter"]                    
                print(f'Err: type = {type(result_dict)} result_dict: {result_dict}')
            elif type(result_dict)==list:
                if len(result_dict) == 0:
                    numbers = ["Azure content filter"]
                else:
                    resp = result_dict[0]            
                    lst_str_nums = re.findall(r'(?:ANSWER:\s*)?(\b\d+\b)', resp)
                    numbers = [int(str_num) for str_num in lst_str_nums]

        return pd.Series(numbers)
    
    def backup(self, result_dict):
        self.backup_lst.append(result_dict)
        if len(self.backup_lst) % 50 == 0:
            task_backup_path = os.path.join(GPT_OUTPUTS_BACKUP_PATH, self.cls_task.name, self.cls_task.backup_name)
            with open(task_backup_path, "w") as f:
                f.write(json.dumps(self.backup_lst))
            
            

  


       



