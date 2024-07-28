from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI
from credentials import OPENAI_API_KEY, GPT_OPENAI_API_BASE, MODEL_NAME

class LLM:
    @staticmethod        
    def create_llm(model):
        llm = None
        if 'azure.com' in GPT_OPENAI_API_BASE:
            llm = AzureChatOpenAI(
                openai_api_version="2023-07-01-preview",
                azure_deployment=model,
                openai_api_key=OPENAI_API_KEY,
                azure_endpoint=GPT_OPENAI_API_BASE,
                temperature=0)
        else:
            llm = ChatOpenAI(
                model=model,
                openai_api_key=OPENAI_API_KEY,
                openai_api_base=GPT_OPENAI_API_BASE,
                temperature=0)
        return llm


import openai  
  
class MyOpenAIChain:  
    def __init__(self, openai_api_key, openai_api_base, model, prompt_template):  
        self.client = openai.OpenAI(  
            api_key=openai_api_key,  
            base_url=openai_api_base,  
        )  
        self.model = model  
        self.prompt_template = prompt_template
  
    def run(self, text):  
        messages=[{
                "role": "user",
                "content": self.prompt_template.format(input=text)   
            }]
        
        completion = self.client.chat.completions.create(  
            model=self.model,  
            messages=messages,   
            temperature=0.0  
        )  
        return completion.choices[0].message.content 


def create_my_openai_chain(prompt_template):
    chain = MyOpenAIChain(  
        openai_api_key=OPENAI_API_KEY,   
        openai_api_base=GPT_OPENAI_API_BASE,   
        model=MODEL_NAME,
        prompt_template=prompt_template,
        )  
    return chain