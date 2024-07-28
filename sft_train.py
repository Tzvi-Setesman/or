# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 20:31:27 2023

@author: DEKELCO
"""
# cd /d D:\NLP\Netivot\experiments

import platform
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments,BitsAndBytesConfig
from datasets import load_metric, Dataset, DatasetDict, load_from_disk
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils import find_all_linear_names, print_trainable_parameters
import wandb

# This is the accumulated labeled data to create the dataset from
PATH_ALL_DATA = './data/all_data_us__updated_to_2023-12-28__10_56.xlsx'
output_dir="./outputs"
model_name ="openchat/openchat-3.5-1210"
LOGGER_RUN_GROUP_NAME = 'tweet-sentiment-llm'
LOGGER_RUN_NAME = 'openchat_3_5_data_ver_1'
# to allow code to run on local windows dev without reporting to wandb server 
LOGGER_MODE = 'dryrun' if  platform.system() == 'Windows' else 'online' 

DO_TRAIN = True
DO_EVAL = False # Not supported - fix Local LLM OpenAI
DO_PREDICT = False # Not supported - fix Local LLM OpenAI

wandb.login()
# start a new wandb run to track this script
wandb.init(project=LOGGER_RUN_GROUP_NAME, name = LOGGER_RUN_NAME, mode=LOGGER_MODE)

data_path = Path('./data')

max_text_length = 500 # Max chars in text


def create_dataset_train_val(df, random_state=42, test_size=700, 
                             max_text_length=500, train_size=-1, 
                             dataset_name = 'tweet_sentiment_us', data_folder = data_path):    
    if 'Unnamed: 0' in df.columns:  
        df = df.drop(columns=['Unnamed: 0'])  
      
    df = df[df.text.str.len() <= max_text_length]  
    # Shuffle (data loader should also shuffle, but not sure)
    df = df.sample(frac=1,random_state=random_state)      
    # Split the dataset into a train set and a validation set (small amount)  
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)  
    
    if train_size > 0:
      train_df = train_df.iloc[:train_size]  
      
    train_dataset = Dataset.from_pandas(train_df)  
    train_dataset = train_dataset.remove_columns(['__index_level_0__'])  # Remove '__index_level_0__' feature from the datasets  
    val_dataset = Dataset.from_pandas(val_df)    
    val_dataset = val_dataset.remove_columns(['__index_level_0__'])  
      
    split_datasets = DatasetDict({  
        'train' : train_dataset,  
        'validation' : val_dataset,  
        })  
      
    
    data_folder = Path(data_folder)
    train_df.to_parquet(data_folder / 'train.parquet')  
    val_df.to_parquet(data_folder / 'validation.parquet')  
    split_datasets.save_to_disk(data_folder / dataset_name) 
    return split_datasets  

def create_train_df_from_all_data(path_all_data):
    df = pd.read_excel(path_all_data)
    df = df.rename(columns={'post_text' : 'text', 'conflict_sentiment_v' : 'labels'})
    df = df[df.upload_date >= '2023-10-20']
    wandb.log({'labels of training df - examine imbalance' : df.labels.value_counts().to_dict()}) 
    # Tweets where embs incorrectly classify as israel_related=1, but gpt or human classified 0
    df_gpt_not_related = df[df.israel_related_class_from_similarity == 1 & (df.labels == 0)]
    # tweets where emb classify as not_israel_related (with high confidence) - they did not reach GPT 4 
    df_emb_sure_not_related = df[df.israel_related_similarity < 0.695]
    df_class_0 = pd.concat([df_gpt_not_related, df_emb_sure_not_related],axis=0).sample(n=3500)
    df_class_9 = df[(df.conflict_sentiment == 9) & (df.labels == 9)].sample(n=1500)
    df_class_3 = df[df.labels == 3].sample(n=2500)
    df_class_1_2 = df[df.labels.isin([1,2])]
    df_train_val = pd.concat([df_class_0, df_class_9, df_class_3, df_class_1_2],axis=0)
    wandb.log({'labels of training df - examine imbalance' : df_train_val.labels.value_counts().to_dict()})
    return df_train_val[['text','labels']]

def create_small_df_from_all_data(path_all_data, n_sample_0=500, n_sample_others=500, classified_by_model='openchat'):
     
    """
    Create small eval dataset to be labeled by both gpt-4 and openchat (OSS)
    ==Usage==
    df = create_small_df_from_all_data('./data/twitter_data/gpt/us/2024-02-04_17-38-52__2024-02-08_18-12-08_IsraelSentimentTweet_classified.openchat.xlsx')
    df = create_small_df_from_all_data('./data/twitter_data/gpt/us/new_old_prompt_gpt_openchat_openchat_IsraelSentimentTweet_classified.xlsx', classified_by_model='openchat_0106')
    df.to_csv('./data/twitter_data/small_df.csv', index=False)
    
    Parameters
    ----------
    path_all_data : TYPE
        DESCRIPTION.

    Returns
    -------
    df with data - same fields that return from twitter-api + openchat classification

    """
    
    df = pd.read_excel(path_all_data)    
    print('origi labels of training df - examine imbalance', df.conflict_sentiment.value_counts().to_dict())
    # Tweets where embs incorrectly classify as israel_related=1, but gpt or human classified 0
    # df_gpt_not_related = df[df.israel_related_class_from_similarity == 1 & (df.labels == 0)]
    # tweets where emb classify as not_israel_related (with high confidence) - they did not reach GPT 4 
    #df_emb_sure_not_related = df[df.israel_related_similarity < 0.695]
    df_class_0 = df[df.conflict_sentiment == 0].sample(n=n_sample_0)
    df_class_9 = df[(df.conflict_sentiment == 9)].sample(n=int(n_sample_others / 3))
    df_class_1_2_3 = df[df.conflict_sentiment.isin([1,2,3])]#.sample(n=int(n_sample_others / 2))
    

    df_train_val = pd.concat([df_class_0, df_class_9, df_class_1_2_3],axis=0)
    print('After sampling- examine imbalance', df_train_val.conflict_sentiment.value_counts().to_dict())
    df_train_val = df_train_val.rename(columns={'conflict_sentiment' : f'conflict_sentiment_{classified_by_model}'})
    df_train_val = df_train_val.drop(columns=['conflict_sentiment_v'])    
    return df_train_val

        
df_train_val = create_train_df_from_all_data(PATH_ALL_DATA)
dataset = create_dataset_train_val(df_train_val)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, quantization_config=bnb_config, attn_implementation="flash_attention_2")
base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(base_model)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

# Change the LORA hyperparameters accordingly to fit your use case
peft_config = LoraConfig(
    r=128,
    lora_alpha=128,
    target_modules=find_all_linear_names(base_model),
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

base_model = get_peft_model(base_model, peft_config)
print_trainable_parameters(base_model)

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['text'])):
        messages = [
            {"role": "user", "content": f"Task: Classify tweets related to the Gaza-Israel war into one of the following categories. Class 1: Condemn Israeli attacks. Class 2: Expressing a balanced attitude towards Israel and Palestines. Class 3: Support Israel or condemn Hamas as terrorists. Class 0: Unrelated to the Gaza conflict. Class 9: Relating to the conflict between Israel and Gaza, but focus on internal American issues and politics without taking a clear side. Tweet: {example['text'][i]}"},
            {"role": "assistant", "content": f"{example['labels'][i]}"},
        ] 
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        
        output_texts.append(text)
    
    return output_texts

# Parameters for training arguments details => https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py#L158
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing =True,
    max_grad_norm= 0.3,
    num_train_epochs=3, 
    learning_rate=2e-4,
    bf16=True,
    save_total_limit=3,
    logging_steps=500,
    output_dir=output_dir,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    report_to="wandb",
)

instruction_template = "GPT4 Correct User:"
response_template = "GPT4 Correct Assistant:"
collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)

trainer = SFTTrainer(
    base_model,
    train_dataset=dataset['train'],
    tokenizer=tokenizer,
    max_seq_length=2048,
    formatting_func=formatting_prompts_func,
    data_collator=collator, 
    args=training_args
    
)

if DO_TRAIN:
    trainer.train() 
    trainer.save_model(output_dir)

    output_dir = Path(output_dir) / "final_checkpoint"
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

wandb.finish()