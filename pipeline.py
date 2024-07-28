from pathlib import Path
import json

import IDs
from db_reader import TweeterMongoDB, DataSource
from tweet_classifier import TweetClassifier, ClassificationTask, TaskName
from embeddings import Embedder, SimilarityCalculator
from translator import AzureTranslator, LangugeDetector
import os
import datetime
from paths import RAW_TWITTER_DATA_DIR, EMNED_TWITTER_DATA_DIR, GPT_OUTPUTS_BACKUP_PATH, TWITTER_SIMILARITY_DIR, TWITTER_CLASSIFIED_DIR, \
    TWITTER_CLASSIFICATION_LOGS, TWITTER_CLUSTERING_DIR, REDUCE_EMBED_TWITTER_DATA_DIR
import pandas as pd
import drive_functions as d

from enum import Enum

class UploadTarget(Enum):
    FOR_VER = 0
    AFTER_VER_MERGE = 1

from merge_files import run_whole_process

class TwitterPipeline:
    
    def __init__(self, country, task_name: TaskName, cluster_min_size, clustering=False):
        self.country = country
        # .csv file name of last df fetched from twitter (from-to datetime)
        self.fname_to_classify = None 
        self.similarity_class_col = None
        self.gpt_classification_path = None
        self.task_name = task_name
        self.cls_task = ClassificationTask(task_name=task_name)
        self.text_column = "post_text"
        self.db = TweeterMongoDB(data_source=DataSource.TWITTER, cls_task= self.cls_task)
        self.gpt_classifier = TweetClassifier(cls_task=self.cls_task)
        self.embedder =  Embedder()
        self.translator = self.create_translator()
        if clustering:
            from clustering import Clusterer
            self.clusterer = Clusterer(cluster_min_size)
        self.create_dirs()
        self.translate_to_en_for_similarity_emb = False
    

    def get_query_start_time(self):
        logs = json.load(open(TWITTER_CLASSIFICATION_LOGS, "r"))
        if self.country not in logs or self.cls_task.name not in logs[self.country]:
            return None
        last_run_time = logs[self.country][self.cls_task.name].split("__")[1].split(".")[0]
        return datetime.datetime.strptime(last_run_time, "%Y-%m-%d_%H-%M-%S")

    
    def create_dirs(self):
        # make_embedding_dirs
        if not os.path.exists(EMNED_TWITTER_DATA_DIR):
            os.makedirs(EMNED_TWITTER_DATA_DIR)
        if not os.path.exists(os.path.join(EMNED_TWITTER_DATA_DIR, self.country)):
            os.makedirs(os.path.join(EMNED_TWITTER_DATA_DIR, self.country))
        if not os.path.exists(os.path.join(EMNED_TWITTER_DATA_DIR, self.country,self.cls_task.name)):
            os.makedirs(os.path.join(EMNED_TWITTER_DATA_DIR, self.country,self.cls_task.name))
        
        if not os.path.exists(os.path.join(TWITTER_SIMILARITY_DIR, self.country,self.cls_task.name)):
           os.makedirs(os.path.join(TWITTER_SIMILARITY_DIR, self.country,self.cls_task.name))    
        # gpt classification backup
        if not os.path.exists(GPT_OUTPUTS_BACKUP_PATH):
            os.makedirs(GPT_OUTPUTS_BACKUP_PATH)
        if not os.path.exists(os.path.join(GPT_OUTPUTS_BACKUP_PATH,self.cls_task.name)):
            os.makedirs(os.path.join(GPT_OUTPUTS_BACKUP_PATH,self.cls_task.name))

        if not os.path.exists(os.path.join(TWITTER_CLASSIFIED_DIR, self.country)):
           os.makedirs(os.path.join(TWITTER_CLASSIFIED_DIR, self.country))
           
        if not os.path.exists(os.path.join(TWITTER_CLASSIFIED_DIR, self.country,self.cls_task.name)):
           os.makedirs(os.path.join(TWITTER_CLASSIFIED_DIR, self.country,self.cls_task.name))
        # clustering
        if not os.path.exists(TWITTER_CLUSTERING_DIR):
                os.makedirs(TWITTER_CLUSTERING_DIR)
        if not os.path.exists(os.path.join(TWITTER_CLUSTERING_DIR, self.country)):
              os.makedirs(os.path.join(TWITTER_CLUSTERING_DIR, self.country))
        if not os.path.exists(os.path.join(TWITTER_CLUSTERING_DIR, self.country,self.cls_task.name)):
            os.makedirs(os.path.join(TWITTER_CLUSTERING_DIR, self.country,self.cls_task.name))

        
    def create_translator(self):
        if self.country in ["us", "uk"]:
            return None # no need to translate
        if self.country == "fr":    
            source_lang = "fr"
        elif self.country == "de":
            source_lang = "de"
        else:
            raise Exception("no translation for country: {}".format(self.country))
        return AzureTranslator(self.text_column, source_lang)     
        
    def run_language_detection(self, df):
        if self.country in ["us", "uk"]:
            default_lang = "en"
        else:
            default_lang = self.country
        language_detector = LangugeDetector(default_lang)
        df = language_detector.detetct_language_on_df(df, self.text_column)
        print(df["lang"].value_counts())
        return df
    
    def run_translation(self, df):
        if self.translator is None:
            return df
        df = self.translator.translate_df(df)
        new_fname = self.fname_to_classify.split(".")[0] + '_translated.csv'
        data_path = os.path.join(RAW_TWITTER_DATA_DIR, self.country, new_fname)
        df.to_csv(data_path, index=False)
        self.text_column = self.text_column + "_translated" 
        return df
    
    def run_embeddings(self, df, embedding_path=None,
                       upload_reduced_embeddings = False):
        if embedding_path is None:
            df = self.embedder.embed_df(df, self.text_column)
            embedding_df = pd.concat([df['post_id'], pd.DataFrame(df['embeddings'].tolist())], axis=1)
            embedding_df.iloc[:, 1:] = embedding_df.iloc[:, 1:].round(5)
            embedding_df.to_csv(os.path.join(str(EMNED_TWITTER_DATA_DIR), str(self.country),self.cls_task.name,
                                             str(self.fname_to_classify.split(".")[0]) + '_' +
                                             self.cls_task.name + '_embeddings.csv'), index=False)
            print ("embeddings being saved to disk")

            if upload_reduced_embeddings:
                self.upload_reduce_embeddings(embedding_df)

        else:
            # save embeddings in one column
            embedding_df = pd.read_csv(embedding_path)
            embedding_df['embeddings'] = embedding_df.iloc[:, 1:].apply(lambda row: row.tolist(), axis=1)
            embedding_df = embedding_df[['post_id','embeddings']]
            df = pd.merge(df, embedding_df, on='post_id')
        return df

    def upload_reduce_embeddings(self, embedding_df):
        from clustering import Clusterer
        embedding_df_to_reduce = embedding_df.drop(columns=['post_id'])
        reduced_df = Clusterer.reduce_dimensions(embedding_df_to_reduce)
        reduced_df = pd.concat([embedding_df['post_id'], reduced_df], axis=1)
        reduce_embeddings_path = os.path.join(REDUCE_EMBED_TWITTER_DATA_DIR, self.country,
                                              self.cls_task.name,
                                              self.fname_to_classify.split(".")[0] + '_reduce_embeddings.xlsx')
        reduced_df.to_excel(str(reduce_embeddings_path), index=False)
        print("Reduce embeddings being saved to disk")
        d.upload_to_drive(reduce_embeddings_path, IDs.REDUCE_EMBEDDINGS_FOLDER_ID_DICT[self.country])
        print("Reduce embedding uploaded to drive")

    def calculate_similarity_per_language(self, lang, df):
        query_title = self.cls_task.get_similarity_query_title()
        query = self.cls_task.get_similarity_query()        
        if lang != "en" and self.translate_to_en_for_similarity_emb:
            az_translator = AzureTranslator("", lang)
            query = az_translator.translate(query)
        similarity_calculator = SimilarityCalculator(query_title, query)
        df = similarity_calculator.calculate_similarity_on_df(df, query_title)
        return df


    def run_similarity(self, df):
        df['original_index'] = df.index
        grouped = df.groupby('lang')
        processed_dfs = [self.calculate_similarity_per_language(lang, group) for lang, group in grouped]
        df = pd.concat(processed_dfs)
        df = df.sort_values('original_index')
        df = df.drop(columns=['original_index'])
        save_name = self.fname_to_classify.split(".")[0] + '_' + self.cls_task.name + "_similarity.xlsx"
        sim_path = os.path.join(TWITTER_SIMILARITY_DIR, self.country, self.cls_task.name, save_name)
        df.loc[:, df.columns != 'embeddings'].to_excel(sim_path, index=False)
        print("similarity 0/1 saved to disk")
        return df

    def filter_by_keywords(self,df):
        keywords = self.cls_task.get_keywords()
        def find_keywords(text):
            found_keywords = []
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    found_keywords.append(keyword)
            return found_keywords

        # Create new column with keywords found in post_text
        df['keywords_found'] = df['post_text'].apply(find_keywords)
        save_name = self.fname_to_classify.split(".")[0] + '_' + self.cls_task.name + "_keywords.xlsx"
        sim_path = os.path.join(TWITTER_SIMILARITY_DIR, self.country, self.cls_task.name, save_name)
        df.loc[:, df.columns != 'embeddings'].to_excel(sim_path, index=False)
        print("keywords_found saved to disk")
        return df

    def run_gpt_classification(self, df,filter_similarity:bool = True):
        # If run_similarity determined (thr) which tweet is candidate - only send candidates to GPT
        query_title = self.cls_task.get_similarity_query_title()
        self.similarity_class_col = query_title + "_class_from_similarity"
        if self.similarity_class_col is None or not self.similarity_class_col in df or filter_similarity is False:
            if 'keywords_found' in df:
                df['original_index'] = df.index
                df_similar = df[df.keywords_found.str.len() > 0]
                df_not_similar = df[df.keywords_found.str.len() == 0]
                df_similar = self.gpt_classifier.classify_df(df_similar, self.text_column)
                df = pd.concat([df_similar, df_not_similar])
                df = df.sort_values('original_index')
                df = df.drop(columns=['original_index'])
            else:    
                df = self.gpt_classifier.classify_df(df, self.text_column)
        else:
            df['original_index'] = df.index
            df_similar = df[df[self.similarity_class_col] == 1]
            df_not_similar = df[df[self.similarity_class_col] == 0]
            df_similar = self.gpt_classifier.classify_df(df_similar, self.text_column)
            df = pd.concat([df_similar, df_not_similar])
            df = df.sort_values('original_index')
            df = df.drop(columns=['original_index'])
        save_name = self.fname_to_classify.split(".")[0] + '_' + self.cls_task.name + "_classified.xlsx"
        self.gpt_classification_path = os.path.join(TWITTER_CLASSIFIED_DIR, self.country, self.cls_task.name, save_name)
        df.loc[:, df.columns != 'embeddings'].to_excel(self.gpt_classification_path, index=False)
        print("GPT classification being saved to disk")
        return df

    def upload_gpt_classification(self, gpt_classification_path=None, upload_gpt_classification_to = UploadTarget.FOR_VER):
        gpt_classification_path = self.gpt_classification_path if gpt_classification_path \
                                                                  is None else gpt_classification_path

        assert gpt_classification_path is not None, "GPT classification path is missing, to upload it to the drive"

        df = pd.read_excel(gpt_classification_path)

        property_col_name = self.cls_task.get_property_col_name()
        assert property_col_name in df.columns, \
            "GPT classification should run before uploading file"

        for_verification_col_name = self.cls_task.get_for_verification_col_name()
        if for_verification_col_name not in df.columns:
            df[for_verification_col_name] = ""
            df.to_excel(gpt_classification_path, index=False)
            print("Added 'for_verification' column to GPT classification file, and saving it to disk")

        # upload file to google drive
        if upload_gpt_classification_to == UploadTarget.FOR_VER:
            target_folder_id_dict = IDs.FOR_VER_FOLDER_ID_DICT 
        elif upload_gpt_classification_to == UploadTarget.AFTER_VER_MERGE:
            target_folder_id_dict = IDs.AFTER_VER_FOLDER_ID_DICT
        else:
            raise Exception('upload_gpt_classification_to must be one of the supported enum values {UploadTarget}')
            
        upload_folder_id = IDs.get_folder_id(self.country, self.cls_task.task, target_folder_id_dict)
        d.upload_to_drive(gpt_classification_path, upload_folder_id)
        print("GPT classification uploaded to drive")
        # Merge after upload
        if upload_gpt_classification_to == UploadTarget.AFTER_VER_MERGE:
            run_whole_process(task_name=self.task_name, country=self.country,merge_from_folder=0)

    def update_log_file(self):
        if not os.path.exists(TWITTER_CLASSIFIED_DIR):
          log = {}
        else:
          with open(TWITTER_CLASSIFICATION_LOGS, "r") as f:
            log = json.load(f)
        if self.country not in log:
          log[self.country] = {}
          log[self.country][self.cls_task.name] = self.fname_to_classify.split(".")[0].replace('_translated', '')
        else:
            if self.cls_task.name not in log[self.country]:
              log[self.country][self.cls_task.name] = self.fname_to_classify.split(".")[0].replace('_translated', '')
            else:
              start_time = log[self.country][self.cls_task.name].split("__")[0]
              end_time = self.fname_to_classify.split("__")[1].split(".")[0].replace('_translated', '')
              log[self.country][self.cls_task.name] = start_time + "__" + end_time
        with open(TWITTER_CLASSIFICATION_LOGS, "w") as f:
          json.dump(log, f, indent=4)

    def run_clustering(self, df,random_state = None):
        """
        df - post_id, text ...+ embeddings 
        dedup by text to remove retweets -- but later add the cluster of the representative tweet to all dups
        
        """
        assert "embeddings" in df.columns, "embeddings column is missing"
        df['original_index'] = df.index
        duplicates = df.duplicated(subset=self.text_column, keep='first')
        df_unique = df[~duplicates]
        print ("number of unique tweets: {}".format(df_unique.shape[0]))
        embedding_df_unique = pd.DataFrame(df_unique['embeddings'].tolist())
        # Cluster 
        df_clustered = self.clusterer.cluster(df = df_unique,
                                              embedding_df=embedding_df_unique,random_state=random_state)
        df_clustered = df_clustered.drop(columns=['embeddings'])
        save_name = self.fname_to_classify.split(".")[0] + "_" + self.cls_task.name + "_clustered_wo_labels.csv"
        df_clustered.to_csv(os.path.join(TWITTER_CLUSTERING_DIR, self.country, self.cls_task.name,save_name), index=False)
        df_clustered = self.clusterer.give_cluster_labels(df_clustered, self.text_column,random_state)
        duplicates_df = df[duplicates]
        duplicates_df["cluster_idx"] = ""
        duplicates_df["cluster_label"] = ""
        duplicates_df = duplicates_df.drop(columns=['embeddings'])
        for index, row in duplicates_df.iterrows():
            original_row = df_clustered[df_clustered[self.text_column] == row[self.text_column]].iloc[0]
            for col in ["cluster_idx", "cluster_label"]:
                duplicates_df.at[index, col] = original_row[col]
        df = pd.concat([df_clustered, duplicates_df])
        df = df.sort_values('original_index')
        df = df.drop(columns=['original_index'])
        save_name = self.fname_to_classify.split(".")[0] + "_" + self.cls_task.name + "_clustered_with_labels.csv"
        df.to_csv(os.path.join(str(TWITTER_CLUSTERING_DIR), self.country,self.cls_task.name, save_name), index=False)
        print("Clustering saved to disk")
        return df


    def get_data(self, data_path,x_days_before: int = 0):        
        if data_path is None:
            df = self.db.query_data_from_last_time(country_or_lang=self.country,
                                                   start_time=self.get_query_start_time(),
                                                   x_days_before = x_days_before)

            self.fname_to_classify = self.db.get_last_table_name(self.country)
        else:
            p_data_path = Path(data_path)
            if p_data_path.suffix.lower() == '.xlsx':
                df = pd.read_excel(p_data_path)                            
            elif p_data_path.suffix.lower() == '.csv':
                df = pd.read_csv(p_data_path)                
            else:
                raise Exception("Unsupported data file format.")
            self.fname_to_classify = data_path.split("\\")[-1]
        return df
            
    def run(self, data_path=None, embedding_path=None, gpt_classification_path = None, run_language_detection=True, run_embedding=True,
            run_similarity=True, run_filter_by_keywords=True, run_gpt=True, run_clustering=True, upload_gpt_classification_to=UploadTarget.FOR_VER,
            random_state = None,x_days_before: int = 0,filter_similarity:bool = True):
        df = self.get_data(data_path=data_path,
                           x_days_before=x_days_before)
        if df.shape[0] == 0:
            print ("no data to classify")
            return  
        # df = self.run_translation(df)
        if run_language_detection:
            df = self.run_language_detection(df)
        if run_embedding:
            df = self.run_embeddings(df, embedding_path)
        if run_similarity:
            df = self.run_similarity(df)
        if run_filter_by_keywords:
            df = self.filter_by_keywords(df)
            
        if run_gpt:
            df = self.run_gpt_classification(df,filter_similarity)
            self.update_log_file()
        if upload_gpt_classification_to != None:
            self.upload_gpt_classification(gpt_classification_path, upload_gpt_classification_to)

        if run_clustering:
            df = self.run_clustering(df,random_state)
