import numpy as np
import pandas as pd
import os
from paths import NEWS_DIR, SUMMARIZED_NEWS_DIR, EMNED_NEWS_DATA_DIR, NEWS_CLUSTERING_DIR
from db_reader import TweeterMongoDB, DataSource
from embeddings import Embedder, EMBED_DIM
from summarization import ArticleSummarizer
from clustering import Clusterer
import warnings
warnings.filterwarnings('ignore')

class NewsPipeline:

    def __init__(self, cluster_min_size):
        self.fname_to_classify = None
        self.similarity_class_col = None
        self.text_column = "content"
        self.db = TweeterMongoDB(DataSource.NEWS)
        self.embedder = Embedder()
        self.clusterer = Clusterer(cluster_min_size, data_type="news")
        self.create_dirs()

    def create_dirs(self):
        # make_embedding_dirs
        if not os.path.exists(NEWS_DIR):
            os.makedirs(NEWS_DIR)
        if not os.path.exists(SUMMARIZED_NEWS_DIR):
            os.makedirs(SUMMARIZED_NEWS_DIR)
        if not os.path.exists(EMNED_NEWS_DATA_DIR):
            os.makedirs(EMNED_NEWS_DATA_DIR)
        if not os.path.exists(NEWS_CLUSTERING_DIR):
            os.makedirs(NEWS_CLUSTERING_DIR)

    def run_embeddings(self, df, embedding_path=None):
        if embedding_path is None:
            df['original_index'] = df.index
            non_relevant_df = df[df['summary'] == '0']
            relevant_df = df[df['summary'] != '0']
            relevant_df = self.embedder.embed_df(relevant_df, 'summary')
            embeddings_placeholder = np.zeros((len(non_relevant_df), EMBED_DIM)).tolist()
            non_relevant_df["embeddings"] = embeddings_placeholder
            df = pd.concat([non_relevant_df, relevant_df])
            df = df.sort_values('original_index')
            df = df.drop(columns=['original_index'])
            emnbeding_df = pd.DataFrame(df['embeddings'].tolist())
            emnbeding_df.to_csv(os.path.join(EMNED_NEWS_DATA_DIR, self.fname_to_classify.split(".")[0] + '_embeddings.csv'), index=False)
            print ("embeddings being saved to disk")
        else:
            emnbeding_df = pd.read_csv(embedding_path)
            df['embeddings'] = emnbeding_df.values.tolist()
        return df    

    def get_query_start_time(self):
        logs = json.load(open(TWITTER_CLASSIFICATION_LOGS, "r"))
        if self.country not in logs or self.task.name not in logs[self.country]:
            return None
        last_run_time = logs[self.country][self.task.name].split("__")[1].split(".")[0]
        return datetime.datetime.strptime(last_run_time, "%Y-%m-%d_%H-%M-%S")

    def relevancy_keyword_filter(self, df):
        keywards = [ "israel", "israeli", "hamas", "jewish", "gaza", "palestinian", "palestine"]
        df['war_relevancy_by_kw'] = df[self.text_column].apply(lambda x: any(word in str(x).lower() for word in keywards)).astype(int)
        print ("after keywards filtering: number of relevant articles: {} out of {} articles".format(df['war_relevancy_by_kw'].sum(), len(df)))
        return df

    def run_summarization(self, df):
        df['original_index'] = df.index
        non_relevant_df = df[df['war_relevancy_by_kw'] == 0]
        non_relevant_df["summary"] = '0'
        relevant_df = df[df['war_relevancy_by_kw'] == 1]
        summarizer = ArticleSummarizer()
        relevant_df = summarizer.summarize_df(relevant_df, self.text_column)
        df = pd.concat([non_relevant_df, relevant_df])
        df = df.sort_values('original_index')
        df = df.drop(columns=['original_index'])
        df.to_csv(os.path.join(SUMMARIZED_NEWS_DIR, self.fname_to_classify), index=False)
        return df
    
    def run_clustering(self, df):
        assert "embeddings" in df.columns, "embeddings column is missing"
        df['original_index'] = df.index
        non_relevant_df = df[df['summary'] == '0']
        non_relevant_df["cluster_idx"] = '-2'
        non_relevant_df["cluster_label"] = 'not_israel_related'
        relevant_df = df[df['summary'] != '0']
        embedding_df = pd.DataFrame(relevant_df['embeddings'].tolist())
        df_clustered = self.clusterer.cluster(relevant_df, embedding_df)
        df_clustered = df_clustered.drop(columns=['embeddings'])
        save_name = self.fname_to_classify.split(".")[0] + "_clustered_wo_labels.csv"
        df_clustered.to_csv(os.path.join(NEWS_CLUSTERING_DIR, save_name), index=False)        
        df_clustered = self.clusterer.give_cluster_labels(df_clustered, 'summary')
        df = pd.concat([df_clustered, non_relevant_df])
        df = df.sort_values('original_index')
        df = df.drop(columns=['original_index'])
        save_name = self.fname_to_classify.split(".")[0] + "_clustered_with_labels.csv"
        df.to_csv(os.path.join(NEWS_CLUSTERING_DIR, save_name), index=False)
        print("Clustering saved to disk")
        return df

    def get_data(self, data_path):
        if data_path is None:
            df = self.db.query_data_from_last_time(self.get_query_start_time())
            self.fname_to_classify = self.db.get_last_table_name(self.country)
        else:
            df = pd.read_csv(data_path)
            self.fname_to_classify = data_path.split("\\")[-1]
        return df

    def run(self, data_path=None, embedding_path=None, run_summarization=True, run_embedding=True, run_clustering=True):
        df = self.get_data(data_path)
        if df.shape[0] == 0:
            print ("no data")
            return  
        # df = self.run_translation(df)
        if run_summarization:
            df = self.relevancy_keyword_filter(df)
            df = self.run_summarization(df)
        if run_embedding:
            df = self.run_embeddings(df, embedding_path)
        if run_clustering:
            df = self.run_clustering(df)


pipe = NewsPipeline(30)
data_path = r'C:\Users\yaniv\Desktop\war\twitter\Tweet_Classifier\data\news_data\summarized\temp.csv'
embedding_path = r'C:\Users\yaniv\Desktop\war\twitter\Tweet_Classifier\data\news_data\embeddings\temp_embeddings.csv'
pipe.run(data_path, embedding_path=None, run_summarization=False, run_embedding=True, run_clustering=True)
