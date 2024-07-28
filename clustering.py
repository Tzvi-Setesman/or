import embeddings
import hdbscan
import umap.umap_ as umap
import numpy as np
from langchain.schema import HumanMessage
import tiktoken
from llm import LLM
import pandas as pd


class Clusterer:
    """
    Clusters texts using umap (reduce_dimensions) -> HDBSCAN --> label clusters with gpt4
    """
    def __init__(self, cluster_min_size, data_type="Twitter"):
        self.sampels_for_gpt = 5
        self.embedder = embeddings.Embedder()
        self.gpt4 = LLM.create_llm('gpt4')
        self.gpt432k = LLM.create_llm('gpt432k') 
        self.current_llm = self.gpt4
        self.cluster_min_size = cluster_min_size
        self.data_type = data_type
        
    def choose_llm(self, text):
        encoding = tiktoken.encoding_for_model('gpt-4')
        num_tokens = len(encoding.encode(text))
        if num_tokens < 8000:
            print ("use gpt4 for naming clusters")
            self.current_llm =  self.gpt4
        else:
            print ("use gpt4-32k for naming clusters")
            self.current_llm =  self.gpt432k

    @staticmethod
    def reduce_dimensions(embedding_df,random_state=None):
        n_components = 5
        umap_model = umap.UMAP(n_neighbors=15,
                        n_components=n_components,
                        min_dist=0.0,
                        metric='cosine',
                        low_memory=True,
                        random_state=random_state)
        umap_model.fit(embedding_df)
        umap_embeddings = umap_model.transform(embedding_df)
        umap_embeddings = np.nan_to_num(umap_embeddings)
        reduced_df = pd.DataFrame(umap_embeddings, columns=[f'dim_{i+1}' for i in range(n_components)])
        return reduced_df


    def cluster(self, df, embedding_df, reduce_dimensions = True,random_state = None):
        print ("start preform clustering")
        if reduce_dimensions:
            cluster_df  = Clusterer.reduce_dimensions(embedding_df,random_state=random_state)
            print("reduce dimensions was finished")
        else:
            cluster_df = embedding_df
            print("reduce dimensions was not needed")


        hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=self.cluster_min_size,
                                    metric='euclidean',
                                    cluster_selection_method='eom',
                                    prediction_data=True
                                        )
        labels = hdbscan_model.fit_predict(cluster_df)
        df['cluster_idx'] = labels
        print ("finished clustering #labels",len(labels))
        return df

    def give_cluster_labels(self, df, text_col,random_state=None):
        print ("start giving cluster labels")
        cluster_samples = self.get_cluster_samples(df=df, text_col=text_col,random_state=random_state)
        self.labels_dict = self.get_cluster_labels(cluster_samples)
        df['cluster_label']  = df['cluster_idx'].apply(lambda x: self.labels_dict[x])
        print ("finished giving cluster labels. Cluster labels distribution:" )
        print(df['cluster_label'].value_counts())
        return df

    def get_cluster_samples(self, df, text_col,random_state = None):
        cluster_samples = {}
        for cluster in df['cluster_idx'].unique():
            cluster_data = df[df['cluster_idx'] == cluster][text_col]
            num_samples = min(len(cluster_data), self.sampels_for_gpt)  
            cluster_samples[cluster] = cluster_data.sample(num_samples, replace=False,
                                                           random_state=random_state).values.tolist()
        cluster_samples = {key: cluster_samples[key] for key in sorted(cluster_samples)}
        return cluster_samples
    
    def get_prompt(self, samples):
        twitter_prompt = "I have conducted a machine learning-based analysis of Twitter posts, focusing on the sentiment related to the Israel-Gaza conflict and global antisemitism. \
                        The following sampels was drawn from the same cluster. Review the samples and assign a descriptive name to the cluster. \n Posts: \n"
        news_prompt = "I have conducted a machine learning-based analysis of news articles, focusing on the sentiment related to the Israel-Gaza conflict and global antisemitism. \
            The following samples were drawn from the same cluster. Review the samples and assign a descriptive name to the cluster. \n Articles: \n"
        if self.data_type == "Twitter":
            prompt = twitter_prompt
        elif self.data_type == "news":
            prompt = news_prompt
        else:
            raise Exception("data type is not supported")
        for i, text in enumerate(samples, start=1):
            prompt += f"Post{i}: {text}\n"
        prompt += "Cluster Name: "
        return prompt
    
    def get_cluster_labels(self, cluster_samples):
        labels = {}
        for cluster_id, samples in cluster_samples.items():
            if cluster_id == -1:
                labels[cluster_id] = "outlier"
                continue
            prompt = self.get_prompt(samples)
            message = HumanMessage(content=prompt)
            try:
                label_name = self.current_llm([message]).content
                label_name = label_name.replace('"', '')
            except Exception as e:
                label_name = "Azure content filter"
                print("the following smaples didn't get a label due to Azure content filter: {}".format(samples))
            labels[cluster_id] = label_name
        return labels
        
