import os
from datetime import datetime
from pathlib import Path
from unittest import TestCase

import pandas as pd

from clustering_pipeline.clustering_pipeline import ClusteringPipeline
from paths import EMNED_TWITTER_DATA_DIR, TWITTER_CLUSTERING_DIR
from pipeline import TwitterPipeline
from tweet_classifier import TaskName, ClassificationTask


class ClusteringTestCase(TestCase):
    task_name = None
    cluster_path = None
    embedding_path = None
    embedding_df_path = None
    country = None
    clustered_with_labels_path = Path(
        r'data/2023-12-28_11-39-00__2023-12-31_16-12-36_clustered_with_labels.csv')

    df_path = r"data/2023-12-28_11-39-00__2023-12-31_16-12-36.csv"

    @classmethod
    def setUpClass(cls):
        cls.country = 'us'
        cls.cluster_min_size = 50
        cls.random_state = 123

        cls.task_name = TaskName.IsraelSentimentTweet
        cls.cls_task = ClassificationTask(cls.task_name)

        cls.embedding_path = os.path.join(EMNED_TWITTER_DATA_DIR, cls.country)
        cls.cluster_path = os.path.join(TWITTER_CLUSTERING_DIR, cls.country)
        
        cls.embedding_file = os.path.join(
            str(cls.embedding_path),
            r"data/2023-12-28_11-39-00__2023-12-31_16-12-36_embeddings.csv")
        cls.twitter_pipeline_cluster_file = os.path.join(
            str(cls.cluster_path),
            r"data/2023-12-28_11-39-00__2023-12-31_16-12-36_clustered_with_labels.csv")

    @classmethod
    def tearDownClass(cls):
        pass

    def test_twitter_pipeline_create_embedding(self):
        pipe = TwitterPipeline(country=self.country, task_name=self.task_name,
                               cluster_min_size=self.cluster_min_size, clustering=False)

        pipe.run(self.df_path,
                 embedding_path=None,
                 run_language_detection=True,
                 run_embedding=True,
                 run_similarity=False,
                 run_gpt=False,
                 run_clustering=False,
                 run_gpt_classification_upload=False)

    def test_twitter_pipeline_cluster_with_embeddings(self):
        pipe = TwitterPipeline(country=self.country, task_name=self.task_name,
                               cluster_min_size=self.cluster_min_size, clustering=True)

        pipe.run(self.df_path,
                 embedding_path=self.embedding_file,
                 run_language_detection=True,
                 run_embedding=True,
                 run_similarity=False,
                 run_gpt=False,
                 run_clustering=True,
                 run_gpt_classification_upload=False,
                 random_state=self.random_state)

    def test_compare_twitter_pipline_and_cluster_pipeline(self):
        df_twitter_pipeline_clustered_ = pd.read_csv(self.twitter_pipeline_cluster_file)
        print("twitter_pipline clustered", df_twitter_pipeline_clustered_.shape,
              df_twitter_pipeline_clustered_['cluster_label'].value_counts())

        start_date = datetime.strptime("2023-12-28_11-39-00", "%Y-%m-%d_%H-%M-%S")  # =None
        end_date = datetime.strptime("2023-12-31_16-13-59", "%Y-%m-%d_%H-%M-%S")  # =None

        embedding_path = os.path.join(self.embedding_path,"data")

        pipe = ClusteringPipeline(country=self.country, task_name=self.task_name, cluster_min_size=self.cluster_min_size)
        df_clustered = pipe.run(start_date=start_date, end_date=end_date, embedding_path=embedding_path,
                                reduce_dimensions=True,
                                only_conflict_related=False,
                                random_state=self.random_state)

        print("cluster_pipeline clustered", df_clustered.shape, df_clustered['cluster_label'].value_counts())

        value_counts_df1 = df_clustered['cluster_label'].value_counts()
        value_counts_df2 = df_twitter_pipeline_clustered_['cluster_label'].value_counts()

        self.assertEqual(value_counts_df1.tolist(),value_counts_df2.tolist())


    def test_compare_to_old_code_no_filter_conflict(self):
        start_date = datetime.strptime("2023-12-28_11-39-00", "%Y-%m-%d_%H-%M-%S")  # =None
        end_date = datetime.strptime("2023-12-31_16-12-36", "%Y-%m-%d_%H-%M-%S")  # =None

        pipe = ClusteringPipeline(country=self.country, task_name=self.task_name,cluster_min_size=self.cluster_min_size)
        df_clustered = pipe.run(start_date=start_date, end_date=end_date, only_conflict_related=False)

        df_clustered_old_code = pd.read_csv(self.clustered_with_labels_path)

        print("df_clustered", df_clustered.shape, df_clustered['cluster_label'].value_counts())
        print("df_clustered_old_code", df_clustered_old_code.shape,
              df_clustered_old_code['cluster_label'].value_counts())

        column_values_equal = df_clustered['post_text'].equals(df_clustered_old_code['post_text'])
        self.assertTrue(column_values_equal)
        print("column_values_equal", column_values_equal)

        self.assertEqual(len(df_clustered['cluster_label'].value_counts()),
                         len(df_clustered_old_code['cluster_label'].value_counts()))

    def test_compare_to_old_code_with_filter_conflict(self):
        start_date = datetime.strptime("2023-12-28_11-39-00", "%Y-%m-%d_%H-%M-%S")  # =None
        end_date = datetime.strptime("2023-12-31_16-12-36", "%Y-%m-%d_%H-%M-%S")  # =None

        pipe = ClusteringPipeline(country=self.country, task_name=self.task_name,
                                  cluster_min_size=self.cluster_min_size)
        df_clustered = pipe.run(start_date=start_date, end_date=end_date, only_conflict_related=True)

        print("df_clustered", df_clustered.shape, df_clustered['cluster_label'].value_counts())

        allowed_values = [1, 2, 3, 9]
        is_valid = df_clustered[self.cls_task.get_for_verification_col_name()].isin(allowed_values).all()
        self.assertTrue(is_valid)
