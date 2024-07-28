import os
from datetime import datetime

import pandas as pd
from pandas import DataFrame

from clustering import Clusterer
from drive_functions_wrapper import download_all_data, list_reduce_embedding, download_file_from_drive
from paths import TWITTER_CLUSTERING_DIR
from tweet_classifier import TaskName, ClassificationTask


class ClusteringPipeline:
    def __init__(self, country, task_name: TaskName, cluster_min_size):

        self.country = country
        self.cls_task = ClassificationTask(task_name=task_name)
        self.clusterer = Clusterer(cluster_min_size)
        self.text_column = "post_text"
        self.file_name = None

    def run(self, start_date, end_date,
            embedding_path = None,
            only_conflict_related = False,
            reduce_dimensions=True,
            random_state=None) -> DataFrame:

        self.set_fn_prefix(end_date, start_date)

        df = self.prepare_df(start_date, end_date, embedding_path)

        df = self.filter(df, start_date, end_date, only_conflict_related)

        df_unique, duplicates = self.remove_duplicates(df)

        embedding_df_unique = pd.DataFrame(df_unique['embeddings'].tolist())

        df_clustered = self.run_clustering(df_unique,
                                           embedding_df_unique,
                                           reduce_dimensions,
                                           random_state)

        df_clustered = self.save_no_labels(df_clustered)

        df_clustered = self.clusterer.give_cluster_labels(df=df_clustered,
                                                          text_col=self.text_column,
                                                          random_state=random_state)

        df = self.returned_duplicates(df, df_clustered, duplicates)

        df = self.save_with_labels(df)

        return df

    def prepare_df(self, start_date, end_date, embedding_path=None):
        embedding_df = self.load_embedding(start_date, end_date, embedding_path)
        print("embedding_df", embedding_df.shape)
        df = self.load_all_data()

        print("download_all_data", df.shape)
        df = pd.merge(df, embedding_df, on='post_id')
        print("merge", df.shape)

        return df

    def set_fn_prefix(self, end_date, start_date):
        start_date_str = start_date.strftime("%Y-%m-%d_%H-%M-%S")
        end_date_str = end_date.strftime("%Y-%m-%d_%H-%M-%S")
        self.file_name = f"{start_date_str}__{end_date_str}"

    def save_no_labels(self, df_clustered) -> DataFrame:
        df_clustered = df_clustered.drop(columns=['embeddings'])
        save_name = self.file_name + "_clustered_wo_labels.csv"
        df_clustered.to_csv(os.path.join(TWITTER_CLUSTERING_DIR, self.country, self.cls_task.name, save_name),
                            index=False)
        return df_clustered

    def save_with_labels(self, df) -> DataFrame:
        df = df.sort_values('original_index')
        df = df.drop(columns=['original_index'])
        save_name = self.file_name + "_clustered_with_labels.csv"
        df.to_csv(os.path.join(TWITTER_CLUSTERING_DIR, self.country, self.cls_task.name, save_name), index=False)
        print("Clustering saved to disk")
        return df

    def returned_duplicates(self, df, df_clustered, duplicates) -> DataFrame:
        duplicates_df = df[duplicates].copy()
        duplicates_df.loc[:, "cluster_idx"] = ""
        duplicates_df.loc[:, "cluster_label"] = ""
        duplicates_df = duplicates_df.drop(columns=['embeddings'])
        for index, row in duplicates_df.iterrows():
            original_row = df_clustered[df_clustered[self.text_column] == row[self.text_column]].iloc[0]
            for col in ["cluster_idx", "cluster_label"]:
                duplicates_df.at[index, col] = original_row[col]
        df = pd.concat([df_clustered, duplicates_df])
        return df

    def run_clustering(self, df_unique, embedding_df_unique,
                       reduce_dimensions, random_state) -> DataFrame:
        df_clustered = self.clusterer.cluster(df=df_unique,
                                              embedding_df=embedding_df_unique,
                                              reduce_dimensions=reduce_dimensions,
                                              random_state=random_state)

        return df_clustered

    def remove_duplicates(self, df):
        df['original_index'] = df.index
        duplicates = df.duplicated(subset=self.text_column, keep='first')
        df_unique = df[~duplicates]
        print("number of unique tweets: {}".format(df_unique.shape[0]))
        return df_unique, duplicates

    def load_all_data(self):
        df = download_all_data(self.country,self.cls_task.task, credentials_path="..")
        df['collection_time'] = df.apply(self.date_format, axis=1)
        return df

    def load_embedding(self, start_date, end_date, embedding_path=None) -> DataFrame:

        if embedding_path is None:
            concatenated_df = self.load_embedding_remotely(end_date, start_date)
        else:
            concatenated_df = self.load_embedding_locally(embedding_path, end_date, start_date)

        concatenated_df['embeddings'] = concatenated_df.iloc[:, 1:].apply(lambda row: row.tolist(), axis=1)
        concatenated_df = concatenated_df[['post_id', 'embeddings']]

        return concatenated_df

    def load_embedding_locally(self, embedding_path, end_date, start_date):
        files = [file for file in os.listdir(str(embedding_path)) if
                 os.path.isfile(os.path.join(str(embedding_path), file))]
        filtered_files = [file for file in files if self.is_file_in_range(file, start_date, end_date)]
        concatenated_df = pd.DataFrame()
        for file in filtered_files:
            file_path = os.path.join(str(embedding_path), file)
            file_extension = os.path.splitext(file)[1]
            if file_extension == ".csv":
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)

            print(file, df.shape)
            concatenated_df = pd.concat([concatenated_df, df], ignore_index=True)
        return concatenated_df

    def load_embedding_remotely(self, end_date, start_date):
        files = list_reduce_embedding(country=self.country, task_name=self.cls_task.task, credentials_path="..")
        filtered_files = [file for file in files['files'] if
                          self.is_file_in_range(file['name'], start_date, end_date)]
        concatenated_df = pd.DataFrame()
        for file in filtered_files:
            df = download_file_from_drive(file, credentials_path="..")
            print(file, df.shape)
            concatenated_df = pd.concat([concatenated_df, df], ignore_index=True)
        return concatenated_df

    def filter(self, df, start_date, end_date, only_conflict_related) -> DataFrame:
        print("filter", df.shape)
        df = df[(df['collection_time'] >= start_date) & (df['collection_time'] <= end_date)]

        if only_conflict_related:
            if self.country == "us":
                df = df[df[self.cls_task.get_for_verification_col_name()].isin([1, 2, 3, 9, "1", "2", "3", "9"])]
            else:  # No manual label verification _v for non-us countries
                df = df[df[self.cls_task.get_property_col_name()].isin([1, 2, 3, 9, "1", "2", "3", "9"])]

        return df

    @staticmethod
    def is_file_in_range(file, start_date, end_date):
        f_start_date = datetime.strptime(file.split('__')[0], '%Y-%m-%d_%H-%M-%S')
        f_end_date = datetime.strptime(file.split('__')[1][0:19], '%Y-%m-%d_%H-%M-%S')
        return (f_start_date <= start_date <= f_end_date) or (f_start_date <= end_date <= f_end_date) or (
                start_date <= f_start_date and end_date >= f_end_date)

    @staticmethod
    def date_format(row):
        try:
            if isinstance(row['collection_time'], datetime):
                formatted_date = row['collection_time']
                # print("datetime", formatted_date,type(formatted_date))
                return formatted_date

            if isinstance(row['collection_time'], str):
                for fmt in ["%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"]:
                    try:
                        formatted_date = datetime.strptime(row['collection_time'], fmt)
                        # print("str", formatted_date,type(formatted_date))
                        return formatted_date
                    except ValueError:
                        pass

            return row['collection_time']
        except Exception as e:
            print(e)
