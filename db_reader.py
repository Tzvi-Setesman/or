import datetime
import json
import os
from enum import Enum
from pathlib import Path

import pandas as pd
import requests

from credentials import MONGO_URL, X_API_KEY
from paths import RAW_TWITTER_DATA_DIR, RAW_NEWS_DATA_DIR, DATETIME_FILE_NAME, TABLE_NAME_FILE_NMAE
from tweet_classifier import ClassificationTask


class DataSource(Enum):
    TWITTER = "twitter"
    NEWS = "news"


class TweeterMongoDB:
    def __init__(self, data_source: DataSource, cls_task: ClassificationTask):
        self.data_source = data_source
        self.cls_task = cls_task

        if data_source == DataSource.TWITTER:
            self.raw_data_dir = RAW_TWITTER_DATA_DIR
        elif data_source == DataSource.NEWS:
            self.raw_data_dir = RAW_NEWS_DATA_DIR

    def request_tweet_id(self, post_id):
        collection_name = "tweets"
        query = {"post_id": post_id}
        fields_to_include = ['profile_name', 'post_id', 'uploader_id', 'collection_time',
                             'upload_date', 'post_text', 'comments_amount',
                             'retweet_amount', 'like_amount', 'views_amount', 'country']
        projection = {field: 1 for field in fields_to_include}

        return self.request_data(collection_name, projection, query)

    def request_twitter_data(self, start_date, end_date, country=None):

        collection_name = "tweets"

        fields_to_include = ['profile_name', 'post_id', 'uploader_id', 'collection_time',
                             'upload_date', 'post_text', 'comments_amount',
                             'retweet_amount', 'like_amount', 'views_amount']

        projection = {field: 1 for field in fields_to_include}

        query = {
            "collection_time": {"$gte": start_date, "$lte": end_date}
        }
        if country is not None:
            query["metadata.country"] = country

        df = self.request_data(collection_name, projection, query)

        if df.shape[0] == 0:
            print("No data was found")
            return df
        df = df[fields_to_include]
        df["country"] = country
        df["post_id"] = df["post_id"] + '_a'
        df["uploader_id"] = df["uploader_id"] + '_a'
        df = df.rename(columns={"profile_name": "username"})
        print(f"Found {df.shape[0]} tweets")
        return df

    # TODO: News query is currently not in use and therefore not tested
    def request_news_data(self, start_date, end_date, lang=None):

        collection_name = "news"

        fields_to_include = ['_id', 'title', 'url', 'content',
                             'lang', 'site_name', 'publishing_time',
                             'collection_time', 'authors', 'tags']

        projection = {field: 1 for field in fields_to_include}

        query = {
            "collection_time": {"$gte": start_date, "$lte": end_date}
        }
        if lang is not None:
            query["lang"] = lang

        df = self.request_data(collection_name, projection, query)

        if df.shape[0] == 0:
            print("No data was found")
            return df
        df = df[fields_to_include]
        print(f"Found {df.shape[0]} articles")
        return df

    def query_data_from_last_time(self, country_or_lang, start_time=None, x_days_before: int = 0):
        df = pd.DataFrame()

        if start_time is None:
            last_query_datetime = self.read_datetime_from_file(country_or_lang)
        else:
            last_query_datetime = start_time
        current_datetime = datetime.datetime.now().replace(microsecond=0)
        current_datetime = current_datetime - datetime.timedelta(days=x_days_before)

        print("Query DB for {} from: {} to {}".format(country_or_lang, last_query_datetime, current_datetime))
        if self.data_source == DataSource.TWITTER:
            df = self.request_twitter_data(last_query_datetime, current_datetime, country_or_lang)
        elif self.data_source == DataSource.NEWS:
            df = self.request_news_data(last_query_datetime, current_datetime, country_or_lang)
        if df.shape[0] == 0:
            return df
        self.write_current_datetime_to_file(current_datetime, country_or_lang)
        self.save_df(df, last_query_datetime, current_datetime, country_or_lang)
        return df

    def read_datetime_from_file(self, country_or_lang):
        if not os.path.exists(os.path.join(self.raw_data_dir, country_or_lang, self.cls_task.name,
                                           DATETIME_FILE_NAME)):
            last_query_datetime = datetime.datetime.now() - datetime.timedelta(hours=20)
            print("No datetime file was found - start query from: ", last_query_datetime)
            return last_query_datetime
        with open(os.path.join(self.raw_data_dir, country_or_lang, self.cls_task.name,
                               DATETIME_FILE_NAME), "r") as file:
            datetime_str = file.read()
        last_query_datetime = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
        return last_query_datetime

    def write_current_datetime_to_file(self, current_datetime, country_or_lang):
        current_datetime_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        cls_task_dir = os.path.join(self.raw_data_dir, country_or_lang, self.cls_task.name)
        if not os.path.exists(cls_task_dir):
            os.makedirs(cls_task_dir)
        with open(os.path.join(Path(str(cls_task_dir)), DATETIME_FILE_NAME), "w") as file:
            file.write(current_datetime_str)

    def save_df(self, df, last_query_datetime, current_datetime, country_or_lang):
        start_date_str = last_query_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        end_date_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        cls_task_dir = os.path.join(self.raw_data_dir, country_or_lang, self.cls_task.name)
        if not os.path.exists(cls_task_dir):
            os.makedirs(cls_task_dir)
        fname = os.path.join(str(cls_task_dir), f"{start_date_str}__{end_date_str}.csv")
        df.to_csv(fname, index=False)
        with open(os.path.join(str(cls_task_dir), TABLE_NAME_FILE_NMAE), "w") as file:
            file.write(f"{start_date_str}__{end_date_str}.csv")

    def get_last_table_name(self, country_or_lang):
        with open(os.path.join(self.raw_data_dir, country_or_lang, self.cls_task.name,
                               TABLE_NAME_FILE_NMAE), "r") as file:
            last_table_name = file.read()
        return last_table_name

    def request_data(self, collection_name, projection, query):

        results_list = []
        skip = 0
        limit = 1000

        headers = {
            "x-api-key": X_API_KEY,
            "Content-Type": "application/json"
        }

        while True:

            params = {
                "collection_name": collection_name,
                "query": json.dumps(query, default=str),  # optional, default: {}
                "projection": json.dumps(projection, default=str),
                "limit": limit,  # optional, default: 10
                "skip": skip  # optional, default: 0
            }

            skip += limit

            result = self.make_request(params=params, headers=headers)

            if len(result['data']) == 0:
                break

            results_list.append(result)

        if len(results_list) == 0:
            return pd.DataFrame()

        return pd.concat([pd.DataFrame(result['data']) for result in results_list], ignore_index=True)

    @staticmethod
    def make_request(params, headers):
        response = requests.get(MONGO_URL, params=params, headers=headers, verify=False)
        return response.json()
