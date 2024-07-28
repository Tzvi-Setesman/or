import datetime
import unittest
from pathlib import Path
from unittest import TestCase

from db_reader import TweeterMongoDB, DataSource
from tweet_classifier import TaskName, ClassificationTask


class DBReaderTestCase(TestCase):
    request_twitter_data_path = Path('test_request_twitter_data.csv')
    request_news_data_path = Path('request_news_data.csv')

    @classmethod
    def setUpClass(cls):
        cls.cls_task = ClassificationTask(TaskName.IsraelSentimentTweet)

    @classmethod
    def tearDownClass(cls):
        if cls.request_twitter_data_path.exists():
            cls.request_twitter_data_path.unlink()

        if cls.request_news_data_path.exists():
            cls.request_news_data_path.unlink()

    def test_request_twitter_data(self):
        def preprocess_date_format(row):
            return datetime.datetime.strptime(row['collection_time'], '%Y-%m-%dT%H:%M:%S.%f')

        db = TweeterMongoDB(DataSource.TWITTER,self.cls_task)
        start_date = datetime.datetime(2024, 1, 7, 00, 00, 00)
        end_date = datetime.datetime(2024, 1, 7, 23, 59, 59)
        country = 'us'
        df = db.request_twitter_data(start_date, end_date, country)
        df.to_csv(self.request_twitter_data_path, index=False)
        self.assertGreater(df.shape[0],0)

        df['collection_time'] = df.apply(preprocess_date_format, axis=1)

        df_filter = df[(df['collection_time'] >= start_date) & (df['collection_time'] <= end_date)]
        self.assertEqual(df.shape[0], df_filter.shape[0])

    def test_tweet_id(self):

        db = TweeterMongoDB(DataSource.TWITTER,self.cls_task)
        df = db.request_tweet_id("1742600116396073081")
        self.assertGreater(df.shape[0], 0)


    @unittest.skip
    def test_request_news_data(self):
        db = TweeterMongoDB(DataSource.NEWS)
        start_date = datetime.datetime(2024, 1, 7, 00, 00, 00)
        end_date = datetime.datetime(2024, 1, 7, 23, 59, 59)
        country = 'us'
        df = db.request_news_data(start_date, end_date, country)
        df.to_csv(self.request_news_data_path, index=False)
        self.assertGreater(df.shape[0], 0)

