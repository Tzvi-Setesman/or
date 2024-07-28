import os
from datetime import datetime

from clustering_pipeline import ClusteringPipeline
from paths import REDUCE_EMBED_TWITTER_DATA_DIR, EMNED_TWITTER_DATA_DIR
from tweet_classifier import TaskName

if __name__ == "__main__":
    country = 'us'  # uk, fr, de
    cluster_min_size = 50
    task_name = TaskName.IsraelSentimentTweet

    start_date = datetime.strptime("2023-12-28_11-39-00", "%Y-%m-%d_%H-%M-%S")  # =None
    end_date = datetime.strptime("2023-12-31_16-13-59", "%Y-%m-%d_%H-%M-%S")  # =None

    # EMNED_TWITTER_DATA_DIR - full embeddings locally dir
    # REDUCE_EMBED_TWITTER_DATA_DIR - reduced embeddings locally dir
    # None - to load from drive,
    embedding_path = os.path.join(EMNED_TWITTER_DATA_DIR, country,task_name.name)

    # True - if should use full embeddings
    # False - if embeddings files were already reduced
    reduce_dimensions = True

    random_state = None

    pipe = ClusteringPipeline(country=country, task_name=task_name, cluster_min_size=cluster_min_size)
    pipe.run(start_date=start_date, end_date=end_date, embedding_path=embedding_path,
             reduce_dimensions = reduce_dimensions,
             only_conflict_related=False,
             random_state=random_state)
