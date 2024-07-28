import os
CURRENT_FILE_PATH = os.path.abspath(__file__)
CURRENT_FOLDER = os.path.dirname(CURRENT_FILE_PATH)
HOME = os.path.join(CURRENT_FOLDER,"data")

# mongo paths
TWITTER_DIR = os.path.join(HOME,'twitter_data')
NEWS_DIR = os.path.join(HOME,'news_data')
RAW_TWITTER_DATA_DIR = os.path.join(TWITTER_DIR,'from_db')
RAW_NEWS_DATA_DIR = os.path.join(NEWS_DIR,'from_db')
DATETIME_FILE_NAME = "last_query_datetime.txt"
TABLE_NAME_FILE_NMAE = "last_table_name.txt"
SUMMARIZED_NEWS_DIR = os.path.join(NEWS_DIR,'summarized')
EMNED_TWITTER_DATA_DIR = os.path.join(TWITTER_DIR,'embeddings')
REDUCE_EMBED_TWITTER_DATA_DIR = os.path.join(TWITTER_DIR,'reduce_embeddings')
EMNED_NEWS_DATA_DIR = os.path.join(NEWS_DIR,'embeddings')
# GPT runner
GPT_OUTPUTS_BACKUP_PATH =  os.path.join(HOME, 'backups')
# MAIN
TWITTER_SIMILARITY_DIR =  os.path.join(TWITTER_DIR,'similarity')
TWITTER_CLASSIFIED_DIR =  os.path.join(TWITTER_DIR,'gpt')
TWITTER_CLUSTERING_DIR =  os.path.join(TWITTER_DIR,'clustering')
TWITTER_CLASSIFICATION_LOGS = os.path.join(TWITTER_CLASSIFIED_DIR, "classification_logs.json")
NEWS_CLASSIFIED_DIR =  os.path.join(NEWS_DIR,'gpt')
NEWS_CLUSTERING_DIR =  os.path.join(NEWS_DIR,'clustering')
NEWS_CLASSIFICATION_LOGS = os.path.join(NEWS_CLASSIFIED_DIR, "classification_logs.json")

