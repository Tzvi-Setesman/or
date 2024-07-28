import pandas as pd
import os
from pathlib import Path
from paths import TWITTER_CLUSTERING_DIR, EMNED_TWITTER_DATA_DIR
from tweet_classifier import TaskName, ClassificationTask
from pipeline import TwitterPipeline, UploadTarget
from drive_functions import download_from_drive, current_dir
import IDs
from drive_functions_wrapper import delete_local_file


def get_conflict_sentiment_values(con_file):
    cls_task = ClassificationTask(task_name=task_name)

    all_data_id_folder = IDs.get_folder_id(country, task_name, IDs.ALL_DATA_FOLDER_ID_DICT)

    # download "all" file for property like: conflict_sentiment values
    all_data_file_name = download_from_drive(parent_folder_id=all_data_id_folder, download_path=current_dir)
    all_data_file_name = all_data_file_name[0]
    all_data_df = pd.read_excel(all_data_file_name)

    for index, row in con_file.iterrows():
        post_id = row['post_id']
        matching_row = all_data_df[all_data_df['post_id'] == post_id]
        if not matching_row.empty:
            con_file.at[index, cls_task.get_property_col_name()] = matching_row[cls_task.get_property_col_name()].values[0]
            if country == "us":
                con_file.at[index, cls_task.get_for_verification_col_name()] = (
                    matching_row[cls_task.get_for_verification_col_name()].values)[0]

    delete_local_file(all_data_file_name)
    print(f"Deleted: {all_data_file_name}")
    return con_file

def concat_files(files_paths, is_embed):
    """
    The function receives files paths, concatenates them into one df and saves it.
    It also receives "is_embed" - a boolean flag which indicates if the files are embedding files (otherwise data files).
    use the function to concat files in order to cluster some dates together, but make sure that you use it twice:
    once for the embedding files and once for the data files.
    PAY ATTENTION: the paths' order is important! sort them from older to newer.
    Assumptions: the files' names are start and end dates, for example:
    "2023-12-12_09-49-42__2023-12-13_09-23-37_embeddings.csv" or
    "2023-12-12_09-49-42__2023-12-13_09-23-37_IsraelSentimentTweet_classified.csv"
    """
    print("start concatenating files")
    dfs = []
    for i in range(len(files_paths)):        
        read_func = None
        ext = Path(files_paths[i]).suffix
        if ext == '.csv':            
            read_func = pd.read_csv
        elif ext == '.xlsx':
            read_func = pd.read_excel
        else:
            raise Exception(f"Unknown ext - not csv or xlsx; {files_paths[i]}")
        dfs.append(read_func(files_paths[i]))
    new_file = pd.concat([dfs[i] for i in range(len(dfs))], ignore_index=True)

    prefix = files_paths[0].split("\\")[-1].split("__")[0]
    suffix = files_paths[-1].split("\\")[-1].split("__")[1]
    if not is_embed:
        suffix = suffix.split("_")[0] + "_" + suffix.split("_")[1] + "_to_cluster.csv"
        print("getting updated <property col> values")
        new_file = get_conflict_sentiment_values(new_file)
    file_name = prefix + "__" + suffix
    if is_embed:
        new_file.to_csv(os.path.join(str(EMNED_TWITTER_DATA_DIR), country, task_name.name, file_name), index=False)
        print(f"Finished concatenating embeddings.\nfile saved to: {os.path.join(EMNED_TWITTER_DATA_DIR, country, file_name)}")
        return os.path.join(EMNED_TWITTER_DATA_DIR, country, file_name)

    new_file.to_csv(os.path.join(str(TWITTER_CLUSTERING_DIR), country, task_name.name, file_name), index=False)
    print(f"Finished concatenating data.\nfile saved to: {os.path.join(TWITTER_CLUSTERING_DIR, country, task_name.name, file_name)}")
    return os.path.join(TWITTER_CLUSTERING_DIR, country, file_name)

def extract_conflict_related_values(df_path, embedding_df_path):
    data_df = pd.read_csv(df_path)
    embed_df = pd.read_csv(embedding_df_path)
    cls_task = ClassificationTask(task_name=task_name)

    if country == "us":
        data = data_df[data_df[cls_task.get_for_verification_col_name()].isin([1, 2, 3, 9, "1", "2", "3", "9"])]
    else: # No manual label verification _v for non-us countries 
        data = data_df[data_df[cls_task.get_property_col_name()].isin([1, 2, 3, 9, "1", "2", "3", "9"])]
    embed_df = embed_df[embed_df.index.isin(data.index)]
    save_name = df_path.split("\\")[-1][:-4] + "_conflict_related.csv"
    data.to_csv(os.path.join(TWITTER_CLUSTERING_DIR, country, task_name.name,save_name), index=False)
    embed_df.to_csv(os.path.join(EMNED_TWITTER_DATA_DIR, country, task_name.name,save_name), index=False)
    df_path = os.path.join(TWITTER_CLUSTERING_DIR, country, task_name.name,save_name)
    embedding_df_path = os.path.join(EMNED_TWITTER_DATA_DIR, country,task_name.name, save_name)
    print("saved conflict related (1239) dfs")
    return df_path, embedding_df_path


if __name__ == "__main__":
    task_names = [TaskName.IsraelSentimentTweet] # TaskName.IsraelSentimentTweet, TaskName.HouthisTweet, TaskName.Unrwa,
    ## Classify a date-range in the past: from: classification_logs to datetime.now() - x_days_before (now)        
    ## If x_days_before > 0, it will be divided into steps of 4 days each from x_days_before, x_days_before - 4, ..., 0
    ## to disable this loop - set x_days_before_loop = False
    x_days_before: int = 0  # Houthis: limit number of records for pipeline (if GPT is too slow or hangs) - 
    x_days_before_loop = False
    
    
    cur_days_before = x_days_before
    
    if x_days_before_loop:
        skip_size = 4
        days_ranges = list(range(x_days_before, -1, -skip_size))
        if not 0 in days_ranges:
            days_ranges.append(0)        
    else:
        days_ranges = [cur_days_before]
        
        
    for task_name in task_names:
        for cur_days_before in days_ranges:
            country = 'us' # uk, fr, de
            run_clustering = False
            cluster_min_size = 50
            df_path = None # r"D:\NLP\Netivot\data\twitter_data\similarity\us\HouthisTweet\2024-03-25_19-02-20__2024-05-05_17-12-28_HouthisTweet_keywords.xlsx" # None # r"D:\NLP\Netivot\data\twitter_data\from_db\us\2023-12-31_16-15-43__2024-01-02_12-08-34.csv" # r"D:\NLP\Netivot\data\twitter_data\clustering\us\2023-10-07_06-00-00__2023-12-26_11-36-06_to_cluster.csv"# './data/twitter_data/from_db/2023-12-19_10-06-47__2023-12-20_09-23-31.csv' # if None read from db, else read from csv_path
            embedding_df_path = None# r"D:\NLP\Netivot\data\twitter_data\embeddings\us\2023-12-25_10-30-38__2023-12-26_11-36-06_embeddings.csv" # r"D:\NLP\Netivot\data\twitter_data\embeddings\us\2023-10-07_06-00-00__2023-12-26_11-36-06_embeddings.csv" # if None calc embeddings, else read from embedding_df
            only_conflict_related = False  # if True - df_path and embedding_df_path shouldn't be None.
            combine_files = False # True to concat multiple df_path and embedding_df_path files- enter files to the following ordered by datetime lists (left is older)
            embed_files_paths_lst = [r"D:\NLP\Netivot\data\twitter_data\embeddings\us\2023-10-07_06-00-00__2023-12-25_10-30-38_embeddings.csv",r"D:\NLP\Netivot\data\twitter_data\embeddings\us\2023-12-25_10-30-38__2023-12-26_11-36-06_embeddings.csv"]  # enter embedding files paths (first oldest, last newest)
            data_files_paths_lst = [r"D:\NLP\Netivot\data\twitter_data\clustering\us\2023-10-07_06-00-00__2023-12-25_10-30-38_to_cluster.csv",r"D:\NLP\Netivot\data\twitter_data\gpt\us\2023-12-25_10-30-38__2023-12-26_11-36-06_IsraelSentimentTweet_classified.csv"]  # enter data files paths (first oldest, last newest)
            gpt_classification_path = None # r"D:\NLP\Netivot\data\twitter_data\gpt\us\2023-12-31_16-15-43__2024-01-02_12-08-34_IsraelSentimentTweet_classified.xlsx"
        
            
            # concat embeddings and datas files
            if combine_files:
                df_path = concat_files(data_files_paths_lst, is_embed=False)
                embedding_df_path = concat_files(embed_files_paths_lst, is_embed=True)
        
            # extract only tweets with verification col like: conflict_sentiment_v in [1, 2, 3, 9]
            if only_conflict_related:
                df_path, embedding_df_path = extract_conflict_related_values(df_path, embedding_df_path)
        
        
            pipe = TwitterPipeline(country=country, task_name=task_name, cluster_min_size=cluster_min_size, clustering=run_clustering)
            pipe.run(df_path, embedding_df_path, gpt_classification_path,
                     run_language_detection=True, run_embedding=True, run_similarity=True, run_filter_by_keywords=False,
                     run_gpt=True, run_clustering=run_clustering, upload_gpt_classification_to=UploadTarget.AFTER_VER_MERGE, # UploadTarget.AFTER_VER_MERGE, # UploadTarget.FOR_VER
                     x_days_before=cur_days_before,filter_similarity=True)
