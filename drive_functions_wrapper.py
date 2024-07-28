import os
import time

import pandas as pd
from pandas import DataFrame

import IDs
from drive_functions import download_from_drive, current_dir, list_files_from_drive
from tweet_classifier import TaskName


def delete_local_file(path):
    try:
        if os.path.exists(path):
            os.remove(path)
        print(f"Deleted: {path}")
    except PermissionError:
        print(f"File {path} is in use. Will retry after a short delay.")
        time.sleep(1)
        delete_local_file(path)


def download_all_data(country, task_name:TaskName, credentials_path=None) -> DataFrame:
    all_data_id_folder = IDs.get_folder_id(country, task_name, IDs.ALL_DATA_FOLDER_ID_DICT)

    all_data_file_name = download_from_drive(parent_folder_id=all_data_id_folder,
                                             download_path=current_dir,
                                             credentials_path=credentials_path)

    all_data_file_name = all_data_file_name[0]
    all_data_df = pd.read_excel(all_data_file_name)
    delete_local_file(all_data_file_name)

    return all_data_df


def list_reduce_embedding(country, task_name:TaskName, credentials_path=None):
    id_folder = IDs.get_folder_id(country, task_name, IDs.REDUCE_EMBEDDINGS_FOLDER_ID_DICT)

    return list_files_from_drive(parent_folder_id=id_folder,
                                 credentials_path=credentials_path)


def download_file_from_drive(file_info, credentials_path=None) -> DataFrame:
    file_name = download_from_drive(file_info=file_info,
                                    download_path=current_dir,
                                    credentials_path=credentials_path)

    file_name = file_name[0]
    file_extension = os.path.splitext(file_name)[1]

    if file_extension == ".csv":
        data_df = pd.read_csv(file_name)
    else:
        data_df = pd.read_excel(file_name)

    delete_local_file(file_name)

    return data_df
