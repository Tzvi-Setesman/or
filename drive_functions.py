# Those functions are being used in the merge file process - to download, upload and move files
import io
import os
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.http import MediaIoBaseDownload

current_dir = os.getcwd()
SERVICE_ACCOUNT_CREDENTIALS_FILE = "winter-flare-404912-16eb155c4b9c.json"


def get_service_account_for_google_drive(credentials_path=None):
    # Replace with the path to your credentials JSON file

    if credentials_path is None:
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_CREDENTIALS_FILE, scopes=['https://www.googleapis.com/auth/drive']
        )
    else:
        credentials_path = os.path.join(credentials_path,SERVICE_ACCOUNT_CREDENTIALS_FILE)
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=['https://www.googleapis.com/auth/drive']
        )

    drive_service = build('drive', 'v3', credentials=credentials)
    return drive_service


def list_files_from_drive(parent_folder_id,credentials_path=None):
    drive_service = get_service_account_for_google_drive(credentials_path)

    try:
        return drive_service.files().list(
            q=f"'{parent_folder_id}' in parents and (mimeType = 'text/csv' or  mimeType = 'application/vnd.google-apps.spreadsheet' or mimeType = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')",
            fields='files(id, name, mimeType)',
            pageSize=1000
        ).execute()

    except Exception as e:
        print(f"Error retrieving files from Google Drive: {e}")


# downloads only excel and google sheets files as xlsx file, returns a list of all downloaded files' names
def download_from_drive(parent_folder_id=None, file_info=None, download_path=".",credentials_path=None):

    if parent_folder_id is None and file_info is None:
        raise Exception("Please provide either a parent folder id or file info")

    drive_service = get_service_account_for_google_drive(credentials_path)
    downloaded_files = []

    try:
        if parent_folder_id is not None:
            results = drive_service.files().list(
                q=f"'{parent_folder_id}' in parents and (mimeType = 'application/vnd.google-apps.spreadsheet' or mimeType = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')",
                fields='files(id, name, mimeType)',
                pageSize=1000
            ).execute()

            files_data = results.get('files', [])
        else:
            files_data = [file_info]

        for file_info in files_data:
            file_id = file_info['id']
            file_name = file_info['name']
            mime_type = file_info.get('mimeType', '')

            if 'application/vnd.google-apps' in mime_type:
                # Export Google Sheets as Excel (xlsx)
                request = drive_service.files().export_media(fileId=file_id, mimeType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                file_extension = 'xlsx'
            else:
                # Download regular Excel files
                request = drive_service.files().get_media(fileId=file_id)
                if 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' in mime_type\
                        or 'text/csv' in mime_type:
                    file_extension = ''  # Already in .xlsx format, no additional extension needed
                else:
                    file_extension = 'unknown'  # You can adjust this based on the actual file type

            file_content = io.BytesIO()
            downloader = MediaIoBaseDownload(file_content, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()

            # Save the file without adding an extra dot
            filename_with_extension = f"{file_name}.{file_extension}".rstrip('.')
            downloaded_files.append(filename_with_extension)
            with open(os.path.join(download_path, filename_with_extension), 'wb') as f:
                f.write(file_content.getvalue())

            print(f"Downloaded: {filename_with_extension}")

    except Exception as e:
        print(f"Error retrieving or downloading files from Google Drive: {e}")

    return downloaded_files


def upload_to_drive(file_path, destination_folder_id):
    drive_service = get_service_account_for_google_drive()

    path = Path(file_path)

    try:
        # Create GoogleDriveFile instance with the name and folder ID
        file_drive = drive_service.files().create(
            body={'name': path.name, 'parents': [destination_folder_id]},
            media_body=MediaFileUpload(file_path, resumable=True)
        ).execute()

        print(f"File '{file_path}' uploaded to Google Drive in folder with ID: {destination_folder_id}")

    except Exception as e:
        print(f"Error uploading file to Google Drive: {e}")


# used with the archive folder - to move files to archive after merging them
def move_file_to_different_folder(file_path, origin_folder_id, destination_folder_id):
    drive_service = get_service_account_for_google_drive()

    try:
        # Get the file ID by searching for the file in the origin folder
        file_id = None
        results = drive_service.files().list(
            q=f"'{origin_folder_id}' in parents and name = '{file_path.split('/')[-1]}'",
            fields='files(id)',
            pageSize=1
        ).execute()

        files_data = results.get('files', [])
        if files_data:
            file_id = files_data[0]['id']

        if file_id:
            # Move the file by updating its parents
            drive_service.files().update(
                fileId=file_id,
                addParents=destination_folder_id,
                removeParents=origin_folder_id,
                fields='id, parents'
            ).execute()

            print(f"File '{file_path}' moved from folder '{origin_folder_id}' to folder '{destination_folder_id}'")

        else:
            print(f"ERROR - File '{file_path}' not found in folder '{origin_folder_id}'")

    except Exception as e:
        print(f"Error moving file: {e}")


