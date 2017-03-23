import os
import logging
from google.cloud import storage

logger = logging.getLogger(__name__)


def download_from_gcs(bucket_name: str, prefix: str, local_fs_loc: str):
    """
    Downloads blobs in a gcs bucket to the local disk. Will create the necessary local directories if they don't exist
    :param bucket_name:
    :param prefix:
    :param local_fs_loc:
    :return:
    """

    check_and_create_parent_folder(local_fs_loc)

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    for blob in bucket.list_blobs(prefix=prefix):
        if "_SUCCESS" in blob.name or "//" in blob.name:
            continue
        split_blob_name = blob.name.split('/')
        split_prefix = prefix.split('/')
        writefile_name = "/".join([b for b in split_blob_name if b not in split_prefix])
        download_location = os.path.join(local_fs_loc, writefile_name)
        check_and_create_parent_folder(download_location)
        blob.download_to_filename(download_location)


def copy_file_to_gcs(bucket_name: str,
                     filename: str,
                     destination_path: str):
    """
    Write a local file to GCS

    :param bucket_name: The name of the bucket to write to
    :param filename: the name of the local file to write
    :param destination_path: the destination path that we are writing to, minus tha "gs://<bucket-name>/"
    :return:
    """
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(destination_path)
    blob.upload_from_filename(filename=filename)


def check_and_create_parent_folder(file_path: str) -> None:
    """
    Check that a file_path's parent folder exists and create it otherwise

    :param file_path: the path of the file we wish to check for
    :return: None
    """
    logger.info("Checking if {parent_folder} exists".format(parent_folder=file_path))
    try:
        logger.info("Creating {parent_folder}".format(parent_folder=file_path))
        os.makedirs("/".join(file_path.split("/")[:-1]))
    except FileExistsError:
        logger.info("Already exists {parent_folder}".format(parent_folder=file_path))
        pass
