import os
import uuid
import time
import boto3
import tempfile
import concurrent.futures
import multiprocessing
import argparse
from typing import List, Tuple, Dict
from dotenv import dotenv_values
import bittensor as bt

# Load environment variables
env_config: Dict[str, str] = {**dotenv_values(".env"), **os.environ}
AWS_ACCESS_KEY_ID: str = env_config.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY: str = env_config.get("AWS_SECRET_ACCESS_KEY")

# Instantiate the AWS S3 client
CLIENT: boto3.client = boto3.client(
    "s3",
    region_name="us-east-1",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)


def get_max_workers() -> int:
    """
    Determines the optimal number of worker threads based on CPU count.

    Returns:
    - int: The maximum number of worker threads.
    """
    cpu_count: int = multiprocessing.cpu_count()
    base_workers: int = cpu_count * 2  # Start with 2 workers per CPU core
    # Cap at 32 workers or 4 times CPU count, whichever is smaller
    return min(base_workers, max(32, cpu_count * 4))


MAX_WORKERS: int = get_max_workers()


def list_masks_for_uid(
    bucket: str, hotkey: str, block_number: int
) -> List[Tuple[str, str]]:
    """
    Lists available mask files for a specific UID and block within their bucket.

    Parameters:
    - bucket (str): The S3 bucket name.
    - hotkey (str): The hotkey associated with the UID.
    - block_number (int): The block number to search for.

    Returns:
    - List[Tuple[str, str]]: A list of tuples containing the bucket and mask file key.
    """
    try:
        if bucket is None:
            return []  # Skip if no bucket
        prefix: str = f"mask-{hotkey}-"
        paginator: boto3.Paginator = CLIENT.get_paginator("list_objects_v2")
        response_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
        mask_files: List[str] = []
        for page in response_iterator:
            if "Contents" in page:
                for obj in page["Contents"]:
                    key: str = obj["Key"]
                    # Check if the key matches the specific block number
                    if key.endswith(f"-{block_number}.pt"):
                        mask_files.append(key)
        return [(bucket, key) for key in mask_files]
    except Exception as e:
        # TODO: Handle specific exceptions (e.g., access denied)
        return []


def download_file(bucket: str, filename: str) -> str:
    """
    Downloads a file from S3 to a temporary file.

    Parameters:
    - bucket (str): The name of the S3 bucket.
    - filename (str): The name of the file to download.

    Returns:
    - str: The path to the downloaded temporary file, or None if the download failed.
    """
    try:
        if bucket is None:
            return None
        unique_temp_file: str = os.path.join(
            tempfile.gettempdir(), f"{uuid.uuid4()}.pt"
        )
        CLIENT.download_file(bucket, filename, unique_temp_file)
        return unique_temp_file
    except Exception as e:
        # TODO: Log or handle exceptions as needed
        return None


def new_download_process(
    block_range: range, uids: List[int], hotkeys: Dict[int, str], buckets: List[str]
) -> None:
    """
    Optimized method of downloading mask files from S3.

    Parameters:
    - block_range (range): The range of blocks to process.
    - uids (List[int]): List of UIDs.
    - hotkeys (Dict[int, str]): Mapping of UIDs to hotkeys.
    - buckets (List[str]): List of buckets corresponding to each UID.

    This function lists available mask files and only attempts to download those that exist.
    """
    total_start_time: float = time.time()
    total_downloaded: int = 0

    for blk in block_range:
        print(f"Processing block: {blk}")
        start_time: float = time.time()
        # Concurrently list masks across buckets
        mask_tasks: List[Tuple[str, str]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(list_masks_for_uid, bucket, hotkeys[uid], blk)
                for uid, bucket in zip(uids, buckets)
                if bucket is not None
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                mask_tasks.extend(result)
        print(f"Found {len(mask_tasks)} masks for block {blk}")

        temp_files: List[str] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(download_file, bucket, key)
                for bucket, key in mask_tasks
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    temp_files.append(result)
                    total_downloaded += 1

        print(
            f"Block {blk}: Downloaded {len(temp_files)} files in {time.time() - start_time:.2f} seconds"
        )
        # Clean up temp files
        for file in temp_files:
            os.remove(file)

    total_time: float = time.time() - total_start_time
    print(f"\nTotal files downloaded: {total_downloaded}")
    print(f"Total time taken: {total_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="New optimized download process")
    parser.add_argument("--netuid", type=int, default=1, help="Subtensor network UID")
    config = parser.parse_args()

    # Initialize the subtensor object
    subtensor: bt.subtensor = bt.subtensor(
        network="test", chain_endpoint="wss://test.finney.opentensor.ai:443/"
    )

    # Sync the metagraph to get current UIDs and hotkeys
    print("Syncing metagraph...")
    metagraph: bt.Metagraph = subtensor.metagraph(config.netuid)
    metagraph.sync()
    print(f"Metagraph synced with {len(metagraph.uids)} UIDs.")

    UIDS: List[int] = metagraph.uids.tolist()
    HOTKEYS: Dict[int, str] = {uid: metagraph.hotkeys[i] for i, uid in enumerate(UIDS)}

    # Get buckets per UID
    BUCKETS: List[str] = []
    for uid in UIDS:
        try:
            # Get the bucket commitment for each UID
            bucket: str = subtensor.get_commitment(config.netuid, uid)
            BUCKETS.append(bucket)
        except Exception as e:
            # Handle exceptions (e.g., UID not committed)
            BUCKETS.append(None)

    # Define the blocks you want to process
    current_block: int = subtensor.block
    BLOCK_RANGE: range = range(current_block - 9, current_block + 1)

    new_download_process(BLOCK_RANGE, UIDS, HOTKEYS, BUCKETS)
