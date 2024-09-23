import os
import uuid
import time
import boto3
import tempfile
import concurrent.futures
import argparse
from typing import List, Dict
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
            return None  # Skip if no bucket
        # Generate a unique temporary file path
        unique_temp_file: str = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.pt")
        # Download the file from S3
        CLIENT.download_file(bucket, filename, unique_temp_file)
        return unique_temp_file
    except Exception as e:
        # TODO: Handle specific exceptions (e.g., network errors)
        return None

def old_download_process(
    block_range: range,
    uids: List[int],
    hotkeys: List[str],
    buckets: List[str]
) -> None:
    """
    Old method of downloading mask files from S3.

    Parameters:
    - block_range (range): The range of blocks to process.
    - uids (List[int]): List of UIDs.
    - hotkeys (List[str]): List of hotkeys corresponding to each UID.
    - buckets (List[str]): List of buckets corresponding to each UID.

    This function iterates over all blocks and UIDs, attempting to download each possible mask file.
    """
    total_start_time: float = time.time()
    total_downloaded: int = 0

    print(f"Downloading masks for blocks: {list(block_range)}")
    for block_number in block_range:
        print(f"\nProcessing block: {block_number}")
        start_time: float = time.time()

        # Generate mask filenames for all UIDs using their hotkeys
        print(f"Generating filenames for block {block_number}...")
        mask_filenames: List[str] = [
            f"mask-{hotkey}-{block_number}.pt" for hotkey in hotkeys
        ]
        print(f"Filename generation completed in {time.time() - start_time:.2f} seconds")

        # Prepare lists for buckets and filenames
        mask_buckets: List[str] = buckets
        temp_files: List[str] = []
        n_downloaded: int = 0

        # Download the masks from all valid files
        print(f"Downloading masks for block {block_number}...")
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(download_file, bucket, filename)
                for bucket, filename in zip(mask_buckets, mask_filenames)
                if bucket is not None
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    temp_files.append(result)
                    n_downloaded += 1

        print(f"Downloaded {n_downloaded} masks in {time.time() - start_time:.2f} seconds")

        # Break the loop when there is nothing to download
        if n_downloaded == 0:
            continue

        # TODO: Process the downloaded mask files as needed

        # Clean up temp files
        for file_path in temp_files:
            os.remove(file_path)
            # Note: Ensure that the temporary files are properly deleted to avoid disk space issues

        total_downloaded += n_downloaded

    total_time: float = time.time() - total_start_time
    print(f"\nTotal files downloaded: {total_downloaded}")
    print(f"Total time taken: {total_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Old download process")
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

    # Retrieve UIDs and hotkeys from the metagraph
    UIDS: List[int] = metagraph.uids.tolist()
    HOTKEYS: List[str] = metagraph.hotkeys

    # Get buckets per UID
    print("Retrieving buckets per UID...")
    buckets: List[str] = []
    for uid in UIDS:
        try:
            # Get the bucket commitment for each UID
            bucket: str = subtensor.get_commitment(config.netuid, uid)
            buckets.append(bucket)
        except Exception as e:
            # Handle exceptions (e.g., UID not committed)
            buckets.append(None)

    # Define the blocks you want to process
    current_block: int = subtensor.block
    BLOCK_RANGE: range = range(current_block - 9, current_block + 1)

    # Call the old download process
    old_download_process(BLOCK_RANGE, UIDS, HOTKEYS, buckets)