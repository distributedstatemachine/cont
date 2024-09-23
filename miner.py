# The MIT License (MIT)
# © 2024 Chakana.tech

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import io
import os
import uuid
import time
import wandb
import boto3
import torch
import tempfile
import argparse
import traceback
import numpy as np
from tqdm import tqdm
import bittensor as bt
import concurrent.futures
import torch.optim as optim
from typing import List, Tuple
from dotenv import dotenv_values
from transformers import LlamaForCausalLM
from torch.optim.lr_scheduler import CosineAnnealingLR
import multiprocessing

from hparams import load_hparams
from dataset import SubsetFineWebEdu2Loader

# Enable cuDNN benchmark for optimized performance
torch.backends.cudnn.benchmark = True

# Instantiate the AWS S3 client.
env_config = {**dotenv_values(".env"), **os.environ}  # Load environment variables.
AWS_ACCESS_KEY_ID = env_config.get("AWS_ACCESS_KEY_ID")  # AWS access key ID.
AWS_SECRET_ACCESS_KEY = env_config.get(
    "AWS_SECRET_ACCESS_KEY"
)  # AWS secret access key.
CLIENT: boto3.client = boto3.client(
    "s3",
    region_name="us-east-1",  # AWS region.
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)


def list_masks_for_uid(bucket, hotkey, blk):
    try:
        if bucket is None:
            return []  # No bucket means no masks
        prefix = f"mask-{hotkey}-"
        paginator = CLIENT.get_paginator("list_objects_v2")
        response_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
        mask_files = []
        for page in response_iterator:
            if "Contents" in page:
                for obj in page["Contents"]:
                    key = obj["Key"]
                    if key.endswith(f"-{blk}.pt"):
                        mask_files.append(key)
        return [(bucket, key) for key in mask_files]
    except Exception as e:
        return []


def download_file(bucket, filename):
    try:
        if bucket is None:
            return None
        unique_temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.pt")
        CLIENT.download_file(bucket, filename, unique_temp_file)
        return unique_temp_file
    except Exception as e:
        return None


def get_max_workers():
    cpu_count = multiprocessing.cpu_count()
    base_workers = cpu_count * 2  # Start with 2 workers per CPU core

    # Cap at 32 workers or 4 times CPU count, whichever is smaller
    return min(base_workers, min(32, cpu_count * 4))


# Set MAX_WORKERS globally
MAX_WORKERS = get_max_workers()


def main(config):
    # Print the configuration settings.
    print("\n", "-" * 40, "Config", "-" * 40)
    print(config)

    # Initialize Bittensor objects.
    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(netuid=config.netuid)
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        raise ValueError(
            f"Wallet {wallet} is not registered on subnet: {metagraph.netuid}"
        )
    my_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    print("\n", "-" * 40, "Objects", "-" * 40)
    print(
        f"Wallet: {wallet}\nSubtensor: {subtensor}\nMetagraph: {metagraph}\nUID: {my_uid}"
    )

    # Initialize my bucket information by submitting it to the chain.
    try:
        if config.bucket != subtensor.get_commitment(config.netuid, my_uid):
            raise ValueError(f"Chain commitment does not match: {config.bucket}")
    except Exception:
        # If not committed or mismatch, commit the bucket to the chain.
        subtensor.commit(wallet, config.netuid, config.bucket)
    print("Bucket:", config.bucket)

    # Initialize Weights and Biases (wandb) for experiment tracking if enabled.
    if config.use_wandb:
        run = wandb.init(
            project="cont", resume="allow", name=f"M{my_uid}", config=config
        )

    # Initialize training state.
    hparams = load_hparams()
    model = None
    upload_history = []
    last_mask_sync = 0
    last_master_sync = 0
    while True:
        try:
            # Sync the current chain state and hyperparameters.
            print("Loading chain state ...")
            start_time = time.time()
            hparams = load_hparams()
            subtensor = bt.subtensor(config=config)
            metagraph = subtensor.metagraph(netuid=config.netuid)
            print(
                f"Loading chain state completed in {time.time() - start_time} seconds"
            )

            # Sync the full model state every hparams.epoch_length
            print(f"Checking epoch sync ...")
            start_time = time.time()
            if (
                model is None
                or subtensor.block - last_master_sync > hparams.epoch_length
            ):
                try:
                    master_uid = int(metagraph.S.argmax())
                    master_bucket = subtensor.get_commitment(config.netuid, master_uid)
                    master_hotkey = metagraph.hotkeys[master_uid]
                    master_filename = f"master-{master_hotkey}.pt"
                    unique_temp_file = os.path.join(
                        tempfile.gettempdir(), f"{uuid.uuid4()}.pt"
                    )
                    CLIENT.download_file(
                        master_bucket, master_filename, unique_temp_file
                    )
                    master_state_dict = torch.load(
                        unique_temp_file, map_location="cpu", weights_only=True
                    )
                    model = LlamaForCausalLM(config=hparams.model_config)
                    model.load_state_dict(master_state_dict)
                    model.to(config.device)
                    model.train()
                    optimizer = optim.AdamW(
                        model.parameters(),
                        lr=config.learning_rate,  # Peak learning rate
                        betas=(
                            config.optimizer_beta1,
                            config.optimizer_beta2,
                        ),  # B1 and B2
                        weight_decay=config.optimizer_weight_decay,  # Weight decay
                    )
                    scaler = torch.cuda.amp.GradScaler("cuda")
                    scheduler = CosineAnnealingLR(
                        optimizer,
                        T_max=hparams.epoch_length,
                        eta_min=4e-5,
                        last_epoch=-1,
                    )
                    last_master_sync = subtensor.block
                    last_mask_sync = last_master_sync
                except Exception as e:
                    print(f"No master: {e} Waiting ...")
                    time.sleep(12)
                    continue
            print(
                f"Checking epoch sync: completed in {time.time() - start_time} seconds"
            )

            # Get current block information.
            print(f"Getting block state ...")
            start_time = time.time()
            current_block = subtensor.block
            all_sync_blocks = list(range(last_mask_sync - 2, current_block + 1))
            last_mask_sync = current_block
            print(f"Getting block completed in {time.time() - start_time} seconds")

            # Get buckets per UID if needed.
            if "buckets" not in locals() or len(buckets) != len(metagraph.uids):
                buckets = []
                for uid in metagraph.uids:
                    try:
                        buckets.append(subtensor.get_commitment(config.netuid, uid))
                    except:
                        buckets.append(None)

            # Prepare a list of (uid, bucket, hotkey)
            uids_info = list(zip(metagraph.uids, buckets, metagraph.hotkeys))

            # Process each block separately to ensure consistency.
            print(f"Downloading masks for blocks: {all_sync_blocks}")
            full_sync_start_time = time.time()
            for blk in all_sync_blocks:
                print(f"Processing block: {blk}")

                # Generate mask indices using the block number.
                print(f"Creating sync mask for block: {blk} ...")
                mask_indices = {}
                torch.manual_seed(blk)
                for name, param in model.named_parameters():
                    param = param.to(config.device)
                    next_mask = (
                        torch.rand(param.shape, device=config.device)
                        < (1 / hparams.compression)
                    ).float()
                    indices = next_mask.flatten().nonzero(as_tuple=False).flatten()
                    mask_indices[name] = indices.cpu()
                print(
                    f"Creating sync block mask completed in {time.time() - start_time} seconds"
                )

                # List masks available for this block.
                print(f"Getting filenames for block: {blk} ...")
                mask_tasks = []
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=MAX_WORKERS
                ) as executor:
                    futures = [
                        executor.submit(list_masks_for_uid, bucket, hotkey, blk)
                        for (uid, bucket, hotkey) in uids_info
                        if bucket is not None
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        mask_tasks.extend(result)
                print(f"Found {len(mask_tasks)} masks for block {blk}")

                # Proceed only if there are masks to download.
                if not mask_tasks:
                    continue

                # Download the masks from all valid files.
                print(f"Downloading masks for block: {blk} ...")
                start_time = time.time()
                temp_files = []
                n_downloaded = 0
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=MAX_WORKERS
                ) as executor:
                    futures = [
                        executor.submit(download_file, bucket, key)
                        for bucket, key in mask_tasks
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        if result:
                            temp_files.append(result)
                            n_downloaded += 1
                print(
                    f"Block {blk}: Downloaded {n_downloaded} files in {time.time() - start_time:.2f} seconds"
                )

                # Skip if no masks were downloaded.
                if n_downloaded == 0:
                    continue

                # Load all masks as state dicts.
                print(f"Loading state dicts for block: {blk} ...")
                start_time = time.time()
                mask_count = 0
                masks_dicts_values = {}
                for file in temp_files:
                    try:
                        mask = torch.load(file, map_location="cpu", weights_only=True)
                        mask_count += 1
                        for name in mask.keys():
                            mask_values = mask[name]["values"]
                            if torch.isnan(mask_values).any():
                                continue
                            param_shape = model.get_parameter(name).shape
                            indices = mask_indices[name]
                            if indices.shape[0] != mask_values.shape[0]:
                                print(
                                    f"Skipping parameter {name} due to shape mismatch."
                                )
                                continue
                            decompressed = torch.zeros(
                                param_shape, device="cpu"
                            ).flatten()
                            decompressed[indices] = mask_values
                            if name not in masks_dicts_values:
                                masks_dicts_values[name] = decompressed.view(
                                    param_shape
                                )
                            else:
                                masks_dicts_values[name] += decompressed.view(
                                    param_shape
                                )
                    except Exception as e:
                        print(f"Error loading mask from {file}: {e}")
                print(
                    f"Loading state dicts completed in {time.time() - start_time} seconds"
                )

                # Average the masks before applying.
                print(f"Averaging {mask_count} masks for block: {blk} ...")
                start_time = time.time()
                for key in masks_dicts_values.keys():
                    masks_dicts_values[key] /= mask_count
                print(f"Averaged state dicts in {time.time() - start_time} seconds")

                # Apply the averaged masks to the model parameters.
                print(f"Applying masks for block: {blk} ...")
                start_time = time.time()
                for name, param in model.named_parameters():
                    if name in masks_dicts_values:
                        if masks_dicts_values[name].shape == param.shape:
                            # Apply the mask values to the parameter.
                            param.data = masks_dicts_values[name].to(param.device)
                        else:
                            print(
                                f"Shape mismatch for {name}: expected {param.shape}, got {masks_dicts_values[name].shape}"
                            )
                print(f"Applying masks completed in {time.time() - start_time} seconds")

                # Delete temporary files and clean up.
                print(f"Deleting temporary files for block: {blk} ...")
                start_time = time.time()
                for file in temp_files:
                    os.remove(file)
                print(f"Deleting files completed in {time.time() - start_time} seconds")

                # Clear caches to free up memory.
                torch.cuda.empty_cache()
                del masks_dicts_values, mask_indices
                print(f"Completed processing for block {blk}")

            # Continue with the rest of your training loop...
            # ...

        except Exception as e:
            print(f"Exception occurred: {e}")
            traceback.print_exc()
            time.sleep(10)  # Wait before retrying to prevent tight loops
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Miner script")
    parser.add_argument("--name", type=str, default=None, help="Optional miner name")
    parser.add_argument(
        "--netuid", type=int, default=212, help="Bittensor network UID."
    )
    parser.add_argument("--bucket", type=str, default="decis", help="S3 bucket name")
    parser.add_argument(
        "--desired_batch_size",
        type=int,
        default=512,
        help="Training batch size per step",
    )
    parser.add_argument(
        "--actual_batch_size",
        type=int,
        default=9,
        help="Training batch size per accumulation.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=4e-4,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--optimizer_beta1", type=float, default=0.9, help="Beta1 for the optimizer"
    )
    parser.add_argument(
        "--optimizer_beta2", type=float, default=0.95, help="Beta2 for the optimizer"
    )
    parser.add_argument(
        "--optimizer_weight_decay",
        type=float,
        default=0.1,
        help="Weight decay for the optimizer",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training (e.g., cpu or cuda)",
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use Weights and Biases for logging"
    )
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    config = bt.config(parser)
    config.subtensor.network = "test"
    config.subtensor.chain_endpoint = "wss://test.finney.opentensor.ai:443/"
    main(config)
