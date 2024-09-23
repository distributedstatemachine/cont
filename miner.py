# miner.py

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
from typing import List, Tuple, Dict
from dotenv import dotenv_values
from transformers import LlamaForCausalLM
from torch.optim.lr_scheduler import CosineAnnealingLR
import multiprocessing

from hparams import load_hparams
from dataset import SubsetFineWebEdu2Loader

# Enable cuDNN benchmark for optimized performance
torch.backends.cudnn.benchmark = True

# Load AWS environment variables
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
        prefix: str = f"mask-{hotkey}-{block_number}"
        response = CLIENT.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if "Contents" in response:
            return [(bucket, obj["Key"]) for obj in response["Contents"]]
        else:
            return []
    except Exception as e:
        # TODO: Handle specific exceptions and possibly log them
        return []


def download_file(bucket: str, key: str) -> str:
    """
    Downloads a file from S3 to a temporary file.

    Parameters:
    - bucket (str): The S3 bucket name.
    - key (str): The key (path) to the file in S3.

    Returns:
    - str: The path to the downloaded temporary file, or None if download failed.
    """
    try:
        if bucket is None:
            return None
        unique_temp_file: str = os.path.join(
            tempfile.gettempdir(), f"{uuid.uuid4()}.pt"
        )
        CLIENT.download_file(bucket, key, unique_temp_file)
        return unique_temp_file
    except Exception as e:
        # TODO: Handle specific exceptions and possibly log them
        return None


def main(config):
    """
    The main function for the miner script.

    Parameters:
    - config: Configuration object with all necessary parameters.
    """
    # Print the configuration settings.
    print("\n", "-" * 40, "Config", "-" * 40)
    print(config)

    # Initialize Bittensor objects.
    wallet: bt.wallet = bt.wallet(config=config)
    subtensor: bt.subtensor = bt.subtensor(config=config)
    metagraph: bt.metagraph = subtensor.metagraph(netuid=config.netuid)
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        raise ValueError(
            f"Wallet {wallet.hotkey.ss58_address} is not registered on subnet: {metagraph.netuid}"
        )
    my_uid: int = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    print("\n", "-" * 40, "Objects", "-" * 40)
    print(
        f"Wallet: {wallet.name}\nSubtensor: {subtensor.network}\nMetagraph UID: {my_uid}"
    )

    # Initialize my bucket information by submitting it to the chain.
    try:
        chain_bucket: str = subtensor.get_commitment(config.netuid, my_uid)
        if config.bucket != chain_bucket:
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
    optimizer = None
    scaler = None
    scheduler = None
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
                    master_uid: int = int(metagraph.S.argmax())
                    master_bucket: str = subtensor.get_commitment(
                        config.netuid, master_uid
                    )
                    master_hotkey: str = metagraph.hotkeys[master_uid]
                    master_filename: str = f"master-{master_hotkey}.pt"
                    unique_temp_file: str = os.path.join(
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
                    scaler = torch.cuda.amp.GradScaler()
                    scheduler = CosineAnnealingLR(
                        optimizer,
                        T_max=hparams.epoch_length,
                        eta_min=4e-5,
                        last_epoch=-1,
                    )
                    last_master_sync = subtensor.block
                    last_mask_sync = last_master_sync
                except Exception as e:
                    print(f"No master model found: {e}\nWaiting for master model ...")
                    time.sleep(12)
                    continue
            print(
                f"Checking epoch sync completed in {time.time() - start_time} seconds"
            )

            # Get the current block and update mask sync.
            print(f"Getting block state ...")
            start_time = time.time()
            current_block: int = subtensor.block
            all_sync_blocks: List[int] = list(
                range(last_mask_sync - 2, current_block + 1)
            )
            last_mask_sync = current_block
            print(f"Getting block completed in {time.time() - start_time} seconds")

            # Get buckets per UID if needs update.
            if "buckets" not in locals() or len(buckets) != len(metagraph.uids):
                buckets: List[str] = []
                for uid in metagraph.uids:
                    try:
                        buckets.append(subtensor.get_commitment(config.netuid, uid))
                    except Exception:
                        buckets.append(None)

            # Prepare UID, bucket, and hotkey information.
            uids_info: List[Tuple[int, str, str]] = [
                (uid, bucket, metagraph.hotkeys[uid])
                for uid, bucket in zip(metagraph.uids.tolist(), buckets)
            ]

            # Download and apply masks for each block.
            print(f"Downloading masks for blocks: {all_sync_blocks}")
            full_sync_start_time = time.time()
            for blk in all_sync_blocks:
                # Generate mask indices based on the block number.
                print(f"Creating sync mask for block: {blk} ...")
                mask_indices: Dict[str, torch.Tensor] = {}
                torch.manual_seed(blk)
                start_time = time.time()
                for name, param in model.named_parameters():
                    # Generate mask indices for this block.
                    param = param.to(config.device)
                    next_mask: torch.Tensor = (
                        torch.rand(param.shape, device=config.device)
                        < (1 / hparams.compression)
                    ).float()
                    indices = next_mask.flatten().nonzero(as_tuple=False).flatten()
                    mask_indices[name] = indices.cpu()
                print(
                    f"Creating sync block mask completed in {time.time() - start_time:.2f} seconds"
                )

                # List masks available for this block.
                print(f"Getting filenames for block: {blk} ...")
                mask_tasks: List[Tuple[str, str]] = []
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
                temp_files: List[str] = []
                n_downloaded: int = 0
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
                mask_count: int = 0
                masks_dicts_values: Dict[str, torch.Tensor] = {}
                for file in temp_files:
                    try:
                        mask = torch.load(file, map_location="cpu", weights_only=True)
                        mask_count += 1
                        for name in mask.keys():
                            mask_values: torch.Tensor = mask[name]["values"]
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
                    f"Loading state dicts completed in {time.time() - start_time:.2f} seconds"
                )

                # Average the masks before applying.
                print(f"Averaging {mask_count} masks for block: {blk} ...")
                start_time = time.time()
                for key in masks_dicts_values.keys():
                    masks_dicts_values[key] /= mask_count
                print(f"Averaged state dicts in {time.time() - start_time:.2f} seconds")

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
                print(
                    f"Applying masks completed in {time.time() - start_time:.2f} seconds"
                )

                # Delete temporary files and clean up.
                print(f"Deleting temporary files for block: {blk} ...")
                start_time = time.time()
                for file in temp_files:
                    os.remove(file)
                print(
                    f"Deleting files completed in {time.time() - start_time:.2f} seconds"
                )

                # Clear caches to free up memory.
                torch.cuda.empty_cache()
                del masks_dicts_values, mask_indices
                print(f"Completed processing for block {blk}")

            # Print completion of mask downloading.
            torch.cuda.empty_cache()
            print(
                f"Downloading masks for blocks: {all_sync_blocks} completed in {time.time() - full_sync_start_time:.2f} seconds"
            )

            # Get the pages for this block and my_uid.
            # This is global and deterministic.
            n_pages: int = max(1, int(config.desired_batch_size * 0.01))
            print(f"Loading {n_pages} pages ...")
            start_time = time.time()
            pages = SubsetFineWebEdu2Loader.next_pages(
                offset=subtensor.block + n_pages, n_pages=n_pages, seed=my_uid
            )
            dataset = SubsetFineWebEdu2Loader(
                batch_size=config.actual_batch_size,
                sequence_length=hparams.sequence_length,
                pages_info=pages,
                tokenizer=hparams.tokenizer,
            )
            print(
                f"Loading {n_pages} pages completed in {time.time() - start_time:.2f} seconds"
            )

            # Train the model on the current pages.
            print("Starting training loop...")
            torch.cuda.empty_cache()
            start_time = time.time()
            optimizer.zero_grad()
            total_loss: float = 0.0
            total_steps: int = config.desired_batch_size // config.actual_batch_size
            progress_bar = tqdm(total=total_steps, desc="Training:")
            for idx, batch in enumerate(dataset):
                input_ids = torch.tensor(batch, dtype=torch.long).to(model.device)
                labels = input_ids.clone()
                labels = torch.where(
                    labels == hparams.tokenizer.pad_token_id, -100, labels
                )
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss / total_steps  # Normalize loss
                scaler.scale(loss).backward()
                progress_bar.update(1)
                if idx >= total_steps - 1:
                    break
            progress_bar.close()

            # Update parameters.
            try:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            except AssertionError as e:
                print(f"An error occurred during the optimizer step: {e}")

            # Clean up.
            del input_ids, labels, outputs
            torch.cuda.empty_cache()

            # Calculate and print average loss.
            average_loss = total_loss / total_steps
            print(
                f"Loss: {average_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}"
            )
            if config.use_wandb:
                wandb.log(
                    {
                        "step_loss": average_loss,
                        "learning_rate": scheduler.get_last_lr()[0],
                        f"incentive_{my_uid}": float(metagraph.I[my_uid]),
                    }
                )
            total_time = time.time() - start_time
            print(f"Training completed in {total_time:.2f} seconds")
            print(f"Steps per second: {total_steps / total_time:.2f}")
            print(
                f"Batches per second: {config.actual_batch_size * total_steps / total_time:.2f}"
            )
            print(
                f"Tokens per second: {hparams.sequence_length * config.actual_batch_size * total_steps / total_time:.2f}"
            )

            # Select the block to produce a mask for.
            next_upload_block: int = subtensor.block

            # Create the upload mask based on the next upload block.
            print(f"Creating upload mask ...")
            start_time = time.time()
            upload_mask: Dict[str, torch.Tensor] = {}
            torch.manual_seed(next_upload_block)
            for name, param in model.named_parameters():
                param = param.to(config.device)
                next_mask = (
                    torch.rand(param.shape, device=config.device)
                    < (1 / hparams.compression)
                ).float()
                upload_mask[name] = next_mask.to("cpu")
            print(
                f"Creating upload block mask completed in {time.time() - start_time:.2f} seconds"
            )

            # Mask the model values given the mask and produce a state dict.
            print("Applying upload mask to model ...")
            model_state_dict: Dict[str, Dict[str, torch.Tensor]] = {}
            for name, param in model.named_parameters():
                param_mask = upload_mask[name].to(param.device)
                param_flat = param.flatten()
                mask_flat = param_mask.flatten()
                unmasked_indices = mask_flat.nonzero(as_tuple=False).flatten()
                unmasked_params = param_flat[unmasked_indices]
                model_state_dict[name] = {"values": unmasked_params.cpu()}
                del unmasked_indices
            del upload_mask
            print(
                f"Applied mask to model completed in {time.time() - start_time:.2f} seconds"
            )

            # Upload the masked state dict.
            print("Uploading mask ...")
            start_time = time.time()
            upload_filename: str = (
                f"mask-{wallet.hotkey.ss58_address}-{next_upload_block}.pt"
            )
            with io.BytesIO() as module_buffer:
                torch.save(model_state_dict, module_buffer)
                module_buffer.seek(0)
                CLIENT.upload_fileobj(module_buffer, config.bucket, upload_filename)
            CLIENT.put_object_acl(
                Bucket=config.bucket,
                Key=upload_filename,
                GrantRead='uri="http://acs.amazonaws.com/groups/global/AllUsers"',
                GrantReadACP='uri="http://acs.amazonaws.com/groups/global/AllUsers"',
            )
            upload_history.append(upload_filename)
            print(
                f"Uploading mask to: {upload_filename} completed in {time.time() - start_time:.2f} seconds"
            )

            # Delete old mask files and clean up.
            print("Deleting old mask files ...")
            start_time = time.time()
            if len(upload_history) > 5:
                to_delete = upload_history.pop(0)
                CLIENT.delete_object(Bucket=config.bucket, Key=to_delete)
            print(
                f"Deleting old mask files completed in {time.time() - start_time:.2f} seconds"
            )

        # Handle keyboard interrupts to allow graceful shutdown.
        except (KeyboardInterrupt, SystemExit):
            print("Training interrupted. Exiting gracefully.")
            break

        # Handle any other exceptions, log the error, and continue after a short delay.
        except Exception as e:
            print(f"Exception occurred: {e}")
            traceback.print_exc()
            time.sleep(5)
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
    # Set default network and chain endpoint if not provided in config
    if not hasattr(config.subtensor, "network"):
        config.subtensor.network = "test"
    if not hasattr(config.subtensor, "chain_endpoint"):
        config.subtensor.chain_endpoint = "wss://test.finney.opentensor.ai:443/"
    main(config)
