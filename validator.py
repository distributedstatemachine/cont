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
import boto3
import torch
import wandb
import random
import tempfile
import argparse
import traceback
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import bittensor as bt
import concurrent.futures
from tqdm import tqdm
from dotenv import dotenv_values
from typing import Dict, Optional, List, Tuple
from transformers import LlamaForCausalLM
from bittensor import Metagraph

from hparams import load_hparams
from dataset import SubsetFineWebEdu2Loader

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


# Main function that runs the validator script.
def main(config):
    # Print the configuration for debugging.
    print("\n", "=" * 40, "Config", "=" * 40)
    print(config)

    # Initialize Bittensor wallet, subtensor, and metagraph.
    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(netuid=config.netuid)
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        raise ValueError(
            f"Wallet {wallet} is not registered on subnet: {metagraph.netuid}"
        )
    my_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    print("\n", "=" * 40, "Objects", "=" * 40)
    print(
        f"Wallet: {wallet}\nSubtensor: {subtensor}\nMetagraph: {metagraph}\nUID: {my_uid}"
    )

    # Assert the chain commitment to ensure the validator's bucket is committed on the chain.
    try:
        if config.bucket != subtensor.get_commitment(config.netuid, my_uid):
            raise ValueError(f"Chain commitment does not match: {config.bucket}")
    except Exception:
        # If not committed, commit the bucket to the chain.
        subtensor.commit(wallet, config.netuid, config.bucket)
    print("Bucket:", config.bucket)

    # Initialize Weights and Biases (wandb) for experiment tracking if enabled.
    if config.use_wandb:
        run = wandb.init(
            project="cont", resume="allow", name=f"V{my_uid}", config=config
        )

    # Init the master model
    hparams = load_hparams()
    model = LlamaForCausalLM(config=hparams.model_config)
    if not config.restart:
        try:
            # Load the last master from my bucket.
            print("Loading master state ...")
            start_time = time.time()
            master_filename = f"master-{wallet.hotkey.ss58_address}.pt"
            unique_temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.pt")
            CLIENT.download_file(config.bucket, master_filename, unique_temp_file)
            master_state_dict = torch.load(
                unique_temp_file, map_location="cpu", weights_only=True
            )
            model.load_state_dict(master_state_dict)
            print(
                f"Loading master state completed in {time.time() - start_time} seconds."
            )
        except Exception as e:
            raise ValueError("There is no master to continue from. Run with --restart")
    model.to(config.device)
    # @formalised : do we need this ?
    model.eval()  # Set the model to evaluation mode.

    # Define the section (epoch) length in terms of blocks.
    SECTION_LENGTH: int = hparams.section_length  # Number of blocks per section.

    # Initialize variables for tracking.
    last_section: int = -1  # Track the last completed section.

    # Start.
    last_n = int(metagraph.n)
    scores = torch.zeros(last_n, dtype=torch.float32)
    while True:
        try:
            # Sync the current chain state.
            print("Loading chain state...")
            start_time = time.time()
            subtensor = bt.subtensor(
                config=config
            )  # Re-instantiate to update block height.
            metagraph = subtensor.metagraph(netuid=config.netuid)
            current_block: int = subtensor.block
            print(
                f"Loading chain state completed in {time.time() - start_time:.2f} seconds"
            )

            # Calculate the current section based on the block number.
            current_section: int = current_block // SECTION_LENGTH
            print(f"Current section: {current_section}")

            # Check if we need to process masks for a new section.
            if last_section != current_section:
                # Process masks for the last completed section.
                if last_section != -1:
                    print(f"Processing masks for section {last_section}")
                    process_section_masks(
                        model=model,
                        section_number=last_section,
                        metagraph=metagraph,
                        config=config,
                        hparams=hparams,
                        my_uid=my_uid,
                        scores=scores,
                        wallet=wallet,
                    )
                last_section = current_section  # Update the last_section tracker.

            time.sleep(10)  # Wait for a short period before checking again.

        except (KeyboardInterrupt, SystemExit):
            print("Validation interrupted. Exiting gracefully.")
            break

        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            time.sleep(5)  # Wait for a short period before retrying.
            continue


def process_section_masks(
    model: LlamaForCausalLM,
    section_number: int,
    metagraph: Metagraph,
    config: bt.Config,
    hparams: Any,
    my_uid: int,
    scores: torch.Tensor,
    wallet: bt.Wallet,
) -> None:
    """
    Processes and applies masks for a given section, evaluates miners, and updates weights.

    Args:
        model (LlamaForCausalLM): The master model to update.
        section_number (int): The section number for which to process masks.
        metagraph (Metagraph): The current metagraph.
        config (bt.Config): Configuration object.
        hparams (Any): Hyperparameters.
        my_uid (int): The UID of the validator.
        scores (torch.Tensor): Tensor containing scores for miners.
        wallet (bt.Wallet): Validator's wallet.

    Returns:
        None
    """
    print(f"Starting mask processing for section {section_number}")
    start_time = time.time()

    # Generate the mask indices based on the section seed.
    print("Generating mask indices...")
    torch.manual_seed(section_number)
    mask_indices: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        mask: torch.Tensor = (
            torch.rand(param.shape, device="cpu") < (1 / hparams.compression)
        ).float()
        indices: torch.Tensor = mask.view(-1).nonzero(as_tuple=False).view(-1)
        mask_indices[name] = indices
    print("Mask indices generated.")

    # Download masks from all miners for the section.
    print("Downloading masks from miners...")
    mask_values_sum: Dict[str, torch.Tensor] = {}
    mask_count: int = 0

    # Prepare filenames.
    mask_filenames: List[Tuple[int, str]] = []
    for uid in metagraph.uids:
        hotkey_address: str = metagraph.hotkeys[uid]
        filename: str = f"mask-{hotkey_address}-section{section_number}.pt"
        mask_filenames.append((uid, filename))

    # Download and aggregate masks.
    masks_per_uid: Dict[int, Dict[str, torch.Tensor]] = {}
    for uid, filename in mask_filenames:
        try:
            unique_temp_file: str = os.path.join(
                tempfile.gettempdir(), f"{uuid.uuid4()}.pt"
            )
            CLIENT.download_file(config.bucket, filename, unique_temp_file)
            mask_state_dict: Dict[str, Any] = torch.load(
                unique_temp_file, map_location="cpu"
            )
            os.remove(unique_temp_file)

            # Store individual masks for evaluation.
            masks_per_uid[uid] = mask_state_dict

            # Aggregate mask values.
            for name in mask_state_dict.keys():
                if name not in mask_values_sum:
                    mask_values_sum[name] = mask_state_dict[name]["values"].clone()
                else:
                    mask_values_sum[name] += mask_state_dict[name]["values"]
            mask_count += 1
        except Exception as e:
            print(f"Failed to download mask {filename}: {e}")

    if mask_count == 0:
        print("No masks downloaded for this section. Skipping.")
        return

    # Average the mask values.
    print("Averaging mask values...")
    for name in mask_values_sum.keys():
        mask_values_sum[name] /= mask_count

    # Apply the averaged mask to the model.
    print("Applying averaged mask to the model...")
    for name, param in model.named_parameters():
        indices: torch.Tensor = mask_indices[name]
        param_flat: torch.Tensor = param.data.view(-1)
        averaged_values: torch.Tensor = mask_values_sum[name].to(param.device)
        param_flat[indices] = averaged_values
        param.data = param_flat.view(param.shape)

    # Clean up.
    torch.cuda.empty_cache()
    print(
        f"Mask processing for section {section_number} completed in {time.time() - start_time:.2f} seconds"
    )

    # Evaluate miners by measuring performance with and without individual masks.
    print("Evaluating miners...")
    evaluate_miners(
        model=model,
        masks_per_uid=masks_per_uid,
        mask_indices=mask_indices,
        metagraph=metagraph,
        config=config,
        hparams=hparams,
        scores=scores,
        wallet=wallet,
    )

    # Upload the updated master model to S3.
    print("Uploading updated master model...")
    upload_master_model(model, config, wallet_hotkey=metagraph.hotkeys[my_uid])
    print("Master model uploaded successfully.")


def upload_master_model(
    model: LlamaForCausalLM, config: bt.Config, wallet_hotkey: str
) -> None:
    """
    Uploads the updated master model to the S3 bucket.

    Args:
        model (LlamaForCausalLM): The master model to upload.
        config (bt.Config): Configuration object.
        wallet_hotkey (str): Wallet hotkey address.

    Returns:
        None
    """
    master_filename: str = f"master-{wallet_hotkey}.pt"
    with io.BytesIO() as buffer:
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        CLIENT.upload_fileobj(buffer, config.bucket, master_filename)
    CLIENT.put_object_acl(Bucket=config.bucket, Key=master_filename, ACL="public-read")


def evaluate_model(
    model: LlamaForCausalLM, dataset: SubsetFineWebEdu2Loader, hparams: Any
) -> float:
    """
    Evaluates the model on the given dataset and returns the average loss.

    Args:
        model (LlamaForCausalLM): The model to evaluate.
        dataset (SubsetFineWebEdu2Loader): The dataset for evaluation.
        hparams (Any): Hyperparameters.

    Returns:
        float: The average loss over the dataset.
    """
    model.eval()
    total_loss = 0.0
    total_steps = 0
    with torch.no_grad():
        for batch in tqdm(dataset):
            input_ids = torch.tensor(batch, dtype=torch.long).to(model.device)
            labels = input_ids.clone()
            labels = torch.where(labels == hparams.tokenizer.pad_token_id, -100, labels)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss.item()
            total_loss += loss
            total_steps += 1
            del input_ids, labels, outputs  # Clean up
            torch.cuda.empty_cache()
    average_loss = total_loss / total_steps
    return average_loss


def evaluate_miners(
    model: LlamaForCausalLM,
    masks_per_uid: Dict[int, Dict[str, torch.Tensor]],
    mask_indices: Dict[str, torch.Tensor],
    metagraph: Metagraph,
    config: bt.Config,
    hparams: Any,
    scores: torch.Tensor,
    wallet: bt.Wallet,
) -> None:
    """
    Evaluates miners by measuring model performance with and without their masks.

    Args:
        model (LlamaForCausalLM): The master model.
        masks_per_uid (Dict[int, Dict[str, torch.Tensor]]): Individual masks per miner.
        mask_indices (Dict[str, torch.Tensor]): Indices of the mask values.
        metagraph (Metagraph): The current metagraph.
        config (bt.Config): Configuration object.
        hparams (Any): Hyperparameters.
        scores (torch.Tensor): Tensor containing scores for miners.
        wallet (bt.Wallet): Validator's wallet.

    Returns:
        None
    """
    # Select a random miner to evaluate.
    uid_to_eval: int = random.choice(list(masks_per_uid.keys()))
    mask_to_eval: Dict[str, torch.Tensor] = masks_per_uid[uid_to_eval]
    print(f"Evaluating miner UID: {uid_to_eval}")

    # Prepare evaluation dataset.
    print("Loading evaluation dataset...")
    start_time = time.time()
    pages = SubsetFineWebEdu2Loader.next_pages(
        offset=0, n_pages=3, seed=uid_to_eval  # Adjust offset as needed
    )
    dataset = SubsetFineWebEdu2Loader(
        batch_size=config.batch_size,
        sequence_length=hparams.sequence_length,
        pages_info=pages,
        tokenizer=hparams.tokenizer,
    )
    print(f"Evaluation dataset loaded in {time.time() - start_time:.2f} seconds.")

    # Remove the miner's mask from the model.
    print("Subtracting miner's mask from the model...")
    for name, param in model.named_parameters():
        indices: torch.Tensor = mask_indices[name]
        param_flat: torch.Tensor = param.data.view(-1)
        mask_values: torch.Tensor = mask_to_eval[name]["values"].to(param.device)
        param_flat[indices] -= mask_values
        param.data = param_flat.view(param.shape)

    # Evaluate model without the miner's mask.
    print("Evaluating without miner's mask...")
    without_loss = evaluate_model(model, dataset, hparams)

    # Add the miner's mask back to the model.
    print("Adding miner's mask back to the model...")
    for name, param in model.named_parameters():
        indices: torch.Tensor = mask_indices[name]
        param_flat: torch.Tensor = param.data.view(-1)
        mask_values: torch.Tensor = mask_to_eval[name]["values"].to(param.device)
        param_flat[indices] += mask_values
        param.data = param_flat.view(param.shape)

    # Evaluate model with the miner's mask.
    print("Evaluating with miner's mask...")
    with_loss = evaluate_model(model, dataset, hparams)

    # Compute the miner's score based on loss difference.
    print("Computing miner's score...")
    score = with_loss - without_loss
    scores[uid_to_eval] = 0.1 * score + 0.9 * scores[uid_to_eval]

    # Compute weights based on scores.
    print("Computing weights...")
    non_zero_indices = scores.nonzero(as_tuple=False).flatten()
    weights = torch.zeros_like(scores, dtype=torch.float32)
    weights[non_zero_indices] = torch.softmax(scores[non_zero_indices], dim=0)
    print("Scores:", scores.tolist())
    print("Weights:", weights.tolist())

    # Set weights on chain based on scores.
    print("Setting weights on chain...")
    subtensor = bt.subtensor(config=config)
    subtensor.set_weights(
        wallet=wallet,
        netuid=config.netuid,
        uids=metagraph.uids.tolist(),
        weights=weights.tolist(),
        wait_for_inclusion=False,
        wait_for_finalization=False,
    )


# Entry point of the script.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validator script")
    parser.add_argument("--name", type=str, default=None, help="Optional name")
    parser.add_argument("--bucket", type=str, default="decis", help="S3 bucket name")
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for eval."
    )
    parser.add_argument(
        "--netuid", type=int, default=218, help="Bittensor network uid."
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
    parser.add_argument(
        "--restart", action="store_true", help="Restart all evaluation history"
    )
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    config = bt.config(parser)
    config.subtensor.chain_endpoint = "wss://test.finney.opentensor.ai:443/"
    main(config)
