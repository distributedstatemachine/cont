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
import numpy as np
import bittensor as bt
import concurrent.futures  
from tqdm import tqdm
from dotenv import dotenv_values
from typing import Dict, Optional, List, Tuple
from transformers import LlamaForCausalLM 

from hparams import load_hparams
from dataset import SubsetFineWebEdu2Loader

# Instantiate the AWS S3 client.
env_config = {**dotenv_values(".env"), **os.environ}  # Load environment variables.
AWS_ACCESS_KEY_ID = env_config.get('AWS_ACCESS_KEY_ID')  # AWS access key ID.
AWS_SECRET_ACCESS_KEY = env_config.get('AWS_SECRET_ACCESS_KEY')  # AWS secret access key.
CLIENT: boto3.client = boto3.client(
    's3',
    region_name='us-east-1',  # AWS region.
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# Main function that runs the validator script.
def main(config):
    # Print the configuration for debugging.
    print('\n', '=' * 40, 'Config', '=' * 40)
    print(config)

    # Initialize Bittensor wallet, subtensor, and metagraph.
    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(netuid=config.netuid)
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        raise ValueError(f'Wallet {wallet} is not registered on subnet: {metagraph.netuid}')
    my_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    print('\n', '=' * 40, 'Objects', '=' * 40)
    print(f'Wallet: {wallet}\nSubtensor: {subtensor}\nMetagraph: {metagraph}\nUID: {my_uid}')

    # Assert the chain commitment to ensure the validator's bucket is committed on the chain.
    try:
        if config.bucket != subtensor.get_commitment(config.netuid, my_uid):
            raise ValueError(f'Chain commitment does not match: {config.bucket}')
    except Exception:
        # If not committed, commit the bucket to the chain.
        subtensor.commit(wallet, config.netuid, config.bucket)
    print('Bucket:', config.bucket)

    # Initialize Weights and Biases (wandb) for experiment tracking if enabled.
    if config.use_wandb:
        run = wandb.init(project='cont', resume='allow', name=f'V{my_uid}', config=config)
        
    # Init the master model
    hparams = load_hparams()
    model = LlamaForCausalLM(config=hparams.model_config) 
    if not config.restart: 
        try:
            # Load the last master from my bucket.
            print('Loading master state ...')
            start_time = time.time()
            master_filename = f'master-{wallet.hotkey.ss58_address}.pt'
            unique_temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.pt")
            CLIENT.download_file(config.bucket, master_filename, unique_temp_file)
            master_state_dict = torch.load(unique_temp_file, map_location='cpu', weights_only=True)
            model.load_state_dict(master_state_dict)
            print(f'Loading master state completed in {time.time() - start_time} seconds.') 
        except Exception as e:
            raise ValueError("There is no master to continue from. Run with --restart")
    model.to(config.device)
        
    # Start.
    last_n = int(metagraph.n)
    scores = torch.zeros( last_n, dtype = torch.float32 )
    last_mask_sync = subtensor.block
    while True:
        try:
            # Load the latest chain state and reset my connection.
            print('Loading chain state ...')
            start_time = time.time()
            hparams = load_hparams()
            subtensor = bt.subtensor(config=config)
            metagraph = subtensor.metagraph(netuid=config.netuid)
            print(f'Loading chain state completed in {time.time() - start_time} seconds.') 
            
            # Get blocks since my latest sync step.
            print('Getting blocks to sync ...')
            start_time = time.time() 
            block = subtensor.block
            all_sync_blocks = [last_mask_sync + i + 1 for i in range(block - last_mask_sync)]
            last_mask_sync = block
            print(f'Getting blocks to sync completed in {time.time() - start_time} seconds.')
        
            # For each missed block, pull all miner masks and add to state.
            print(f'Downloading masks for blocks: {all_sync_blocks} ...') 
            full_sync_start_time = time.time()
            masks_per_block_per_uid = {}
            mask_count_per_block = {}
            for blk in all_sync_blocks:
                masks_per_block_per_uid[ blk ] = {}
                
                # Get mask file names for all miners for this block.
                print(f'Getting filenames for blk: {blk} ...')
                start_time = time.time()
                if 'buckets' not in locals() or metagraph.n != len( buckets ):
                    buckets = []
                    for uid in metagraph.uids:
                        buckets.append(subtensor.get_commitment(config.netuid, uid))
                mask_filenames = []
                for uid in metagraph.uids:
                    mask_filenames.append((uid, f"mask-{str(metagraph.hotkeys[uid])}-{blk}.pt"))
                print(f'Get filenames completed in {time.time() - start_time} seconds.')
            
                # Download each of the filenames in parallel
                print(f'Downloading mask for blk: {blk} ...')
                start_time = time.time()
                temp_files = []
                n_downloaded = 0
                def download_file(bucket, uid, filename):
                    try:
                        unique_temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.pt")
                        CLIENT.download_file(bucket, filename, unique_temp_file)
                        return (uid, unique_temp_file)
                    except Exception as e:
                        # File does not exist.
                        return None
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(download_file, bucket, uid, filename) for bucket, (uid, filename) in zip(buckets, mask_filenames)]
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        if result:
                            temp_files.append(result)
                            n_downloaded += 1
                print(f'Downloading {n_downloaded} masks completed in {time.time() - start_time} seconds.')
                
                # Break the loop when there is nothing to download.
                if n_downloaded == 0:
                    print (f'Downloaded no valid masks for block: {blk}. Continue ... ')
                    continue
                
                # Init the mask for this block.
                print(f'Creating sync mask for block: {blk} ...')
                mask_indices = {}
                torch.manual_seed(blk)
                start_time = time.time()
                for name, param in model.named_parameters():
                    param = param.to(config.device)
                    next_mask = (torch.rand(param.shape, device=config.device) < (1 / hparams.compression)).float()
                    indices = next_mask.flatten().nonzero(as_tuple=False).flatten()
                    mask_indices[name] = indices
                print(f'Creating sync block mask completed in {time.time() - start_time} seconds.') 
            
                # Load all masks as state dicts and decompress given mask.
                print(f'Loading state dicts for block: {blk} ...')
                start_time = time.time()
                mask_count = 0
                masks_dicts_values = {}
                for uid, file in temp_files:
                    masks_per_block_per_uid[ blk ][ uid ] = {}
                    mask = torch.load(file, map_location='cpu', weights_only=True)
                    mask_count += 1
                    for name in mask.keys():
                        mask_values = mask[name]['values']
                        if torch.isnan(mask_values).any():
                            continue
                        param_shape = model.get_parameter(name).shape
                        indices = mask_indices[name] 
                        decompressed = torch.zeros(param_shape, device='cpu').flatten() 
                        decompressed[indices] = mask_values
                        masks_per_block_per_uid[ blk ][ uid ][ name ] = decompressed.view(param_shape)
                        if name not in masks_dicts_values:
                            masks_dicts_values[name] = decompressed.view(param_shape)
                        else:
                            masks_dicts_values[name] += decompressed.view(param_shape)
                mask_count_per_block[ blk ] = mask_count
                print(f'Loading state dicts completed in {time.time() - start_time} seconds.')
                
                # Average all masks together.
                print(f'Averaging {mask_count} masks for block: {blk} ...')
                start_time = time.time()
                for key in masks_dicts_values.keys():
                    masks_dicts_values[key] /= mask_count
                print(f'Averaged state dicts in {time.time() - start_time} seconds.')
                
                # Apply averages to my local model state.
                print(f'Applying {mask_count} masks for block: {blk} ...')
                start_time = time.time()  # Start timing
                for name, param in model.named_parameters():
                    indices = mask_indices[name]
                    if name in masks_dicts_values:
                        if masks_dicts_values[name].shape == param.shape:
                            # Apply the mask values to the flattened param data.
                            on_device = masks_dicts_values[name].to(model.device).flatten()
                            param_flat = param.data.flatten()
                            param_flat[indices] = on_device[indices]
                            param.data.copy_(param_flat.view(param.shape))
                            del on_device, param_flat
                        else:
                            print(f"Shape mismatch for {name}: expected {param.shape}, got {masks_dicts_values[name].shape}")
                del masks_dicts_values
                print(f'Applying {mask_count} masks completed in {time.time() - start_time} seconds.')  # Print timing after this step
                
                # Delete files and clean state.
                print(f'Deleting files for block: {blk} ...')
                start_time = time.time()
                for uid, file in temp_files:
                    os.remove(file)
                print(f'Deleting files completed in {time.time() - start_time} seconds')
                
            # Finish syncing masks.
            torch.cuda.empty_cache()
            print(f'Finished downloading masks for blocks: {all_sync_blocks} in {time.time() - full_sync_start_time} seconds.')
            
            # Upload a full copy of the model weights to master
            print('Uploading master ...')
            start_time = time.time()
            model_state_dict = model.state_dict()
            upload_filename = f'master-{wallet.hotkey.ss58_address}.pt'
            with io.BytesIO() as module_buffer:
                torch.save(model_state_dict, module_buffer)
                module_buffer.seek(0)  # Reset the buffer's position to the beginning.
                CLIENT.upload_fileobj(module_buffer, config.bucket, upload_filename)
            CLIENT.put_object_acl(
                Bucket=config.bucket,
                Key=upload_filename,
                GrantRead='uri="http://acs.amazonaws.com/groups/global/AllUsers"',
                GrantReadACP='uri="http://acs.amazonaws.com/groups/global/AllUsers"'
            )
            print(f'Uploading master completed in {time.time() - start_time} seconds.')
        
            block_to_eval = None
            if len(list(masks_per_block_per_uid.keys())) == 0:
                print ('No masks to eval. Continue ...')
                continue # Nothing to do.
            block_to_eval = max(list(masks_per_block_per_uid.keys()))
            if len(list(masks_per_block_per_uid[block_to_eval].keys())) == 0:
                print ('No masks to eval. Continue ...')
                continue # Nothing to do.
            
            # Eval this UID.
            uid_to_eval = random.choice(list(masks_per_block_per_uid[block_to_eval].keys()))
            mask_count = mask_count_per_block[block_to_eval]
            mask = masks_per_block_per_uid[block_to_eval][uid_to_eval]
            del masks_per_block_per_uid
            
            # Get the pages for this block and my_uid.
            # This is global and deterministic
            print('Loading page for eval ...')
            start_time = time.time()  # Start timing
            pages = SubsetFineWebEdu2Loader.next_pages(
                offset=block_to_eval,
                n_pages=3,
                seed=uid_to_eval
            )
            dataset = SubsetFineWebEdu2Loader(
                batch_size=config.batch_size,
                sequence_length=hparams.sequence_length,
                pages_info=pages,
                tokenizer=hparams.tokenizer
            )
            print(f'Loading page for eval completed in {time.time() - start_time} seconds.')
            
            # Remove the mask from the model.
            print('Removing the mask ...')
            start_time = time.time()
            for name, param in model.named_parameters():
                param.data.sub_(mask[name].to(model.device) / mask_count)
            print(f'Removing the mask completed in {time.time() - start_time} seconds.')
            
            # Evaluate my model on the current page.
            print('Evaluating without mask ...')
            start_time = time.time()
            model.eval()
            without_avg_loss = 0
            for idx, batch in enumerate(tqdm(dataset)):
                input_ids = torch.tensor(batch, dtype=torch.long).to(model.device)
                labels = input_ids.clone()
                labels = torch.where(labels == hparams.tokenizer.pad_token_id, -100, labels)
                with torch.no_grad(): # Turn of grad calculation.
                    with torch.cuda.amp.autocast():  # Enable autocasting for mixed precision
                        outputs = model(input_ids=input_ids, labels=labels)
                without_avg_loss += outputs.loss.item()
            without_avg_loss /= (idx + 1)
            del input_ids, labels, outputs
            torch.cuda.empty_cache()
            print(f'Evaluating without mask completed in {time.time() - start_time} seconds.')
            
            # Add the mask back to the model.
            print('Adding the mask ...')
            start_time = time.time()
            for name, param in model.named_parameters():
                param.data.add_(mask[name].to(model.device) / mask_count)
            print(f'Adding the mask completed in {time.time() - start_time} seconds.')
            
            # Evaluate my model on the current page.
            print('Evaluating with mask ...')
            start_time = time.time()
            model.eval()
            with_avg_loss = 0
            for idx, batch in enumerate(tqdm(dataset)):
                input_ids = torch.tensor(batch, dtype=torch.long).to(model.device)
                labels = input_ids.clone()
                labels = torch.where(labels == hparams.tokenizer.pad_token_id, -100, labels)
                with torch.no_grad(): # Turn of grad calculation.
                    with torch.cuda.amp.autocast():  # Enable autocasting for mixed precision
                        outputs = model(input_ids=input_ids, labels=labels)
                with_avg_loss += outputs.loss.item()
            with_avg_loss /= (idx + 1)
            del input_ids, labels, outputs
            torch.cuda.empty_cache()
            print(f'Evaluating with mask completed in {time.time() - start_time} seconds.')
            
            # Compute the miner score for their mask.
            print('Computing weights ...')
            start_time = time.time()
            if len(scores) < int(metagraph.n):
                scores = torch.concat([scores, torch.zeros( int(metagraph.n) - len(scores), dtype=torch.float32)])
            scores[uid_to_eval] = 0.1 * (with_avg_loss - without_avg_loss) + 0.9 * scores[uid_to_eval]
            # Compute the weights
            non_zero_indices = scores.nonzero(as_tuple=False).flatten()
            weights = torch.zeros_like(scores, dtype=torch.float32)
            weights[non_zero_indices] = torch.softmax(scores[non_zero_indices], dim=0)
            print ('scores', scores.tolist())
            print ('weights', weights.tolist())
            print(f'Computing weights completed in {time.time() - start_time} seconds.')

            # Set weights on chain based on moving scores.
            print('Settting weights on chain ...')
            start_time = time.time()
            subtensor.set_weights(
                wallet=wallet,
                netuid=config.netuid,
                uids=metagraph.uids.tolist(),
                weights=weights.tolist(),
                wait_for_inclusion=False,
                wait_for_finalization=False,
            )
            print(f'Settting weights on chain completed in {time.time() - start_time} seconds.')
                 
        # Handle keyboard interrupts to allow graceful shutdown.
        except (KeyboardInterrupt, SystemExit):
            break

        # Handle any other exceptions, log the error, clean up, and continue.
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            time.sleep(5)  # Wait for a short period before retrying.
            continue

# Entry point of the script.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validator script')
    parser.add_argument('--name', type=str, default=None, help='Optional name')
    parser.add_argument('--bucket', type=str, default='decis', help='S3 bucket name')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for eval.')
    parser.add_argument('--netuid', type=int, default=212, help='Bittensor network uid.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., cpu or cuda)')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
    parser.add_argument('--restart', action='store_true', help='Restart all evaluation history')
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    config = bt.config(parser)
    config.subtensor.chain_endpoint = 'wss://test.finney.opentensor.ai:443/'
    main(config)
