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

import os
import copy
import math
import time
import torch
import wandb
import random
import argparse
import bittensor as bt
import numpy as np
from types import SimpleNamespace
from transformers import Adafactor
from typing import List, Tuple
from transformers import LlamaForCausalLM 

# Import constants and utility functions specific to the project.
from common import *
from dataset import SubsetFineWebEdu2Loader

def main(config):
    """
    Main function for the miner script.

    This function initializes the model, sets up training parameters, and
    enters the main training loop where the model is trained and periodically
    uploaded to the S3 bucket for validation.

    Args:
        config: The configuration object containing training parameters and settings.
    """
    # Print the configuration settings.
    print('\n', '-' * 40, 'Config', '-' * 40)
    print(config)
    
    # Initialize Bittensor wallet with the provided configuration.
    wallet = bt.wallet(config=config)
    # Initialize Bittensor subtensor with the provided configuration.
    subtensor = bt.subtensor(config=config)
    # Retrieve the metagraph for the specified netuid.
    metagraph = subtensor.metagraph(netuid=config.netuid)
    
    # Ensure the wallet's hotkey is registered on the subnet.
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        raise ValueError(f'Wallet {wallet} is not registered on subnet: {metagraph.netuid}')
    
    # Get the UID (unique identifier) of the wallet's hotkey in the metagraph.
    my_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    
    # Print initialized objects for debugging purposes.
    print('\n', '-' * 40, 'Objects', '-' * 40)
    print(f'Wallet: {wallet}\nSubtensor: {subtensor}\nMetagraph: {metagraph}\nUID: {my_uid}')
    
    # Check if the bucket committed on-chain matches the configured bucket.
    try:
        if config.bucket != subtensor.get_commitment(config.netuid, my_uid):
            raise ValueError(f'Chain commitment does not match: {config.bucket}')
    except Exception:
        # If not committed or mismatch, commit the bucket to the chain.
        subtensor.commit(wallet, config.netuid, config.bucket)
    print('Bucket:', config.bucket)

    # Initialize Weights and Biases (wandb) for experiment tracking if enabled.
    if config.use_wandb:
        run = wandb.init(project='cont', resume='allow', name=f'M{my_uid}', config=config)
    
    # Train loop.       
    current_master_meta = None
    upload_history = []  # List of previous uploads
    update_block = subtensor.block
    while True:
        try:
    
            # Resynchronize the chain state to get the latest metagraph.
            subtensor = bt.subtensor(config=config)
            metagraph = subtensor.metagraph(netuid=config.netuid)
            
            # Sync the latest hparams from the global training state gist.
            hparams = load_hparams()
            
            # Get the master.
            latest_master_meta = get_latest_metadata( key = 'model', uid = int(metagraph.S.argmax()), metagraph = metagraph, subtensor = subtensor, CLIENT=CLIENT)
            if latest_master_meta is None:
                print ('No Valid master waiting ...')
                time.sleep(12)
                continue
                
            # Update the master model
            if current_master_meta is None or latest_master_meta.model_hash != current_master_meta.model_hash:
                print ('Loading the new master...')
                # We can update the model by loading the delta.
                applied_delta = False
                
                # Attempt to load the master model via the delta.
                if hasattr( latest_master_meta, 'delta' ) and current_master_meta is not None and model is not None:
                    print ('Applying delta....')
                    try:
                        # Apply delta from the latest master if exists.
                        delta_meta = SimpleNamespace( **latest_master_meta.delta ) 
                        delta = download_model( metadata = delta_meta, device = 'cpu', CLIENT = CLIENT )
                        for (name, model_param), (_, delta_param) in zip( model.named_parameters(), delta.named_parameters() ):
                            model_param.data.add_( delta_param.data.to( model.device ) )
                            delta_param.data = delta_param.data.cpu() # Remove from GPU here.
                        master = copy.deepcopy( model ).to('cpu')
                        torch.cuda.empty_cache()
                        applied_delta = True
                        del delta
                        print ('Successfully applied delta.')
                    except Exception as e:
                        print ( f'Failed to apply delta with error:{e}' )
                        applied_delta = False
                        
                # If we failed to apply the delta to update load the state from master.
                # Or if the hashes are not correct.
                if applied_delta == False or hash_model( master ) != latest_master_meta.model_hash:
                    print ('Loading master model directly.')
                    # Otherwise just get the master directly.
                    master = download_model( metadata = latest_master_meta, device='cpu', CLIENT = CLIENT )    
                    if master == None:
                        print ('Waiting for master...')
                        time.sleep(12)
                        continue
                    model = copy.deepcopy( master )
                    scaler = torch.amp.GradScaler()            
                    optimizer = Adafactor(
                        model.parameters(),
                        lr = config.learning_rate,
                        relative_step = False
                    )
                    model.to(config.device)
                    model.train()
                    model.gradient_checkpointing_enable()
                    torch.cuda.empty_cache()
                
                # Update the master    
                current_master_meta = latest_master_meta
                update_block = subtensor.block

            # Iterate over the number of pages to train per epoch.
            for step in range(config.pages_per_epoch):
                # Generate the current training window based on the subtensor block.
                local_pages: List[Tuple[str, int, str]] = SubsetFineWebEdu2Loader.next_pages(
                    offset = subtensor.block * hparams.window_speed + 100,  # Offset into the future to avoid overlap with validators.
                    n_pages = hparams.window_size,
                    seed = my_uid  # Seed with miner's UID for consistency.
                )
    
                # Select a random page from the evaluation window for training.
                local_page = random.choice(local_pages)
    
                # Create the dataset for the selected page.
                dataset = SubsetFineWebEdu2Loader(
                    batch_size = config.actual_batch_size,
                    sequence_length = hparams.sequence_length,
                    pages_info = [ local_page ],
                    tokenizer = hparams.tokenizer
                )
    
                # Calculate the number of gradient accumulation steps to achieve desired batch size.
                accumulation_steps = config.desired_batch_size // config.actual_batch_size
    
                # Zero the gradients of the optimizer.
                optimizer.zero_grad()
                # Training loop over batches in the dataset.
                for idx, batch in enumerate(dataset):
                    # Convert the batch to a PyTorch tensor and move to the device.
                    input_ids = torch.tensor(batch, dtype=torch.long).to(config.device)
                    labels = input_ids.clone()  # Clone input_ids to use as labels.
    
                    # Mask the padding tokens in labels by setting them to -100.
                    # This tells the loss function to ignore these positions.
                    labels = torch.where(labels == hparams.tokenizer.pad_token_id, -100, labels)
    
                    # Forward pass with mixed precision.
                    with torch.amp.autocast( config.device, dtype = torch.bfloat16 ):
                        outputs = model(input_ids=input_ids, labels=labels)
                        loss = outputs.loss  # Get the loss value.
                        loss = loss / accumulation_steps  # Normalize loss for gradient accumulation.
    
                    # Backward pass to compute gradients with scaled loss.
                    scaler.scale(loss).backward()
                    # Perform optimizer step after accumulating gradients.
                    if (idx + 1) % accumulation_steps == 0:
                        # Unscale the gradients and perform optimizer step.
                        scaler.step(optimizer)
                        # Update the scaler for next iteration.
                        scaler.update()
                        # Zero the gradients for the next step.
                        optimizer.zero_grad()
    
                    # Log training progress to console.
                    print(f"Loss: {(loss.item() * accumulation_steps):.4f}")
    
                    # Log metrics to wandb if enabled.
                    if config.use_wandb:
                        wandb.log({
                            "incentive": float(metagraph.I[my_uid]),
                            "loss": loss.item() * accumulation_steps
                        })
                    
                    # TODO: Delete unnecessary tensors to free up GPU memory.
                    del input_ids, labels, outputs
                    torch.cuda.empty_cache()
    
            # After training, remove the previous model from S3 if it exists.
            if len(upload_history) > 10:
                to_delete = upload_history.pop(0)
                CLIENT.delete_object(Bucket=config.bucket, Key=to_delete.filename)
                CLIENT.delete_object(Bucket=config.bucket, Key=to_delete.metadata_filename)
                
            # Compute the delta between the current model and the master.
            delta = copy.deepcopy( model ).to('cpu')
            for (name, delta_param), (_, master_param) in zip(delta.named_parameters(), master.named_parameters()):
                delta_param.data.sub_( master_param.data.to('cpu') )
    
            # Upload the current delta to S3 for evaluation.
            compression = 1 - min( 0.99, (subtensor.block - update_block) / 1000 )
            use_compression = False if compression < 0.05 else True
            if config.use_wandb: wandb.log({'compression': compression} )
            upload_history.append( upload_model(
                key = 'delta',
                wallet = wallet,
                model = delta,
                block = int(time.time()),  # Use current timestamp as block number.
                extras = {},  # Additional metadata can be added here.
                bucket = config.bucket,
                CLIENT = CLIENT,
                use_compression = use_compression,
                compression_percent = compression,
            ))
            del delta
        
        # Handle keyboard interrupts to allow graceful shutdown.
        except (KeyboardInterrupt, SystemExit):
            # Clean up by deleting the model from S3 if it exists.
            print("Training interrupted. Exiting gracefully.")
            break
    
        # Handle any other exceptions, log the error, and continue after a short delay.
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)
            continue

if __name__ == "__main__":
    # Create an argument parser for command-line options.
    parser = argparse.ArgumentParser(description='Miner script')
    
    # Add command-line arguments with default values and help descriptions.
    parser.add_argument('--name', type=str, default=None, help='Optional miner name')
    parser.add_argument('--netuid', type=int, default=212, help='Bittensor network UID.')
    parser.add_argument('--bucket', type=str, default='decis', help='S3 bucket name')
    parser.add_argument('--desired_batch_size', type=int, default=1, help='Desired total batch size for training')
    parser.add_argument('--actual_batch_size', type=int, default=1, help='Actual batch size per step')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer')
    parser.add_argument('--optimizer_beta1', type=float, default=0.9, help='Beta1 for the optimizer')
    parser.add_argument('--optimizer_beta2', type=float, default=0.95, help='Beta2 for the optimizer')
    parser.add_argument('--optimizer_weight_decay', type=float, default=0.1, help='Weight decay for the optimizer')
    parser.add_argument('--pages_per_epoch', type=int, default=1, help='Number of pages to train per epoch')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., cpu or cuda)')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
    
    # Add arguments from Bittensor modules for wallet and subtensor configurations.
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    
    # Parse the arguments to create a configuration object.
    config = bt.config(parser)
    
    # Set the chain endpoint for the subtensor (fixed value).
    config.subtensor.network = 'test'
    config.subtensor.chain_endpoint = 'wss://test.finney.opentensor.ai:443/'
    
    # Call the main function with the parsed configuration.
    main(config)
