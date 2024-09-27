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
import time
import typing
import threading
from threading import Lock
import requests
import numpy as np
import aiohttp
import asyncio
import pickle
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer


class SubsetLoader(IterableDataset):
    """
    Base class for data-specific subset loader classes.

    # TODO: Make this class abstract
    """

    def __init__(
        self,
        batch_size=None,
        sequence_length=None,
        num_pages=None,
        tokenizer: AutoTokenizer = None,
        pack_samples: bool = False,
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_pages = num_pages
        self.tokenizer = tokenizer
        self.pack_samples = pack_samples

        self.num_rows_per_page = 100

        # Buffer to hold pages loaded from the api
        self.buffer = []

        # Buffer to hold pages already loaded into a batch
        self.used_buffer = []

        # Buffer to hold padded pages
        self.padded_buffer = []

    def fetch_data_for_pages(self, pages):
        """
        Set the pages to be used to fill the buffer. Then fetch the page data
        to the buffer.
        """

        self.pages = pages

        # Empty the buffer if it is not.
        self.buffer = []

        for page in self.pages:
            self._fetch_data_for_page(page)

    def _get_pad_size(self, input_ids):
        """
        Get the number of tokens to be padded to the sample to match
        the max allowed sequence length.
        If sample packing is activated, then return 1
        """

        if self.pack_samples:
            return 1

        sample_size = len(input_ids)

        remainder = sample_size % self.sequence_length
        pad_size = self.sequence_length - remainder

        # Apply modulo again to guarantee a pad size of 0 if remainder is 0
        pad_size = pad_size % self.sequence_length

        return pad_size

    def _refill_padded_buffer(self):
        """
        This methods pulls one page from `self.buffer`, pads it and pushs
        it to the `self.padded_buffer`.
        """

        while self.buffer and len(self.padded_buffer) < self.sequence_length:

            input_ids = []

            # search for EOS token index and cut the buffer at it.
            EOS_index = self.buffer.index(self.tokenizer.eos_token_id)
            input_ids = self.buffer[: EOS_index + 1]
            self.buffer = self.buffer[EOS_index + 1 :]

            self.used_buffer += input_ids

            # Add to padded buffer without the EOS token.
            self.padded_buffer += input_ids[:-1]

            # Pad
            self.padded_buffer += [self.tokenizer.eos_token_id] * self._get_pad_size(
                input_ids=input_ids[:-1]
            )

    def __iter__(self):
        self.buffer = self.used_buffer + self.buffer
        self.padded_buffer = []

        # Pad and prepare one page for batching
        self._refill_padded_buffer()

        return self

    def __next__(self):
        batch = []

        while len(self.padded_buffer) >= self.sequence_length:
            batch.append(self.padded_buffer[: self.sequence_length])
            self.padded_buffer = self.padded_buffer[self.sequence_length :]
            self._refill_padded_buffer()

            if len(batch) == self.batch_size:
                return np.stack(batch)

        raise StopIteration


# class SubsetFineWebEdu2Loader(SubsetLoader):

#     name: str = "HuggingFaceFW/fineweb-edu-score-2"
#     rows_base_url: str = "https://datasets-server.huggingface.co/rows"
#     size_base_url: str = "https://datasets-server.huggingface.co/size"

#     retry_limit: int = 10  # Number of retries
#     retry_delay: int = 5  # Seconds to wait between retries
#     num_rows_per_page: int = 100


#     @staticmethod
#     def next_pages( offset: int, n_pages: int, seed: str, num_rows_per_page:int = 100 ):
#         configs_data = SubsetFineWebEdu2Loader.fetch_dataset_configs()
#         rng = np.random.default_rng(hash(seed) & 0xffffffff)  # Create a generator with a seed
#         rng.bit_generator.advance( offset )  # Efficiently skip ahead `n` steps
#         result = []
#         for _ in range( n_pages ):
#             config = rng.choice( list(configs_data.keys() ))
#             choice = rng.integers(0, configs_data[config]['num_rows'] - 1 - num_rows_per_page)
#             result.append( (str(config), int(choice), configs_data[config]['split'] ) )
#         return result

#     def __init__(
#             self,
#             batch_size=None,
#             sequence_length=None,
#             num_pages = None,
#             pages_info = None,
#             tokenizer: AutoTokenizer=None,
#             pack_samples: bool=False,
#     ):
#         super().__init__(batch_size,
#                          sequence_length,
#                          num_pages,
#                          tokenizer,
#                          pack_samples)

#         # Get the dataset configs and their row sizes
#         self.configs_data = SubsetFineWebEdu2Loader.fetch_dataset_configs()

#         if pages_info != None:
#             self._fetch( pages_info )

#         elif self.num_pages:
#             self._fetch_data_to_buffer(self.num_pages)

#     def _fetch( self, page_info: typing.Tuple[ str, int, str ] ):

#         self.pages = page_info
#         attempts = 0
#         num_pages = len(self.pages)
#         for (config_name, page, split) in self.pages:
#             # Create the request parameters
#             params = dict(dataset=self.name,
#                         config=config_name,
#                         split=split,
#                         offset=page,
#                         limit=self.num_rows_per_page
#             )
#             try:
#                 response = requests.get(self.rows_base_url, params=params)
#                 response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
#                 for row in response.json()["rows"]:
#                     content = row["row"]["text"]

#                     # get the tokenized and encoded sample
#                     input_ids = self.tokenizer(content, truncation=True)["input_ids"]
#                     self.buffer += input_ids
#                     self.buffer += [self.tokenizer.eos_token_id]

#                 response.close()

#             except requests.exceptions.RequestException as e:

#                 response.close()
#                 attempts += 1
#                 if attempts < num_pages * self.retry_limit:
#                     pass

#                 else:
#                     raise


#     def _fetch_data_to_buffer(self, num_pages):
#         """
#         Randomly sample pages and add their data to the buffer.
#         If a page is inaccessible, another one is sampled.
#         this method sets the `pages` property
#         """

#         self.pages = []
#         attempts = 0

#         while len(self.pages) < num_pages:

#             # randomly sample one page
#             config_name, page, split = self.get_random_pages(num_pages = 1)[0]

#             # Create the request parameters
#             params = dict(dataset=self.name,
#                           config=config_name,
#                           split=split,
#                           offset=page,
#                           limit=self.num_rows_per_page
#             )

#             try:
#                 response = requests.get(self.rows_base_url, params=params)

#                 response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code

#                 # Add the page since the request was successful
#                 self.pages.append((config_name, page, split))

#                 for row in response.json()["rows"]:
#                     content = row["row"]["text"]

#                     # get the tokenized and encoded sample
#                     input_ids = self.tokenizer(content, truncation=True)["input_ids"]
#                     self.buffer += input_ids
#                     self.buffer += [self.tokenizer.eos_token_id]

#                 response.close()

#             except requests.exceptions.RequestException as e:

#                 response.close()
#                 attempts += 1
#                 if attempts < num_pages * self.retry_limit:
#                     pass

#                 else:
#                     raise

#     def fetch_data_to_rows(self, num_pages):

#         rows = []
#         attempts = 0
#         num_downloaded_pages = 0

#         while num_downloaded_pages < num_pages:

#             # randomly sample one page
#             config_name, page, split = self.get_random_pages(num_pages = 1)[0]

#             # Create the request parameters
#             params = dict(dataset=self.name,
#                           config=config_name,
#                           split=split,
#                           offset=page,
#                           limit=self.num_rows_per_page
#             )

#             try:
#                 response = requests.get(self.rows_base_url, params=params)

#                 response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code

#                 num_downloaded_pages += 1

#                 for row in response.json()["rows"]:
#                     rows.append(row["row"]["text"])

#             except requests.exceptions.RequestException as e:
#                 attempts += 1
#                 if attempts < num_pages * self.retry_limit:
#                     pass

#                 else:
#                     raise


#         return rows

#     def get_random_pages(self, num_pages):
#         """
#         Randomly sample one page.
#         A page is a row number of a given split of a given dataset dump.
#         """
#         pages = []

#         for _ in range(num_pages):

#             # Choose a random config
#             config_name = random.choice(list(self.configs_data.keys()))

#             # Choose a random page (row)
#             page = random.randint(0,
#                                   self.configs_data[config_name]['num_rows'] - 1 - self.num_rows_per_page)

#             split = self.configs_data[config_name]['split']

#             pages.append((config_name, page, split))

#         return pages

#     def get_page_names(self):
#         """
#         This is a utility function that returns the page names that were used.
#         Each page as a single string instead of a tuple
#         """

#         page_names = []

#         if hasattr(self, 'pages'):
#             page_names = [f'{cfg_name}_{num_rows}_{split}' for
#                            cfg_name, num_rows, split in self.pages]

#         return page_names

#     @staticmethod
#     def fetch_dataset_configs() -> typing.Dict[str, typing.Dict]:
#         """
#         Fetch the different dump names, aka configs, aka samples, of the
#         dataset.
#         The returned value is a dictionary with dump names as keys and
#         a dict of the number of rows and the split as values.
#         """
#         # Request parameters
#         params = dict(
#             dataset = SubsetFineWebEdu2Loader.name
#             )

#         attempt = 0
#         while attempt < SubsetFineWebEdu2Loader.retry_limit:
#             try:
#                 response = requests.get(SubsetFineWebEdu2Loader.size_base_url, params=params)
#                 response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code

#                 # Extract the configs dict
#                 configs_dict = response.json()['size']['splits']

#                 # Now create a dict with config names (except 'default') as
#                 # keys, and the number of rows as values
#                 configs_data = {entry['config']: {'num_rows': entry['num_rows'] ,
#                                                   'split': entry['split']}
#                                 for entry in configs_dict
#                                 if entry['config'] != 'default'
#                                 }

#                 return configs_data

#             except requests.exceptions.RequestException as e:
#                 attempt += 1
#                 if attempt < SubsetFineWebEdu2Loader.retry_limit:
#                     time.sleep(SubsetFineWebEdu2Loader.retry_delay)  # Wait before the next retry
#                 else:
#                     raise

#     def _fetch_data_for_page(self, page):

#         retry_limit = 10

#         attempt = 0
#         while attempt < retry_limit:
#             config_name, page, split = page

#             # Create the request parameters
#             params = dict(dataset=self.name,
#                           config=config_name,
#                           split=split,
#                           offset=page,
#                           limit=self.num_rows_per_page
#             )

#             try:

#                 response = requests.get(self.rows_base_url, params=params)

#                 response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code

#                 for row in response.json()["rows"]:
#                     content = row["row"]["text"]
#                     input_ids = self.tokenizer(content, truncation=True)["input_ids"]
#                     self.buffer += input_ids
#                     self.buffer += [self.tokenizer.eos_token_id]

#                 break  # If the request was successful, break out of the retry loop

#             except requests.exceptions.RequestException as e:
#                 attempt += 1
#                 if attempt < self.retry_limit:
#                     time.sleep(self.retry_delay)  # Wait before the next retry
#                 else:
#                     raise


class SubsetFineWebEdu2Loader(SubsetLoader):
    """
    Data loader for the Subset of the FineWebEdu2 dataset with background prefetching and disk caching.

    This class manages data loading, prefetching, and provides batches for training.
    """

    # Class Variables
    name: str = "HuggingFaceFW/fineweb-edu-score-2"
    rows_base_url: str = "https://datasets-server.huggingface.co/rows"
    size_base_url: str = "https://datasets-server.huggingface.co/size"
    retry_limit: int = 10
    retry_delay: int = 1.0  # seconds

    @staticmethod
    def next_pages(offset: int, n_pages: int, seed: str, num_rows_per_page: int = 100):
        configs_data = SubsetFineWebEdu2Loader.fetch_dataset_configs()
        rng = np.random.default_rng(
            hash(seed) & 0xFFFFFFFF
        )  # Create a generator with a seed
        rng.bit_generator.advance(offset)  # Efficiently skip ahead `n` steps
        result = []
        for _ in range(n_pages):
            config = rng.choice(list(configs_data.keys()))
            choice = rng.integers(
                0, configs_data[config]["num_rows"] - 1 - num_rows_per_page
            )
            result.append((str(config), int(choice), configs_data[config]["split"]))
        return result

    def __init__(
        self,
        batch_size: int,
        sequence_length: int,
        tokenizer: AutoTokenizer,
        current_block: int,
        my_uid: int,
        hparams: typing.Any,
        data_dir: str = "./cached_pages",
        prefetch_depth: int = 10,
        pack_samples: bool = False,
    ):
        super().__init__(batch_size, sequence_length, tokenizer, pack_samples)
        # if tokenizer is None:
        #     self.tokenizer = AutoTokenizer.from_pretrained('gpt2')  # Or your specific tokenizer
        # else:
        #     self.tokenizer = tokenizer

        # self.current_block: int = current_block
        self.current_block: int = (
            current_block  # Private variable to store the current block
        )
        self.my_uid: int = my_uid
        self.hparams: typing.Any = hparams
        self.prefetch_depth: int = prefetch_depth
        self.data_dir: str = data_dir
        self.tokenizer = tokenizer
        os.makedirs(self.data_dir, exist_ok=True)

        # Synchronization
        self.lock = Lock()
        self.stop_event = threading.Event()
        self.prefetch_thread = threading.Thread(target=self._prefetch_data, daemon=True)
        self.prefetch_thread.start()

        # Local variables
        self.block_data: typing.Optional[typing.List[np.ndarray]] = None
        self.batch_iter: typing.Optional[typing.Iterator] = None

    @property
    def current_block(self) -> int:
        print(f"Debug: Accessing current_block, value = {self._current_block}")
        return self._current_block

    @current_block.setter
    def current_block(self, value: int):
        print(f"Debug: Setting current_block to {value}")
        self._current_block = int(value)  # Ensure it's an integer

    def _prefetch_data(self):
        """Background thread that prefetches data for future blocks."""
        while not self.stop_event.is_set():
            with self.lock:
                # Determine which blocks need to be prefetched
                blocks_to_prefetch = [
                    block
                    for block in range(
                        self.current_block, self.current_block + self.prefetch_depth
                    )
                    if not self._is_block_cached(block)
                ]

            if blocks_to_prefetch:
                print(f"Prefetching blocks: {blocks_to_prefetch}")
                for block_number in blocks_to_prefetch:
                    if self.stop_event.is_set():
                        break
                    self._fetch_and_cache_block(block_number)

            # Clean up old blocks to maintain cache size
            self._cleanup_cache()

            time.sleep(1)  # Wait before next prefetch cycle

    def _fetch_and_cache_block(self, block_number: int):
        """Fetches data for a block and caches it on disk."""
        pages = self.next_pages(offset=block_number, n_pages=1, seed=self.my_uid)
        try:
            batches = self._fetch_data_for_pages(pages)
            self._save_batches_to_disk(block_number, batches)
            print(f"Block {block_number} prefetched and cached.")
        except Exception as e:
            print(f"Error prefetching block {block_number}: {e}")

    def _fetch_data_for_pages(
        self, pages: typing.List[typing.Tuple[str, int, str]]
    ) -> typing.List[np.ndarray]:
        """Fetches data asynchronously for given pages."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        data = loop.run_until_complete(self._fetch_data_async(pages))
        loop.close()
        return data

    async def _fetch_data_async(
        self, pages: typing.List[typing.Tuple[str, int, str]]
    ) -> typing.List[np.ndarray]:
        """Asynchronously fetches data for pages and processes them."""
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_page(session, page) for page in pages]
            results = await asyncio.gather(*tasks)
        batches = []
        for input_ids in results:
            self.buffer.extend(input_ids)
            self._refill_padded_buffer()
            while len(self.padded_buffer) >= self.sequence_length:
                batch = self.padded_buffer[: self.sequence_length]
                self.padded_buffer = self.padded_buffer[self.sequence_length :]
                batches.append(batch)
        return batches

    async def _fetch_page(
        self, session: aiohttp.ClientSession, page_info: typing.Tuple[str, int, str]
    ) -> typing.List[int]:
        config_name, page, split = page_info
        params = {
            "dataset": self.name,
            "config": config_name,
            "split": split,
            "offset": page,
            "limit": 100,
        }
        attempt = 0
        max_attempts = self.retry_limit
        while attempt < max_attempts:
            try:
                print(f"Attempting to fetch page with params: {params}")
                async with session.get(self.rows_base_url, params=params) as response:
                    print(f"Received response with status: {response.status}")
                    response.raise_for_status()
                    data = await response.json()
                    print(f"Successfully fetched data for page: {page_info}")
                    content = []
                    for row in data["rows"]:
                        text = row["row"]["text"]
                        tokens = self.tokenizer(text, truncation=True)["input_ids"]
                        content.extend(tokens + [self.tokenizer.eos_token_id])
                    return content
            except aiohttp.ClientResponseError as e:
                attempt += 1
                print(
                    f"Attempt {attempt} for page {page_info}: HTTP Error {e.status}: {e.message}"
                )
                if e.status == 404:
                    print(f"Page {page_info} not found. Stopping retries.")
                    break  # Exit the loop if the page does not exist
                await asyncio.sleep(self.retry_delay)
            except Exception as e:
                attempt += 1
                print(f"Attempt {attempt} for page {page_info}: Error: {e}")
                await asyncio.sleep(self.retry_delay)
        raise Exception(
            f"Failed to fetch page {page_info} after {max_attempts} attempts."
        )

    def _refill_padded_buffer(self):
        """Refills the padded buffer by processing tokens in the buffer."""
        while len(self.buffer) >= self.sequence_length:
            input_ids = self.buffer[: self.sequence_length]
            self.buffer = self.buffer[self.sequence_length :]
            pad_size = self._get_pad_size(input_ids)
            input_ids.extend([self.tokenizer.pad_token_id] * pad_size)
            self.padded_buffer.extend(input_ids)

    def _save_batches_to_disk(
        self, block_number: int, batches: typing.List[np.ndarray]
    ) -> None:
        """Saves batches to disk."""
        file_path = os.path.join(self.data_dir, f"block_{block_number}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(batches, f)

    def _load_batches_from_disk(
        self, block_number: int
    ) -> typing.Optional[typing.List[np.ndarray]]:
        """Loads batches from disk."""
        file_path = os.path.join(self.data_dir, f"block_{block_number}.pkl")
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                batches = pickle.load(f)
            return batches
        return None

    def _is_block_cached(self, block_number: int) -> bool:
        """Checks if a block is cached on disk."""
        file_path = os.path.join(self.data_dir, f"block_{block_number}.pkl")
        return os.path.exists(file_path)

    def _cleanup_cache(self):
        """Deletes cached data for blocks that are no longer needed."""
        with self.lock:
            print(f"Debug: self.current_block type = {type(self.current_block)}")
            print(f"Debug: self.current_block value = {self.current_block}")

            # Try to access the underlying value directly
            current_block_value = getattr(self, "_current_block", self.current_block)
            print(f"Debug: current_block_value type = {type(current_block_value)}")
            print(f"Debug: current_block_value = {current_block_value}")

            # Ensure current_block_value is an integer
            try:
                current_block_value = int(current_block_value)
            except ValueError:
                print(
                    f"Error: Unable to convert current_block_value to int. Value: {current_block_value}"
                )
                return  # Exit the method if we can't get a valid integer

            cached_blocks = [
                int(f.split("_")[1].split(".")[0])
                for f in os.listdir(self.data_dir)
                if f.startswith("block_")
            ]
            print(f"Debug: cached_blocks = {cached_blocks}")

            # Use a try-except block to identify which comparison is failing
            blocks_to_delete = []
            for block in cached_blocks:
                try:
                    if block < current_block_value:
                        blocks_to_delete.append(block)
                except TypeError as e:
                    print(f"Debug: Comparison failed for block {block}. Error: {e}")
                    print(
                        f"Debug: block type = {type(block)}, current_block_value type = {type(current_block_value)}"
                    )

            print(f"Debug: blocks_to_delete = {blocks_to_delete}")

            for block in blocks_to_delete:
                file_path = os.path.join(self.data_dir, f"block_{block}.pkl")
                try:
                    os.remove(file_path)
                    print(f"Deleted cached block {block}.")
                except Exception as e:
                    print(f"Error deleting block {block}: {e}")

    def __iter__(self):
        """Returns an iterator over batches for the current block."""
        self.block_data = self._load_batches_from_disk(self.current_block)
        if self.block_data is not None:
            self.batch_iter = iter(self.block_data)
            return self
        else:
            raise StopIteration("Data for current block is not yet available.")

    def __next__(self):
        """Returns the next batch of data."""
        if self.batch_iter is not None:
            try:
                return next(self.batch_iter)
            except StopIteration:
                # End of data for the current block
                self._after_block_processed()
                raise StopIteration
        else:
            raise StopIteration("No batch iterator available.")

    def update_current_block(self, new_block):
        """Updates the current block number and resets iterators."""
        with self.lock:
            print(
                f"Debug: Updating current_block. new_block type: {type(new_block)}, value: {new_block}"
            )
            if isinstance(new_block, property):
                print(
                    "Warning: new_block is a property object. Attempting to get its value."
                )
                try:
                    new_block = new_block.fget(
                        self
                    )  # This tries to get the value of the property
                except:
                    print("Error: Failed to get property value. Using default.")
                    new_block = 0

            new_block = int(new_block)  # Ensure it's an integer
            print(f"Debug: Final new_block value: {new_block}")

            if new_block != self.current_block:
                self.current_block = new_block
                self.block_data = None
                self.batch_iter = None

    def _after_block_processed(self):
        """Handles cleanup after a block has been processed."""
        # Delete the block's data from cache
        file_path = os.path.join(self.data_dir, f"block_{self.current_block}.pkl")
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted cached block {self.current_block} after processing.")
            except Exception as e:
                print(f"Error deleting block {self.current_block}: {e}")

    @staticmethod
    def fetch_dataset_configs() -> typing.Dict[str, typing.Dict]:
        """
        Fetch the different dump names, aka configs, aka samples, of the
        dataset.
        The returned value is a dictionary with dump names as keys and
        a dict of the number of rows and the split as values.
        """
        # Request parameters
        params = dict(dataset=SubsetFineWebEdu2Loader.name)

        attempt = 0
        while attempt < SubsetFineWebEdu2Loader.retry_limit:
            try:
                response = requests.get(
                    SubsetFineWebEdu2Loader.size_base_url, params=params
                )
                response.raise_for_status()  # Raises an HTTPError for bad responses

                # Extract the configs dict
                configs_dict = response.json()["size"]["splits"]

                # Create a dict with config names (except 'default') as keys, and the number of rows and split as values
                configs_data = {
                    entry["config"]: {
                        "num_rows": entry["num_rows"],
                        "split": entry["split"],
                    }
                    for entry in configs_dict
                    if entry["config"] != "default"
                }

                return configs_data

            except requests.exceptions.RequestException as e:
                attempt += 1
                if attempt < SubsetFineWebEdu2Loader.retry_limit:
                    time.sleep(
                        SubsetFineWebEdu2Loader.retry_delay
                    )  # Wait before the next retry
                else:
                    raise

    def stop_prefetching(self):
        """Stops the prefetching thread and cleans up resources."""
        self.stop_event.set()
        if self.prefetch_thread.is_alive():
            self.prefetch_thread.join()

    def __del__(self):
        """Destructor to ensure the prefetching thread is stopped."""
        self.stop_prefetching()
