import random 
import os 
import numpy as np
import torch
import pandas as pd
import datetime
import pickle

def seed_everything(seed: int):
    """
    Function to set the seed for all random number generators used in the program.

    This function sets the seed for the random number generators in the `random`, `numpy`, and `torch` libraries,
    and also sets the seed for the hash function used by Python's built-in hash function.
    It also sets the `torch` library to use deterministic algorithms for convolution operations,
    and enables the benchmark mode for `torch` to select the fastest algorithms for the hardware configuration.

    Parameters:
    seed (int): The seed for the random number generators.

    Returns:
    None
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



def load_parquet(path: str):
    """
    Function to load a parquet file.

    This function loads a parquet file and returns it as a pandas dataframe.

    Parameters:
    path (str): The path to the parquet file.

    Returns:
    pd.DataFrame: The dataframe containing the parquet file data.
    """
    return pd.read_parquet(path)


def generate_timestamp_filename(extension=".json"):
    """
    Generates a filename with the current timestamp.

    Args:
    - extension (str): File extension to be appended. Default is ".txt".

    Returns:
    - str: Filename based on the current timestamp.
    """
    # Get the current timestamp
    timestamp = datetime.datetime.now()

    # Format the timestamp and add the file extension
    # Example format: '2023-03-15_12-30-45.txt'
    filename = timestamp.strftime("%Y-%m-%d_%H-%M-%S") + extension

    return filename




def get_first_k_words(s, k):
    """
    Returns the first k words from the given string.

    Args:
    - s (str): The string to extract words from.
    - k (int): The number of words to extract.

    Returns:
    - str: A string containing the first k words.
    """
    # Split the string into words
    words = s.split()

    # Take the first k words and join them back into a string
    return ' '.join(words[:k])


def mkdir_p(path, enable_log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    enable_log : bool
        Whether to print result for directory creation
    """
    import errno
    if os.path.exists(path): return
    try:
        os.makedirs(path)
        if enable_log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and enable_log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def init_path(dir_or_file_list):
    if isinstance(dir_or_file_list, list):
        return [_init_path(_) for _ in dir_or_file_list]
    else:  # single file
        return _init_path(dir_or_file_list)


def get_dir_of_file(f_name):
    return os.path.dirname(f_name) + '/'


def _init_path(dir_or_file):
    if dir_or_file.startswith('~'):
        dir_or_file = os.path.expanduser(dir_or_file)
    path = get_dir_of_file(dir_or_file)
    if not os.path.exists(path):
        mkdir_p(path)
    return dir_or_file.replace('//', '/')

def pickle_save(var, f_name):
    init_path(f_name)
    pickle.dump(var, open(f_name, 'wb'))
    print(f'Saved {f_name}')


def pickle_load(f_name):
    return pickle.load(open(f_name, 'rb'))