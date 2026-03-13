# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    # Separate sequences by class
    seqs_array = np.array(seqs, dtype=object)
    labels_array = np.array(labels)
    
    pos_indices = np.where(labels_array)[0]
    neg_indices = np.where(~labels_array)[0]
    
    pos_seqs = seqs_array[pos_indices]
    neg_seqs = seqs_array[neg_indices]
    
    # Balance by sampling the majority class to match minority class
    n_pos = len(pos_seqs)
    n_neg = len(neg_seqs)
    
    if n_pos < n_neg:
        # Sample negative sequences to match positive
        sampled_neg_indices = np.random.choice(len(neg_seqs), size=n_pos, replace=False)
        sampled_seqs = np.concatenate([pos_seqs, neg_seqs[sampled_neg_indices]])
        sampled_labels = np.concatenate([np.ones(n_pos, dtype=bool), np.zeros(n_pos, dtype=bool)])
    else:
        # Sample positive sequences to match negative
        sampled_pos_indices = np.random.choice(len(pos_seqs), size=n_neg, replace=False)
        sampled_seqs = np.concatenate([pos_seqs[sampled_pos_indices], neg_seqs])
        sampled_labels = np.concatenate([np.ones(n_neg, dtype=bool), np.zeros(n_neg, dtype=bool)])
    
    return list(sampled_seqs), list(sampled_labels)

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    # Define nucleotide to index mapping
    nuc_to_idx = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    
    # Get sequence length and number of sequences
    seq_len = len(seq_arr[0])
    n_seqs = len(seq_arr)
    
    # Initialize encoding array
    encodings = np.zeros((n_seqs, seq_len * 4))
    
    # Encode each sequence
    for i, seq in enumerate(seq_arr):
        for j, nuc in enumerate(seq):
            if nuc in nuc_to_idx:
                idx = nuc_to_idx[nuc]
                encodings[i, j * 4 + idx] = 1
    
    return encodings