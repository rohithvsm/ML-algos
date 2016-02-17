"""
This module contains the functions for calculating the information gain of a
dataset as defined by the ID3 (Information Theoretic) heuristic.
"""

import math
import collections
from dtree import get_values

NUMERIC = 'numeric'


def entropy(data, target_attr):
    """Calculates the entropy of the given data set for the target
    attribute."""

    val_freq = collections.defaultdict(int)
    data_entropy = 0.0

    # Calculate the frequency of each of the values in the target attr
    for record in data:
        val_freq[record[target_attr]] += 1.0

    # Calculate the entropy of the data for the target attribute
    for freq in val_freq.values():
        data_entropy += (-freq / len(data)) * math.log(freq / len(data), 2)

    return data_entropy


def gain(data, attr, target_attr, attributes_meta):
    """Calculates the information gain after splitting by attr."""

    val_freq = collections.defaultdict(int)
    subset_entropy = 0.0
    candidate_split = None

    if attributes_meta[attr] == NUMERIC:
        candidate_split = get_values(data, attr, attributes_meta)
        for record in data:
            if record[attr] <= candidate_split:
                val_freq['<=%s' % candidate_split] += 1.0
            else:
                val_freq['>%s' % candidate_split] += 1.0
    else:
        # Calculate the frequency of each of the values in the target attribute
        for record in data:
            val_freq[record[attr]] += 1.0

    # Calculate the sum of the entropy for each subset of records weighted
    # by their probability of occurring in the training set.
    for val in val_freq:
        val_prob = val_freq[val] / sum(val_freq.values())

        if attributes_meta[attr] == NUMERIC:
            if val.startswith('<='):
                data_subset = [record for record in data if record[attr] <=
                               candidate_split]
            else:
                data_subset = [record for record in data if record[attr] >
                               candidate_split]
        else:
            data_subset = [record for record in data if record[attr] == val]

        subset_entropy += val_prob * entropy(data_subset, target_attr)

    # return the information gain
   return entropy(data, target_attr) - subset_entropy
