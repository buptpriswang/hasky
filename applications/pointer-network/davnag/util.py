#!/usr/bin/env python
# ==============================================================================
#          \file   util.py
#        \author   chenghuige  
#          \date   2017-05-16 15:51:24.025141
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np

def generate_nested_sequence(length, min_seglen=5, max_seglen=10):
    """Generate low-high-low sequence, with indexes of the first/last high/middle elements"""

    # Low (1-5) vs. High (6-10)
    seq_before = [(random.randint(1,5)) for x in xrange(random.randint(min_seglen, max_seglen))]
    seq_during = [(random.randint(6,10)) for x in xrange(random.randint(min_seglen, max_seglen))]
    seq_after = [random.randint(1,5) for x in xrange(random.randint(min_seglen, max_seglen))]
    seq = seq_before + seq_during + seq_after

    # Pad it up to max len with 0's
    seq = seq + ([0] * (length - len(seq)))
    return [seq, len(seq_before), len(seq_before) + len(seq_during)-1]


def create_one_hot(length, index):
    """Returns 1 at the index positions; can be scaled by client"""
    a = np.zeros([length])
    a[index] = 1.0
    return a

def print_pointer(arr, first, second):
    """Pretty print the array, along with pointers to the first/second indices"""
    first_string = " ".join([(" " * (2 - len(str(x))) + str(x)) for x in arr])
    print(first_string)
    second_array = ["  "] * len(arr)
    second_array[first] = "^1"
    second_array[second] = "^2"
    if (first == second):
        second_array[first] = "^B"
    second_string = " " + " ".join([x for x in second_array])
    print(second_string)
  
