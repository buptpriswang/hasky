from __future__ import absolute_import, division, print_function

import numpy as np


class DataGenerator(object):
  def __init__(self):
    """Construct a DataGenerator."""
    pass
  def next_batch(self, batch_size, N):
    """Return the next `batch_size` examples from this data set."""

    # A sequence of random numbers from [0, 1]
    encoder_inputs = []

    # Sorted sequence that we feed to encoder
    # In inference we feed an unordered sequence again
    decoder_inputs = []

    # Ordered sequence where one hot vector encodes position in the input array
    decoder_targets = []

    for _ in range(N):
      encoder_inputs.append(np.zeros([batch_size, 1]))
    for _ in range(N + 1):
      decoder_inputs.append(np.zeros([batch_size, 1]))
      decoder_targets.append(np.zeros([batch_size, 1]))

    for b in range(batch_size):
      shuffle = np.random.permutation(N)
      sequence = np.sort(np.random.random(N))
      shuffled_sequence = sequence[shuffle]

      for i in range(N):
        encoder_inputs[i][b] = shuffled_sequence[i]
        decoder_inputs[i + 1][b] = sequence[i]
        decoder_targets[shuffle[i]][b] = i + 1

      # Points to the stop symbol
      decoder_targets[N][b] = 0

    return encoder_inputs, decoder_inputs, decoder_targets

#Reader:  [array([[ 0.11703694]]), array([[ 0.07738304]]), array([[ 0.94157312]]), array([[ 0.31190214]]), array([[ 0.62422142]])]
#Decoder:  [array([[ 0.]]), array([[ 0.07738304]]), array([[ 0.11703694]]), array([[ 0.31190214]]), array([[ 0.62422142]]), array([[ 0.94157312]])]
#Writer:  [array([ 2.]), array([ 1.]), array([ 4.]), array([ 5.]), array([ 3.]), array([ 0.])]

if __name__ == "__main__":
  dataset = DataGenerator()
  r, d, w = dataset.next_batch(1, 5)
  print("Reader: ", r)
  print("Decoder: ", d)
  print("Writer: ", w)

