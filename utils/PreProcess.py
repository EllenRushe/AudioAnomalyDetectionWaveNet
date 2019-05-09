import sys 
import numpy as np
import tensorflow as tf

def linear2mu(x, mu=255):
    """
    Taken from: https://github.com/soroushmehr/sampleRNN_ICLR2017/blob/master/datasets/dataset.py
    @article{mehri2016samplernn, Author = {Soroush Mehri and Kundan Kumar and Ishaan Gulrajani and Rithesh Kumar and Shubham Jain and Jose Sotelo and Aaron Courville and Yoshua Bengio}, Title = {SampleRNN: An Unconditional End-to-End Neural Audio Generation Model}, Year = {2016}, Journal = {arXiv preprint arXiv:1612.07837}, }
    From Joao
    x should be normalized between -1 and 1
    Converts an array according to mu-law and discretizes it
    Note:
        mu2linear(linear2mu(x)) != x
        Because we are compressing to 8 bits here.
        They will sound pretty much the same, though.
    :usage:
        >>> bitrate, samples = scipy.io.wavfile.read('orig.wav')
        >>> norm = __normalize(samples)[None, :]  # It takes 2D as inp
        >>> mu_encoded = linear2mu(2.*norm-1.)  # From [0, 1] to [-1, 1]
        >>> print mu_encoded.min(), mu_encoded.max(), mu_encoded.dtype
        0, 255, dtype('int16')
        >>> mu_decoded = mu2linear(mu_encoded)  # Back to linear
        >>> print mu_decoded.min(), mu_decoded.max(), mu_decoded.dtype
        -1, 0.9574371, dtype('float32')
    """
    x_mu = np.sign(x) * np.log(1 + mu*np.abs(x))/np.log(1 + mu)
    return ((x_mu + 1)/2 * mu).astype('int64')

def mu2linear(x, mu=255):
    # Taken from: https://github.com/soroushmehr/sampleRNN_ICLR2017/blob/master/datasets/dataset.py
    # @article{mehri2016samplernn, Author = {Soroush Mehri and Kundan Kumar and Ishaan Gulrajani and Rithesh Kumar and Shubham Jain and Jose Sotelo and Aaron Courville and Yoshua Bengio}, Title = {SampleRNN: An Unconditional End-to-End Neural Audio Generation Model}, Year = {2016}, Journal = {arXiv preprint arXiv:1612.07837}, }
    """
    From Joao with modifications
    Converts an integer array from mu to linear
    For important notes and usage see: linear2mu
    """
    mu = float(mu)
    x = x.astype('float32')
    y = 2. * (x - (mu+1.)/2.) / (mu+1.)
    return np.sign(y) * (1./mu) * ((1. + mu)**np.abs(y) - 1.)


def shift_offset(x):
  # Zero at the beginning, slice off the last number.
  start_zero= tf.pad(x, [[0,0],[0, 1], [0,0]])
  return tf.manip.roll(start_zero, shift=1, axis=1)[:,:-1,]


# def calc_receptive_field(kernel_size, stack_size, num_stacks):
#     '''
#     Calculate size of input window based on network architecture
#     for a single stack, the receptive field is: 
#     (kernel_size - 1)*(sum(dilations)+1) so for n stacks we add 1 
#     n times. 
#     '''
#     # Total number of layers: stack_size*num_stacks
#     dilations = [2**(i% stack_size) for i in range(stack_size*num_stacks)]
#     receptive_field = (kernel_size -1) * (sum(dilations) + num_stacks) 
#     return receptive_field


# def normalize_audio(samples, bit_rate):
#      return samples/(2**(bit_rate-1))


