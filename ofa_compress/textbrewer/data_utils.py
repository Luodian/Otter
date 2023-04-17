import numpy as np
import random

def masking(tokens, p = 0.1, mask='[MASK]'):
    """
    Returns a new list by replacing elements in `tokens` by `mask` with probability `p`.

    Args:
        tokens (list): list of tokens or token ids.
        p (float): probability to mask each element in `tokens`.
    Returns:
        A new list by replacing elements in `tokens` by `mask` with probability `p`.
    """
    outputs = tokens[:]
    for i in range(len(tokens)):
        if np.random.rand() < p:
            outputs[i] = mask 
    return outputs

def deleting(tokens, p = 0.1):
    """
    Returns a new list by deleting elements in `tokens` with probability `p`.

    Args:
        tokens (list): list of tokens or token ids.
        p (float): probability to delete each element in `tokens`.
    Retunrns:
        a new list by deleting elements in :`tokens` with probability `p`.
    """
    choice = np.random.binomial(1,1-p,len(tokens))
    outputs = [tokens[i] for i in range(len(tokens)) if choice[i]==1]
    return outputs


def n_gram_sampling(tokens, 
                    p_ng = [0.2,0.2,0.2,0.2,0.2],
                    l_ng = [1,2,3,4,5]):
    """
    Samples a length `l` from `l_ng` with probability distribution `p_ng`, then returns a random span of length `l` from `tokens`.

    Args:
        tokens (list): list of tokens or token ids.
        p_ng (list): probability distribution of the n-grams, should sum to 1.
        l_ng (list): specify the n-grams.
    Returns:
        a n-gram random span from `tokens`.
    """
    span_length = np.random.choice(l_ng,p= p_ng)
    start_position = max(0,np.random.randint(0,len(tokens)-span_length+1))
    n_gram_span = tokens[start_position:start_position+span_length]
    return n_gram_span


def short_disorder(tokens, p = [0.9,0.1,0,0,0]):  # untouched + four cases abc, bac, cba, cab, bca
    """
    Returns a new list by disordering `tokens` with probability distribution `p` at every possible position. Let `abc` be a 3-gram in `tokens`, 
    there are five ways to disorder, corresponding to five probability values:

        | abc -> abc
        | abc -> bac
        | abc -> cba
        | abc -> cab
        | abc -> bca
    
    Args:
        tokens (list): list of tokens or token ids.
        p (list): probability distribution of 5 disorder types, should sum to 1.
    Returns:
        a new disordered list
    """
    i = 0
    outputs = tokens[:]
    l = len(tokens)
    while i < l-1:
        permutation = np.random.choice([0,1,2,3,4],p=p)
        if permutation!=0 and i==l-2:
            outputs[i], outputs[i+1] = outputs[i+1], outputs[i]
            i += 2
        elif permutation==1:
            outputs[i], outputs[i+1] = outputs[i+1], outputs[i]
            i += 2
        elif permutation==2:
            outputs[i], outputs[i+2] = outputs[i+2], outputs[i]
            i +=3
        elif permutation==3:
            outputs[i],outputs[i+1],outputs[i+2] = outputs[i+2],outputs[i],outputs[i+1]
            i += 3
        elif permutation==4:
            outputs[i],outputs[i+1],outputs[i+2] = outputs[i+1],outputs[i+2],outputs[i]
            i += 3
        else:
            i += 1
    return outputs

def long_disorder(tokens,p = 0.1, length=20):
    """
    Performs a long-range disordering. If ``length>1``, then swaps the two halves of each span of length `length` in `tokens`; 
    if ``length<=1``, treats `length` as the relative length. For example::
    
        >>>long_disorder([0,1,2,3,4,5,6,7,8,9,10], p=1, length=0.4)
        [2, 3, 0, 1, 6, 7, 4, 5, 8, 9]

    Args:
        tokens (list): list of tokens or token ids.
        p (list): probability to swaps the two halves of a spans at possible positions.
        length (int or float): length of the disordered span.
    Returns:
        a new disordered list
    """
    outputs = tokens[:]
    if int(length) <= 1:
        length = len(tokens)*length
    length = (int(length)+1) //2 * 2
    i = 0
    while i<=len(outputs)-length:
        if np.random.rand() < p:
            outputs[i:i+length//2], outputs[i+length//2:i+length] = outputs[i+length//2:i+length], outputs[i:i+length//2]
            i += length
        else:
            i += 1
    return outputs