import torch.nn.functional as F
import torch.nn as nn
import torch
from typing import List

from textbrewer.compatibility import mask_dtype
import math


def kd_ce_loss_with_mask(logits_S, logits_T, target, padding_idx, temperature=1,constraint_range=None, constraint_masks=None):
    '''
    Calculate the cross entropy between logits_S and logits_T

    :param logits_S: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param logits_T: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param temperature: A float or a tensor of shape (batch_size, length) or (batch_size,)
    '''
    constraint_start = None
    constraint_end = None
    if constraint_range is not None:
        constraint_start, constraint_end = constraint_range.split(',')
        constraint_start = int(constraint_start)
        constraint_end = int(constraint_end)
    logits_T = logits_T[target != padding_idx]
    logits_S = logits_S[target != padding_idx]
    beta_logits_T = logits_T / temperature
    beta_logits_S = logits_S / temperature
    p_T = F.softmax(beta_logits_T, dim=-1)
    # loss = -(p_T * F.log_softmax(beta_logits_S, dim=-1)).sum(dim=-1).mean()
    loss = -(p_T * F.log_softmax(beta_logits_S, dim=-1))
    if constraint_start is not None and constraint_end is not None:
        loss[:,  4:constraint_start] = 0
        loss[:,  constraint_end:] = 0
    if constraint_masks is not None:
        constraint_masks = constraint_masks[target != padding_idx]
        loss.masked_fill_(~constraint_masks, 0)
    ntokens = loss.numel()
    loss = loss.sum()/ntokens
    return loss



def value_relation_loss(kvs_S, kvs_T, mask=None):
    '''
    The value relation loss used in MiniLM (https://arxiv.org/pdf/2002.10957.pdf)
    :param: tuple of torch.tensor kvs_S: tuple of two tensor, each tensor is of the shape (*batch_size*, *num_head*, *length*, *hidden_size*)
    :param: tuple of torch.tensor kvs_T: tuple of two tensor, each tensor is of the shape (*batch_size*, *num_head*, *length*, *hidden_size*)
    NOTE: We use log softmax here to avoid numerical issues caused by explicit log.
    .. math::
        VR = softmax((VV^T)/\sqrt{d_k})
        loss = mean(KL(VR_T || VR_S))
    '''

    # bs * head * seq len * hidden
    _, _, v_S = kvs_S
    _, _, v_T = kvs_T

    d_S = torch.tensor(v_S.shape[-1])
    d_T = torch.tensor(v_T.shape[-1])

    # different head num
    if v_S.shape[1] != v_T.shape[1]:

        def ops(x):
            return torch.flatten(x.permute(0, 2, 1, 3), 2, 3)

        # bs * seq len * hidden dimension
        v_S, v_T = ops(v_S), ops(v_T)

        # bs * seq len * seq len
        VR_S = torch.log_softmax(
            (torch.matmul(v_S, v_S.permute(0, 2, 1))) / torch.sqrt(d_S),
            dim=-1)
        VR_T = torch.log_softmax(
            (torch.matmul(v_T, v_T.permute(0, 2, 1))) / torch.sqrt(d_T),
            dim=-1)
    else:
        # same head num

        # bs * seq len * seq len
        VR_S = torch.log_softmax(
            (torch.matmul(v_S, v_S.permute(0, 1, 3, 2))) / torch.sqrt(d_S),
            dim=-1)
        VR_T = torch.log_softmax(
            (torch.matmul(v_T, v_T.permute(0, 1, 3, 2))) / torch.sqrt(d_T),
            dim=-1)
    loss = F.kl_div(VR_S, VR_T, log_target=True)
    return loss


def key_relation_loss(kvs_S, kvs_T, mask=None):
    '''
    The value relation loss used in MiniLM (https://arxiv.org/pdf/2002.10957.pdf)
    :param: tuple of torch.tensor kvs_S: tuple of two tensor, each tensor is of the shape (*batch_size*, *num_head*, *length*, *hidden_size*)
    :param: tuple of torch.tensor kvs_T: tuple of two tensor, each tensor is of the shape (*batch_size*, *num_head*, *length*, *hidden_size*)
    .. math::
        KR = softmax((VV^T)/\sqrt{d_k})
        loss = mean(KL(KR_T || KR_S))
    '''

    # bs * head * seq len * hidden
    _, k_S, _ = kvs_S
    _, k_T, _ = kvs_T

    d_S = torch.tensor(k_S.shape[-1])
    d_T = torch.tensor(k_T.shape[-1])

    # different head num
    if k_S.shape[1] != k_T.shape[1]:

        def ops(x):
            return torch.flatten(x.permute(0, 2, 1, 3), 2, 3)

        # bs * seq len * hidden dimension
        k_S, k_T = ops(k_S), ops(k_T)

        # bs * seq len * seq len
        KR_S = torch.log_softmax(
            (torch.matmul(k_S, k_S.permute(0, 2, 1))) / torch.sqrt(d_S),
            dim=-1)
        KR_T = torch.log_softmax(
            (torch.matmul(k_T, k_T.permute(0, 2, 1))) / torch.sqrt(d_T),
            dim=-1)
    else:
        # same head num

        # bs * seq len * seq len
        KR_S = torch.log_softmax(
            (torch.matmul(k_S, k_S.permute(0, 1, 3, 2))) / torch.sqrt(d_S),
            dim=-1)
        KR_T = torch.log_softmax(
            (torch.matmul(k_T, k_T.permute(0, 1, 3, 2))) / torch.sqrt(d_T),
            dim=-1)
    loss = F.kl_div(KR_S, KR_T, log_target=True)
    return loss


def query_relation_loss(kvs_S, kvs_T, mask=None):
    '''
    The value relation loss used in MiniLM (https://arxiv.org/pdf/2002.10957.pdf)
    :param: tuple of torch.tensor kvs_S: tuple of two tensor, each tensor is of the shape (*batch_size*, *num_head*, *length*, *hidden_size*)
    :param: tuple of torch.tensor kvs_T: tuple of two tensor, each tensor is of the shape (*batch_size*, *num_head*, *length*, *hidden_size*)
    .. math::
        QR = softmax((VV^T)/\sqrt{d_k})
        loss = mean(KL(QR_T || QR_S))
    '''

    # bs * head * seq len * hidden
    q_S, _, _ = kvs_S
    q_T, _, _ = kvs_T

    d_S = torch.tensor(q_S.shape[-1])
    d_T = torch.tensor(q_T.shape[-1])

    # different head num
    if q_S.shape[1] != q_T.shape[1]:

        def ops(x):
            return torch.flatten(x.permute(0, 2, 1, 3), 2, 3)

        # bs * seq len * hidden dimension
        q_S, q_T = ops(q_S), ops(q_T)

        # bs * seq len * seq len
        QR_S = torch.log_softmax(
            (torch.matmul(q_S, q_S.permute(0, 2, 1))) / torch.sqrt(d_S),
            dim=-1)
        QR_T = torch.log_softmax(
            (torch.matmul(q_T, q_T.permute(0, 2, 1))) / torch.sqrt(d_T),
            dim=-1)
    else:
        # same head num

        # bs * seq len * seq len
        QR_S = torch.log_softmax(
            (torch.matmul(q_S, q_S.permute(0, 1, 3, 2))) / torch.sqrt(d_S),
            dim=-1)
        QR_T = torch.log_softmax(
            (torch.matmul(q_T, q_T.permute(0, 1, 3, 2))) / torch.sqrt(d_T),
            dim=-1)
    loss = F.kl_div(QR_S.log(), QR_T.log(), log_target=True)
    return loss
