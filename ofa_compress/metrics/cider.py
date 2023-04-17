from collections import OrderedDict
from .ciderD import CiderD


def calculate_cider_scores(gen_res, gt_res, CiderD_scorer):
    '''
    gen_res: generated captions, list of str
    gt_idx: list of int, of the same length as gen_res
    gt_res: ground truth captions, list of list of str.
        gen_res[i] corresponds to gt_res[gt_idx[i]]
        Each image can have multiple ground truth captions
    '''
    gen_res_size = len(gen_res)

    res = OrderedDict()
    for i in range(gen_res_size):
        res[i] = [gen_res[i].strip()]

    gts = OrderedDict()
    gt_res_ = [
        [gt_res[i][j].strip() for j in range(len(gt_res[i]))]
        for i in range(len(gt_res))
    ]
    for i in range(gen_res_size):
        gts[i] = gt_res_[i]

    res_ = [{'image_id': i, 'caption': res[i]} for i in range(len(res))]
    _, scores = CiderD_scorer.compute_score(gts, res_)
    return scores
