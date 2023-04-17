import torch.nn.functional as F
import torch

# x is between 0 and 1
def linear_growth_weight_scheduler(x):
    return x

def linear_decay_weight_scheduler(x):
    return 1-x

def constant_temperature_scheduler(logits_S, logits_T, base_temperature):
    '''
    Remember to detach logits_S 
    '''
    return base_temperature


def flsw_temperature_scheduler_builder(beta,gamma,eps=1e-4, *args):
    '''
    adapted from arXiv:1911.07471
    '''
    def flsw_temperature_scheduler(logits_S, logits_T, base_temperature):
        v = logits_S.detach()
        t = logits_T.detach()
        with torch.no_grad():
            v = v/(torch.norm(v,dim=-1,keepdim=True)+eps)
            t = t/(torch.norm(t,dim=-1,keepdim=True)+eps)
            w = torch.pow((1 - (v*t).sum(dim=-1)),gamma)
            tau = base_temperature + (w.mean()-w)*beta
        return tau
    return flsw_temperature_scheduler


def cwsm_temperature_scheduler_builder(beta,*args):
    '''
    adapted from arXiv:1911.07471
    '''
    def cwsm_temperature_scheduler(logits_S, logits_T, base_temperature):
        v = logits_S.detach()
        with torch.no_grad():
            v = torch.softmax(v,dim=-1)
            v_max = v.max(dim=-1)[0]
            w = 1 / (v_max + 1e-3)
            tau = base_temperature + (w.mean()-w)*beta
        return tau
    return cwsm_temperature_scheduler
