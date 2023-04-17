import torch
from typing import List
    
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def initializer_builder(std):
    _std = std
    def init_weights(module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=_std)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    return init_weights


class LayerNode:
    def __init__(self,name,parent=None,value=None,fullname=None):
        self.name = name
        self.fullname = fullname
        self.value = None
        self.children_name = {}
        self.parent = parent
    def __contains__(self, key):
        return key in self.children_name
    def __getitem__(self,key):
        return self.children_name[key]
    def __setitem__(self,key,value):
        self.children_name[key]=value
    def update(self,value):
        if self.parent:
            if self.parent.value is None:
                self.parent.value = value
            else:
                if isinstance(value,(tuple,list)):
                    old_value = self.parent.value
                    new_value = [old_value[i]+value[i] for i in range(len(value))]
                    self.parent.value = new_value
                else:
                    self.parent.value += value

            self.parent.update(value)
    
    def format(self, level=0, total=None ,indent='--',max_level=None,max_length=None):
        string =''
        if total is None:
            total = self.value[0]
        if level ==0:
            max_length = self._max_name_length(indent,'  ',max_level=max_level) + 1
            string += '\n'
            string +=f"{'LAYER NAME':<{max_length}}\t{'#PARAMS':>15}\t{'RATIO':>10}\t{'MEM(MB)':>8}\n"

        if max_level is not None and level==max_level:
            string += f"{indent+self.name+':':<{max_length}}\t{self.value[0]:15,d}\t{self.value[0]/total:>10.2%}\t{self.value[1]:>8.2f}\n"
        else:
            if len(self.children_name)==1:
                string += f"{indent+self.name:{max_length}}\n"
            else:
                string += f"{indent+self.name+':':<{max_length}}\t{self.value[0]:15,d}\t{self.value[0]/total:>10.2%}\t{self.value[1]:>8.2f}\n"
            for child_name, child in self.children_name.items():
                string += child.format(level+1, total, 
                                       indent='  '+indent, max_level=max_level,max_length=max_length) 
        return string

    def _max_name_length(self,indent1='--', indent2='  ',level=0,max_level=None):
        length = len(self.name) + len(indent1) + level *len(indent2)
        if max_level is not None and level >= max_level:
            child_lengths = []
        else:
            child_lengths = [child._max_name_length(indent1,indent2,level=level+1,max_level=max_level) 
                            for child in self.children_name.values()]
        max_length = max(child_lengths+[length])
        return max_length


def display_parameters(model,max_level=None):
    """
    Display the numbers and memory usage of module parameters.

    Args:
        model (torch.nn.Module or dict): the model to be inspected.
        max_level (int or None): The max level to display. If ``max_level==None``, show all the levels.
    Returns:
        A formatted string and a :class:`~textbrewer.utils.LayerNode` object representing the model.
    """
    if isinstance(model,torch.nn.Module):
        state_dict = model.state_dict()
    elif isinstance(model,dict):
        state_dict = model
    else:
        raise TypeError("model should be either torch.nn.Module or a dict")
    hash_set = set()
    model_node = LayerNode('model',fullname='model')
    current = model_node
    for key,value  in state_dict.items():
        names = key.split('.')
        for i,name in enumerate(names):
            if name not in current:
                current[name] = LayerNode(name,parent=current,fullname='.'.join(names[:i+1]))
            current = current[name]
        
        if (value.data_ptr()) in hash_set:
            current.value = [0,0]
            current.name += "(shared)"
            current.fullname += "(shared)"
            current.update(current.value)
        else:
            hash_set.add(value.data_ptr())
            current.value = [value.numel(),value.numel() * value.element_size() / 1024/1024]
            current.update(current.value)
            
        current = model_node

    result = model_node.format(max_level=max_level)
    #print (result)
    return result, model_node

