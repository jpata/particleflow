import torch
import torch.nn as nn
from torch.nn import Sequential as Seq,Linear,ReLU,BatchNorm1d
from torch_scatter import scatter_mean
import numpy as np
import json

class model_io:
    SPECIAL_LAYERS=[
        ".nn2.0",
        # ".conv1.lin_h",
        # ".conv1.lin_p"
    ]

    def __init__(self,device,model,
                model_state_dict,
                activation_dest, dic):

        self.device=device
        self.model=model
        self.model.load_state_dict(model_state_dict)
        self.dest=activation_dest
        self.dic=dic

        # declare variables
        self.L=dict()           # layers
        self.A=activation_dest  # activations
        # self.R=dict()          # relevance scores

        self._rules=dict()     # rules to use for each layer
        self._hook_handles=[]  # collection of all hook handles

        # # extract layers and register hooks
        # self._extract_layers("",model,)

        self.L=dict()
        for name, module in model.named_modules():
            # print(name)
            if name=='conv1' or name=='conv2':
                self.L[name]=module
            else:
                self.L['.'+name]=module

        for key, value in list(self.L.items()):
            if key not in self.dic.keys():
                del self.L[key]

        self.n_layers=len(self.L.keys())

        # register rules for each layer
        self._register_rules()

    """
    rules functions
    """
    def _register_rules(self):
        for layer_name in self.L.keys():
            layer=self.L[layer_name]
            layer_class=layer.__class__.__name__
            if layer_class=="BatchNorm1d":
                rule="z"
            else:
                rule="eps"
            self._rules[layer_name]=rule

    def get_rule(self,index=None,layer_name=None):
        assert (not index is None) or (not layer_name is None), "at least one of (index,name) must be provided"
        if layer_name is None:
            layer_name=self.index2name(index)

        if hasattr(self,"_rules"):
            return self._rules[layer_name]
        else:
            self._register_rules()
            return self._rules[layer_name]

    """
    layer functions
    """

    def _extract_layers(self,name,model):
        l=list(model.named_children())

        if len(l)==0:
            self.L[name]=copy_layer(model)
        else:
            l=list(model.named_children())
            for i in l:
                self._extract_layers(name+"."+i[0],i[1])

    def get_layer(self,index=None,name=None):
        assert (not index is None) or (not name is None), "at least one of (index,name) must be provided"
        if name is None:
            name=self.index2name(index)
        return self.L[name]

    """
    general getters
    """
    def index2name(self,idx:int)->str:
        if not hasattr(self,"_i2n"):
            self._i2n=[]
            for i,n in enumerate(self.A.keys()):
                self._i2n.append(n)
        return self._i2n[idx-2]

    def name2index(self,name:str)->int:
        if not hasattr(self,"_i2n"):
            self._i2n=[]
            for i,n in enumerate(self.A.keys()):
                self._i2n.append(n)
        return self._i2n.index(name)

    """
    reset and setter functions
    """
    def _clear_hooks(self):
        for hook in self._hook_handles:
            hook.remove()

    def reset(self):
        """
        reset the prepared model
        """
        pass
        # self._clear_hooks()
        # self.A=dict()
        # self.R=dict()

    def set_dest(self,activation_dest):
        self.A=activation_dest

def copy_layer(layer):
    """
    create a deep copy of provided layer
    """
    layer_cp=eval("nn."+layer.__repr__())
    layer_cp.load_state_dict(layer.state_dict())

    return layer_cp.to(self.device)

def copy_tensor(tensor,dtype=torch.float32):
    """
    create a deep copy of the provided tensor,
    outputs the copy with specified dtype
    """

    return tensor.clone().detach().requires_grad_(True).to(self.device)
