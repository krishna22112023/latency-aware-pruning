# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Literal, Optional, Union

import peft
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
from peft import LoftQConfig, LoraRuntimeConfig

from peft.utils import PeftType, transpose
from peft import PeftConfig

#import bitsandbytes as bnb
#from bitsandbytes.nn.modules import Int8Params


@dataclass
class LoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.Lora`].

    Args:
        r (`int`): Lora attention dimension
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        merge_weights (`bool`):
            Whether to merge the weights of the Lora layers with the base transformer model in `eval` mode.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        enable_lora ( `List[bool]`): Used with `lora.MergedLinear`.
        lora_bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=None, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=None, metadata={"help": "Lora dropout"})
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the Lora model"}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    enable_lora: Optional[List[bool]] = field(default=None, metadata={"help": "Used with `lora.MergedLinear`."})
    lora_bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    bias: str = field(default="none", metadata={"help": "Bias"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    use_rslora: bool = field(
        default=False,
        metadata={
            "help": (
                "When set to True, uses <a href='https://doi.org/10.48550/arXiv.2312.03732'>Rank-Stabilized LoRA</a>"
                " which sets the adapter scaling factor to `lora_alpha/math.sqrt(r)`, since it"
                " was proven to work better. Otherwise, it will use the original default"
                " value of `lora_alpha/r`."
            )
        },
    )
    init_lora_weights: bool | Literal["gaussian", "olora", "pissa", "pissa_niter_[number of iters]", "loftq"] = field(
        default=True,
        metadata={
            "help": (
                "How to initialize the weights of the LoRA layers. Passing `'True'` (default) results in the default "
                "initialization from the reference implementation from Microsoft. Passing `'gaussian'` results "
                "in Gaussian initialization scaled by the LoRA rank for linear and layers. Setting the initialization "
                "to `'False'` leads to completely random initialization and *is discouraged.*"
                "Passing `'olora'` results in OLoRA initialization."
                "Passing `'pissa'` results in PiSSA initialization."
                "Passing `'pissa_niter_[number of iters]'` initiates Fast-SVD-based PiSSA initialization, "
                "where [number of iters] indicates the number of subspace iterations to perform fsvd, and must be a nonnegative integer."
                "Pass `'loftq'` to use LoftQ initialization"
            ),
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index. "
            "This only works when target_modules is a list of str."
        },
    )
    layer_replication: Optional[list[tuple[int, int]]] = field(
        default=None,
        metadata={
            "help": (
                "This enables using LoRA to effectively expand a transformer model to a larger size by repeating some layers. "
                "The transformation handles models (currently Llama, Bert or Falcon compatible architectures) with "
                "a module list in the model which it modifies to expand the number of modules. "
                "Base weights are shared so the memory usage is close to the original model. The intended use is these base weights "
                "remain fixed during finetuning but each layer has a separate LoRA adapter so the layers can be specialed via "
                "the adapter layers fit during fine tuning."
                "The format is a list of [start, end) pairs which specify the layer ranges to stack. For example:\n"
                "   Original model has 5 layers labelled by their position in the model: `[0, 1, 2, 3, 4]`\n"
                "   layer_replication: `[[0, 4], [2, 5]]`\n"
                "   Final model will have this arrangement of original layers: `[0, 1, 2, 3, 2, 3, 4]`\n"
                "This format is based on what is used for pass-through merges in mergekit. It makes it simple to select sequential "
                "ranges of a model and stack them while reusing layers at either end of each sequence."
            )
        },
    )
    rank_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to ranks which are different from the default rank specified by `r`. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 8`}"
            )
        },
    )
    alpha_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to alphas which are different from the default alpha specified by `lora_alpha`. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 32`}"
            )
        },
    )
    megatron_config: Optional[dict] = field(
        default=None,
        metadata={
            "help": (
                "The TransformerConfig from Megatron. It is used to create LoRA's parallel linear layer."
                "You can get it like this, `core_transformer_config_from_args(get_args())`, "
                "these two functions being from Megatron."
                "You need to specify this parameter when you want to apply LoRA to the ColumnParallelLinear and "
                "RowParallelLinear layers of megatron."
                "It should be noted that we may not be able to use the `save_pretrained` and `from_pretrained` "
                "functions, because TransformerConfig may not necessarily be serialized."
                "But when using megatron, we can use `get_peft_model_state_dict` function and "
                "megatron's framework, they can also save and load models and configurations."
            )
        },
    )
    megatron_core: Optional[str] = field(
        default="megatron.core",
        metadata={
            "help": (
                "The core module from Megatron, it is used to create LoRA's parallel linear layer. "
                "It only needs to be passed in when you need to use your own modified megatron core module. "
                "Otherwise, it will use the default value `megatron.core`. "
            )
        },
    )
    # dict type is used when loading config.json
    loftq_config: Union[LoftQConfig, dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The configuration of LoftQ. If this is passed, then LoftQ will be used to quantize the backbone "
                "weights and initialize Lora layers. Also set `init_lora_weights='loftq'` in this case."
            )
        },
    )
    use_dora: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable <a href='https://arxiv.org/abs/2402.09353'>'Weight-Decomposed Low-Rank Adaptation' (DoRA)</a>. This technique decomposes the updates of the "
                "weights into two parts, magnitude and direction. Direction is handled by normal LoRA, whereas the "
                "magnitude is handled by a separate learnable parameter. This can improve the performance of LoRA, "
                "especially at low ranks. Right now, DoRA only supports linear and Conv2D layers. DoRA introduces a bigger"
                "overhead than pure LoRA, so it is recommended to merge weights for inference."
            )
        },
    )
    runtime_config: LoraRuntimeConfig = field(
        default_factory=LoraRuntimeConfig, metadata={"help": "Runtime configurations"}
    )

    def keys(self):
        """Returns the keys of the config attributes."""
        return self.__dict__.keys()

    def __post_init__(self):
        self.peft_type = PeftType.LORA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        # if target_modules is a regex expression, then layers_to_transform should be None
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        # if target_modules is a regex expression, then layers_pattern should be None
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")

        if self.use_dora and self.megatron_config:
            raise ValueError("DoRA does not support megatron_core, please set `use_dora=False`.")

        # handle init_lora_weights and loftq_config
        if self.init_lora_weights == "loftq":
            import importlib

            if not importlib.util.find_spec("scipy"):
                raise ImportError("The required package 'scipy' is not installed. Please install it to continue.")
            if self.loftq_config is None:
                raise ValueError("`loftq_config` must be specified when `init_lora_weights` is 'loftq'.")

        # Using post training conversion of modified base weights to restore their initial values (PiSSA, OLoRA) cannot
        # be correctly done when using rslora + rank_pattern/alpha_pattern. We can't really know if the user intends
        # this when they'll eventually call save_pretrained (i.e. if they'll pass
        # path_initial_model_for_weight_conversionl). Therefore, we only warn but don't raise an error here.
        if (
            self.use_rslora
            and (self.rank_pattern or self.alpha_pattern)
            and (
                (isinstance(self.init_lora_weights, str) and (self.init_lora_weights.startswith("pissa")))
                or (self.init_lora_weights == "olora")
            )
        ):
            msg = (
                "Using Rank-Stabilized LoRA with rank_pattern/alpha_pattern and post-training conversion of modified "
                "base weights (PiSSA, OLoRA) means that you won't be able to pass "
                "`path_initial_model_for_weight_conversion` to `save_pretrained` to restore the initial values of the "
                "base weights; if you intend to do this, please ensure not to use rslora or rank_pattern/alpha_pattern."
            )
            warnings.warn(msg)

        # convert loftq_config to dict
        if self.loftq_config and not isinstance(self.loftq_config, dict):
            self.loftq_config = vars(self.loftq_config)

        self._custom_modules: Optional[dict[type[nn.Mmodule], type[nn.Module]]] = None

    def to_dict(self):
        """
        Returns the configuration for your adapter model as a dictionary. Removes runtime configurations.
        """
        rv = super().to_dict()
        rv.pop("runtime_config")
        return rv
    
    def __getitem__(self, key):
        return self

class LoraModel(torch.nn.Module):
    """
    Creates Low Rank Adapter (Lora) model from a pretrained transformers model.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM, LoraConfig >>> from peft import LoraModel, LoraConfig >>>
        config = LoraConfig(
            peft_type="LORA", task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, target_modules=["q", "v"],
            lora_dropout=0.01, )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> lora_model = LoraModel(config, model)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    def __init__(self, config, model):
        super().__init__()
        self.peft_config = config
        self.model = model
        self._find_and_replace()
        mark_only_lora_as_trainable(self.model, self.peft_config.lora_bias)
        self.forward = self.model.forward

    def _find_and_replace(self):
        '''loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )'''
        is_target_modules_in_base_model = False
        kwargs = {
            "r": self.peft_config.r,
            "lora_alpha": self.peft_config.lora_alpha,
            "lora_dropout": self.peft_config.lora_dropout,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
            "merge_weights": self.peft_config.merge_weights or self.peft_config.inference_mode,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None
                '''if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                    kwargs.update(
                        {
                            "has_fp16_weights": target.state.has_fp16_weights,
                            "memory_efficient_backward": target.state.memory_efficient_backward,
                            "threshold": target.state.threshold,
                            "index": target.index,
                        }
                    )
                    if self.peft_config.enable_lora is None:
                        new_module = Linear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                    else:
                        kwargs.update({"enable_lora": self.peft_config.enable_lora})
                        new_module = MergedLinear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)'''
                if isinstance(target, peft.tuners.lora.layer.Linear) and self.peft_config.enable_lora is None:
                    new_module = Linear(target.in_features, target.out_features, bias=bias, **kwargs)
                elif self.peft_config.enable_lora is not None:
                    kwargs.update({"enable_lora": self.peft_config.enable_lora})
                    if isinstance(target, Conv1D):
                        in_features, out_features = (
                            target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                        )
                    else:
                        in_features, out_features = target.in_features, target.out_features
                        if kwargs["fan_in_fan_out"]:
                            warnings.warn(
                                "fan_in_fan_out is set to True but the target module is not a Conv1D. "
                                "Setting fan_in_fan_out to False."
                            )
                            kwargs["fan_in_fan_out"] = False
                    new_module = Linear(in_features, out_features, bias=bias, **kwargs)
                else:
                    # Add a fallback or error message here
                    raise ValueError("No valid condition met for creating a new module.")
                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `lora_only` to work
def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class LoraLayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False

class Linear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        self.lora_mask = nn.Parameter(torch.ones(out_features), requires_grad=False)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            self.is_prune = True
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lora_A.train(mode)
        self.lora_B.train(mode)
        if not mode and self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += (
                    transpose(self.lora_B.weight @ self.lora_A.weight, self.fan_in_fan_out) * self.scaling
                )
            self.merged = True
        elif self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= (
                    transpose(self.lora_B.weight @ self.lora_A.weight, self.fan_in_fan_out) * self.scaling
                )
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        self.lora_A.eval()
        self.lora_B.eval()

    def forward(self, x: torch.Tensor):
        if self.disable_adapters:
            if self.r > 0 and self.merged:
                self.weight.data -= (
                    transpose(self.lora_B.weight @ self.lora_A.weight, self.fan_in_fan_out) * self.scaling
                )
                self.merged = False
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            if self.r > 0:
                # Ensure consistent data types
                lora_output = self.lora_B(self.lora_A(self.lora_dropout(x).to(self.lora_A.weight.dtype))) * self.scaling
                result += lora_output.to(result.dtype)
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if hasattr(self, 'lora_mask'):
            result *= self.lora_mask.reshape(1, 1, -1)
        return result


'''class Linear8bitLt(bnb.nn.Linear8bitLt, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features,
        out_features,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        bnb.nn.Linear8bitLt.__init__(
            self,
            in_features,
            out_features,
            bias=kwargs.get("bias", True),
            has_fp16_weights=kwargs.get("has_fp16_weights", True),
            memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
            threshold=kwargs.get("threshold", 0.0),
            index=kwargs.get("index", None),
        )
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=False)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            self.lora_mask = nn.Parameter(torch.ones(out_features), requires_grad=False)
            self.is_prune = True
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor):
        result = super().forward(x)
        # mask = (self.weight.data != 0).to(dtype=self.lora_A.weight.data.dtype)
        if self.disable_adapters:
            return result
        elif self.r > 0:
            if not torch.is_autocast_enabled():
                expected_dtype = result.dtype

                if x.dtype != torch.float32:
                    x = x.float()
                output = self.lora_B(self.lora_A(self.lora_dropout(x))).to(expected_dtype) * self.scaling
                result += output
            else:
                output = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
                result += output
        if hasattr(self, 'lora_mask'):
            result *= self.lora_mask.reshape(1, 1, -1)
        return result'''