from mlora.model.modelargs import MultiLoraBatchData
from mlora.config import LoraConfig
from mlora.profiler.profiler import set_backward_tracepoint, nvtx_range
from mlora.model.LoraFunction import BatchLoraFunction, BatchLoraArgs

import math
import torch
import bitsandbytes

from typing import Dict, Optional, Tuple, List


class Lora(torch.nn.Module):
    def __init__(self, adapter_name: str):
        super().__init__()

        self.adapter_name_: str = adapter_name

        self.lora_a_: torch.Tensor = None  # the shape is r, in_dim
        self.lora_b_: torch.Tensor = None  # the shape is out_dim, r

        self.r_: int = 0
        self.alpha_: int = 0
        self.dropout_: float = 0.0
        self.scaling_: float = 0.0

    def set_parameter(self, r: int, alpha: int, dropout: float):
        self.r_ = r
        self.alpha_ = alpha
        self.dropout_ = dropout
        self.scaling_ = alpha / r


class Linear(torch.nn.Module):
    def __init__(self, weight: torch.nn.Module):
        # the weight just wrapper the module from LlamaForCausalLM
        # the name for debug
        super().__init__()

        if not isinstance(weight, torch.nn.Linear):
            assert isinstance(weight, bitsandbytes.nn.Linear8bitLt) or isinstance(
                weight, bitsandbytes.nn.Linear4bit), f"error type - {type(weight)}."
        else:
            weight.requires_grad_(False)

        self.device_ = weight.weight.device
        self.weight_ = weight
        self.enable_lora_: bool = False
        self.loras_: Dict[str, Lora] = {}

    def init_lora_weight(self,
                         lora_config: LoraConfig,
                         lora_tensor: Tuple[Optional[torch.Tensor],
                                            Optional[torch.Tensor]] = (None, None)):
        # if the lora_tensor is not (None, None), use it to init the lora weight
        assert isinstance(lora_tensor, Tuple)
        assert len(lora_tensor) == 2
        assert ((lora_tensor[0] is None) and (lora_tensor[1] is None)) or (isinstance(
            lora_tensor[0], torch.Tensor) and isinstance(lora_tensor[1], torch.Tensor))

        adapter_name = lora_config.adapter_name_
        r = lora_config.r_
        alpha = lora_config.lora_alpha_
        dropout = lora_config.lora_dropout_

        if adapter_name not in self.loras_:
            self.loras_[adapter_name] = Lora(adapter_name)
        self.loras_[adapter_name].set_parameter(r, alpha, dropout)

        if isinstance(self.weight_, bitsandbytes.nn.Linear4bit):
            out_dim, in_dim = self.weight_.out_features, self.weight_.in_features
        else:
            out_dim, in_dim = self.weight_.weight.shape

        def random_init_lora_a_tensor(lora: Lora):
            lora.__dict__["lora_a_"] = torch.zeros(
                size=(r, in_dim), device=self.device_, requires_grad=True, dtype=torch.float32)
            torch.nn.init.kaiming_normal_(lora.lora_a_, a=math.sqrt(5))

        def zero_init_lora_b_tensor(lora: Lora):
            lora.__dict__["lora_b_"] = torch.zeros(
                size=(out_dim, r), device=self.device_, requires_grad=True, dtype=torch.float32)

        def replace_init_lora_tensor(lora: Lora, lora_a: torch.Tensor, lora_b: torch.Tensor):
            lora.__dict__["lora_a_"] = lora_a.to(device=self.device_).to(
                torch.float32).detach().requires_grad_(True)
            lora.__dict__["lora_b_"] = lora_b.to(device=self.device_).to(
                torch.float32).detach().requires_grad_(True)

        # ensuer it's none, so we can use the __dict__ to init it
        assert self.loras_[adapter_name].lora_a_ is None
        assert self.loras_[adapter_name].lora_b_ is None

        if lora_tensor == (None, None):
            random_init_lora_a_tensor(self.loras_[adapter_name])
            zero_init_lora_b_tensor(self.loras_[adapter_name])
        else:
            replace_init_lora_tensor(self.loras_[adapter_name], *lora_tensor)

        self.enable_lora_ = True

    def get_batch_lora_function_args(self, input_args: MultiLoraBatchData):
        ret_args: List[BatchLoraArgs] = []
        ret_tensors = ()

        for batch_config in input_args.lora_batch_data_config_:
            adapter_name = batch_config.adapter_name_
            start_idx = batch_config.batch_start_idx_
            end_idx = batch_config.batch_end_idx_

            dropout = self.loras_[
                adapter_name].dropout_ if adapter_name in self.loras_ else 0.0
            scaling = self.loras_[
                adapter_name].scaling_ if adapter_name in self.loras_ else 0.0

            lora_a = self.loras_[
                adapter_name].lora_a_ if adapter_name in self.loras_ else None
            lora_b = self.loras_[
                adapter_name].lora_b_ if adapter_name in self.loras_ else None

            ret_args.append(BatchLoraArgs(
                batch_start_index_=start_idx,
                batch_end_index_=end_idx,
                dropout_=dropout,
                scaling_=scaling
            ))

            ret_tensors += (lora_a, lora_b)

        return (ret_args,) + ret_tensors

    def forward(self, data: torch.Tensor, input_args: MultiLoraBatchData) -> torch.Tensor:
        # data shape is: batch_size * max_seq_len * dim
        # result = data @ self.weight_.transpose(0, 1)
        if not self.enable_lora_:
            return self.weight_.forward(data)

        with nvtx_range("f_linear"):
            result = self.weight_.forward(data)
        set_backward_tracepoint(result.grad_fn, "b_linear")

        function_args = self.get_batch_lora_function_args(input_args)
        if len(function_args[0]) == 0:
            return result

        with nvtx_range("f_lora"):
            result = BatchLoraFunction.apply(result, data, *function_args)
        set_backward_tracepoint(result.grad_fn, "b_lora")

        return result
