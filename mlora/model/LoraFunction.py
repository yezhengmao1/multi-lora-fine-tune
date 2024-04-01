import torch

import mlora_op

from typing import List
from dataclasses import dataclass


@dataclass
class BatchLoraArgs:
    batch_start_index_: int = -1
    batch_end_index_: int = -1
    dropout_: float = 0.0
    scaling_: float = 0.0


class BatchLoraFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, linear_result: torch.Tensor, batch_data: torch.Tensor,
                batch_args: List[BatchLoraArgs], *args):
        # preprare the data to mlora_op
        loras = [lora for lora in args]

        inargs: List[mlora_op.BatchLoraArgs] = []
        max_rank = 0
        max_batch_size = max(
            [arg.batch_end_index_ - arg.batch_start_index_ for arg in batch_args])

        for arg, lora in zip(batch_args, args[::2]):
            lora_rank = lora.shape[0] if lora is not None else -1
            inargs.append(mlora_op.BatchLoraArgs(
                arg.batch_start_index_,
                arg.batch_end_index_,
                lora_rank,
                arg.dropout_,
                arg.scaling_))
            max_rank = lora_rank if lora_rank > max_rank else max_rank

        _, seq_len, out_dim = linear_result.shape
        in_dim = batch_data.shape[2]

        linear_result = linear_result.to(torch.float)
        tmp_dropout = torch.ones((max_batch_size * seq_len * in_dim),
                                 dtype=torch.float, device=batch_data.device)
        tmp_data = torch.zeros((max_batch_size * seq_len * max_rank),
                               dtype=torch.float, device=batch_data.device)

        assert linear_result.is_contiguous()
        assert batch_data.is_contiguous()
        assert tmp_dropout.is_contiguous()
        assert tmp_data.is_contiguous()

        # batch_data will not be changed, we will alloc cuda memory in mlora_op
        mlora_op.batch_lora_forward(
            tmp_dropout, linear_result, batch_data, tmp_data, inargs, in_dim, out_dim, seq_len, loras)

        ctx.inargs = inargs
        ctx.max_rank = max_rank
        ctx.max_batch_size = max_batch_size
        ctx.save_for_backward(*(batch_data, *args))

        return linear_result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # prealloc the memory for backward
        batch_data = ctx.saved_tensors[0]
        loras = ctx.saved_tensors[1:]
        # convert the tuple to list, the cpp will convert it to vector
        loras = [lora for lora in loras]

        grad_linear_result = grad_output if ctx.needs_input_grad[0] else None
        grad_batch_data = torch.zeros_like(
            batch_data) if ctx.needs_input_grad[1] else None
        grad_batch_args = None
        grad_loras = [torch.zeros_like(
            lora) if lora is not None else None for lora in loras]

        _, seq_len, out_dim = grad_output.shape
        in_dim = batch_data.shape[2]

        tmp_data = torch.zeros(
            (ctx.max_batch_size * seq_len * ctx.max_rank), dtype=torch.float, device=grad_output.device)
        tmp_lora_data = torch.zeros(
            (ctx.max_batch_size * max(in_dim, out_dim) * ctx.max_rank), dtype=torch.float, device=grad_output.device)
        tmp_one_vector = torch.ones(
            (ctx.max_batch_size), dtype=torch.float, device=grad_output.device)

        grad_output = grad_output.contiguous()

        assert grad_output.is_contiguous()
        assert batch_data.is_contiguous()
        assert tmp_data.is_contiguous()

        mlora_op.batch_lora_backward(
            grad_output, batch_data, grad_batch_data, tmp_data, tmp_lora_data, tmp_one_vector, ctx.inargs, in_dim, out_dim, seq_len, loras, grad_loras)

        return grad_linear_result, grad_batch_data, grad_batch_args, *grad_loras
