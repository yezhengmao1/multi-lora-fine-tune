from mlora.model.LoraFunction import BatchLoraArgs, BatchLoraFunction

import torch
import unittest
import torch.nn.functional as F


class TestLoraFunction(unittest.TestCase):
    def set_test_tensor(self):
        torch.manual_seed(42)
        self.weight = torch.randn(
            128, 128, dtype=torch.float, requires_grad=False, device="cuda:0")

        self.lora_a_0 = torch.randn(
            8, 128, dtype=torch.float, requires_grad=True, device="cuda:0")
        self.lora_b_0 = torch.randn(
            128, 8, dtype=torch.float, requires_grad=True, device="cuda:0")

        self.lora_a_1 = torch.randn(
            8, 128, dtype=torch.float, requires_grad=True, device="cuda:0")
        self.lora_b_1 = torch.randn(
            128, 8, dtype=torch.float, requires_grad=True, device="cuda:0")

        self.input = torch.randn(
            6, 10, 128, dtype=torch.float, requires_grad=True, device="cuda:0")

    def lora_pytorch(self):
        self.set_test_tensor()
        linear_result = self.input @ self.weight

        input_0 = self.input[0: 2]
        input_0 = F.dropout(input_0, 0.00)
        input_0 = input_0 @ self.lora_a_0.t() @ self.lora_b_0.t()
        input_0 = input_0 * 2

        input_1 = self.input[2: 4]
        input_1 = F.dropout(input_1, 0.00005)
        input_1 = input_1 @ self.lora_a_1.t() @ self.lora_b_1.t()
        input_1 = input_1 * 1

        batch_output = torch.zeros_like(self.input)
        batch_output[0: 2] = input_0
        batch_output[2: 4] = input_1

        linear_result = linear_result + batch_output

        loss = linear_result.sum()
        self.py_loss = loss.item()
        loss.backward()
        self.py_grad_a_0 = self.lora_a_0.grad
        self.py_grad_a_1 = self.lora_a_1.grad
        self.py_grad_b_0 = self.lora_b_0.grad
        self.py_grad_b_1 = self.lora_b_1.grad
        self.py_grad_input = self.input.grad

    def lora_mlora(self):
        self.set_test_tensor()
        inargs = [BatchLoraArgs(0, 2, 0.00, 2),
                  BatchLoraArgs(2, 4, 0.00005, 1),
                  BatchLoraArgs(4, 6, 0.00, 0)]
        linear_result = self.input @ self.weight

        result = BatchLoraFunction.apply(
            linear_result, self.input, inargs, self.lora_a_0, self.lora_b_0, self.lora_a_1, self.lora_b_1, None, None)

        loss = result.sum()
        self.mlora_loss = loss.item()
        loss.backward()
        self.mlora_grad_a_0 = self.lora_a_0.grad
        self.mlora_grad_a_1 = self.lora_a_1.grad
        self.mlora_grad_b_0 = self.lora_b_0.grad
        self.mlora_grad_b_1 = self.lora_b_1.grad
        self.mlora_grad_input = self.input.grad

    def test_lora(self):
        self.lora_pytorch()
        self.lora_mlora()

        assert abs(self.mlora_loss - self.py_loss) < 1e-2

        assert torch.allclose(self.py_grad_input, self.mlora_grad_input, 1e-4)
        assert torch.allclose(self.py_grad_a_0, self.mlora_grad_a_0, 1e-4)
        assert torch.allclose(self.py_grad_a_1, self.mlora_grad_a_1, 1e-4)
        assert torch.allclose(self.py_grad_b_0, self.mlora_grad_b_0, 1e-4)
        assert torch.allclose(self.py_grad_b_1, self.mlora_grad_b_1, 1e-4)


if __name__ == '__main__':
    unittest.main()
