import torch

import mlora_op
import unittest
import torch.nn.functional as F


class TestMLoRAOp(unittest.TestCase):
    def test_init_dropout(self):
        inargs = [mlora_op.BatchLoraArgs(0, 2, 8, 0.00, 2.0),
                  mlora_op.BatchLoraArgs(2, 4, -1, 0.00, 2.0),
                  mlora_op.BatchLoraArgs(4, 6, -1, 0.05, 1.0),
                  mlora_op.BatchLoraArgs(6, 8, 8, 0.05, 1.0),
                  mlora_op.BatchLoraArgs(8, 10, 8, 0.05, 2.0)]
        dropout = torch.ones((10, 7, 8), dtype=torch.float,
                             device="cuda:0")

        mlora_op.init_dropout_and_scaling(None, inargs, 7, 8)
        mlora_op.init_dropout_and_scaling(dropout, inargs, 7, 8)


if __name__ == '__main__':
    unittest.main()
