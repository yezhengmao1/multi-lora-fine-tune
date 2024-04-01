from setuptools import setup
from torch.utils import cpp_extension

setup(name="mlora_op",
      ext_modules=[cpp_extension.CUDAExtension(
          name="mlora_op",
          sources=["mlora_op.cpp"],
      )],

      cmdclass={"build_ext": cpp_extension.BuildExtension})
