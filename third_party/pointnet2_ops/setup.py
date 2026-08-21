import glob
import os
import os.path as osp

from setuptools import find_packages, setup
import torch.utils.cpp_extension as cpp_extension


if os.environ.get("CONTACTDIFF_ALLOW_CUDA_TOOLKIT_MISMATCH") == "1":
    # Some managed GPU images expose only a newer system nvcc than the CUDA
    # runtime bundled with PyTorch. PointNet++ contains plain CUDA kernels and
    # no cuDNN/cuBLAS calls, so permit an explicitly requested local build and
    # require a post-build import/forward smoke test before use.
    cpp_extension._check_cuda_version = lambda *_args, **_kwargs: None

BuildExtension = cpp_extension.BuildExtension
CUDAExtension = cpp_extension.CUDAExtension

this_dir = osp.dirname(osp.abspath(__file__))
_ext_src_root = osp.join("pointnet2_ops", "_ext-src")
_ext_sources = glob.glob(osp.join(_ext_src_root, "src", "*.cpp")) + glob.glob(
    osp.join(_ext_src_root, "src", "*.cu")
)

requirements = ["torch>=1.4"]

exec(open(osp.join("pointnet2_ops", "_version.py")).read())

setup(
    name="pointnet2_ops",
    version=__version__,
    author="Erik Wijmans",
    packages=find_packages(),
    install_requires=requirements,
    ext_modules=[
        CUDAExtension(
            name="pointnet2_ops._ext",
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-Xfatbin", "-compress-all"],
            },
            include_dirs=[osp.join(this_dir, _ext_src_root, "include")],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
)
