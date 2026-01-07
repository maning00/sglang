import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


def get_platform_config():
    ROCM_HOME = os.environ.get("ROCM_HOME", "/opt/rocm")
    CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")

    rocm_available = os.path.exists(os.path.join(ROCM_HOME, "include", "hip"))
    cuda_available = os.path.exists(os.path.join(CUDA_HOME, "include", "cuda.h"))

    force_backend = os.environ.get("VTENSOR_BACKEND", "auto").lower()

    if force_backend == "hip" or (force_backend == "auto" and rocm_available):
        print(f"[vTensor] Building with HIP/ROCm backend (ROCM_HOME={ROCM_HOME})")
        include_dirs = [
            os.path.join(ROCM_HOME, "include"),
            os.path.join(ROCM_HOME, "include", "hip"),
        ]
        library_dirs = [
            os.path.join(ROCM_HOME, "lib"),
            os.path.join(ROCM_HOME, "lib64"),
        ]
        libraries = ["amdhip64"]
        extra_compile_args = ["-DVTENSOR_USE_HIP=1", "-D__HIP_PLATFORM_AMD__"]

    elif force_backend == "cuda" or (force_backend == "auto" and cuda_available):
        print(f"[vTensor] Building with CUDA backend (CUDA_HOME={CUDA_HOME})")
        include_dirs = [os.path.join(CUDA_HOME, "include")]
        library_dirs = [
            os.path.join(CUDA_HOME, "lib64"),
            os.path.join(CUDA_HOME, "lib64", "stubs"),
        ]
        libraries = ["cuda", "cudart"]
        extra_compile_args = ["-DVTENSOR_USE_HIP=0"]

    else:
        raise RuntimeError(
            "Neither CUDA nor ROCm found. Please set CUDA_HOME or ROCM_HOME environment variable.\n"
            f"  CUDA_HOME={CUDA_HOME} (exists: {cuda_available})\n"
            f"  ROCM_HOME={ROCM_HOME} (exists: {rocm_available})"
        )

    return include_dirs, library_dirs, libraries, extra_compile_args


include_dirs, library_dirs, libraries, extra_compile_args = get_platform_config()

setup(
    name="vTensor",
    version="0.2",
    packages=find_packages(),
    install_requires=[
        "torch >= 2.0",
    ],
    ext_modules=[
        CppExtension(
            name="vTensor",
            sources=["vtensor.cpp"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    author="antgroup",
    author_email="@antgroup.com",
    description="VMM-based Tensor library for FlowMLA",
)
