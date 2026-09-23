import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils import cpp_extension

ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "csrc"
PACKAGE_KERNEL_DIR = ROOT / "torch_ggml_ops" / "kernels" / "gfx1151"
# ROCm builds use the exact-key GGTensile bundle; CUDA builds compile the tiled
# BF16 tensor-core kernels in csrc/cuda/ directly into the extension.
IS_ROCM = torch.version.hip is not None
SOURCES = (
    [
        "csrc/mmq_hip.cu",
        "csrc/mmq_bundle.cpp",
        "csrc/mmq_bundle_loader.cpp",
    ]
    if IS_ROCM
    else ["csrc/mmq_cuda.cu"]
)
HEADER_DEPENDENCIES = [
    path.relative_to(ROOT).as_posix()
    for pattern in ("*.h", "*.cuh")
    for path in sorted(CSRC.rglob(pattern))
    if not path.name.endswith("_hip.cuh")
]
CUDAExtension = cpp_extension.CUDAExtension


def _enable_ccache() -> None:
    ccache = shutil.which("ccache")
    hipcc = shutil.which("hipcc")
    if ccache is None or hipcc is None:
        return

    os.environ.setdefault(
        "PYTORCH_NVCC",
        f"{shlex.quote(ccache)} {shlex.quote(hipcc)}",
    )
    if "CXX" not in os.environ:
        for candidate in (Path("/usr/lib/ccache/c++"), Path("/usr/lib64/ccache/c++")):
            if candidate.is_file():
                os.environ["CXX"] = str(candidate)
                break


_enable_ccache()


class BuildExtension(cpp_extension.BuildExtension):
    def run(self) -> None:
        if not IS_ROCM:
            super().run()
            return
        subprocess.run(
            [sys.executable, "tools/mmq_deployment_bundle.py"],
            cwd=ROOT,
            check=True,
        )
        super().run()
        built_kernel_dir = (
            Path(self.build_lib) / "torch_ggml_ops" / "kernels" / "gfx1151"
        )
        built_kernel_dir.mkdir(parents=True, exist_ok=True)
        package_files = list(PACKAGE_KERNEL_DIR.glob("*.hsaco"))
        expected = {path.name for path in package_files}
        for artifact in built_kernel_dir.iterdir():
            if artifact.is_file() and artifact.name not in expected:
                artifact.unlink()
        for artifact in package_files:
            shutil.copy2(artifact, built_kernel_dir / artifact.name)
        shutil.rmtree(built_kernel_dir / "hip_controls", ignore_errors=True)


stable_defines = [
    "-DTORCH_TARGET_VERSION=0x020A000000000000",
    "-DTORCH_STABLE_ONLY",
    # The stable C shim declares the CUDA stream accessor only under USE_CUDA.
    *([] if IS_ROCM else ["-DUSE_CUDA"]),
]

setup(
    packages=find_packages(exclude=("tests", "tests.*")),
    ext_modules=[
        CUDAExtension(
            name="torch_ggml_ops._C",
            sources=SOURCES,
            include_dirs=[str(CSRC)],
            depends=HEADER_DEPENDENCIES,
            extra_compile_args={
                "cxx": ["-O3", *stable_defines],
                "nvcc": ["-O3", *stable_defines],
            },
            extra_link_args=["-ldl"],
            py_limited_api=True,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    package_data={"torch_ggml_ops": ["kernels/gfx1151/*.hsaco"]},
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
