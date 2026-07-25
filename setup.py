import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils import cpp_extension

ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "csrc"
PACKAGE_KERNEL_DIR = ROOT / "torch_ggml_ops" / "kernels" / "gfx1151"
SOURCES = ["csrc/mmq_hip.cu", "csrc/mmq_bundle.cpp"]
HEADER_DEPENDENCIES = [
    path.relative_to(ROOT).as_posix()
    for pattern in ("*.h", "*.cuh")
    for path in sorted(CSRC.rglob(pattern))
    if not path.name.endswith("_hip.cuh")
]
BUNDLE_BUILD_INPUTS = [
    "tools/build_mmq_bundle.py",
    "tools/mmq_bundle_wrapper_source.py",
]

CUDAExtension = cpp_extension.CUDAExtension


class BuildExtension(cpp_extension.BuildExtension):
    def run(self) -> None:
        subprocess.run(
            [sys.executable, "tools/build_mmq_bundle.py"],
            cwd=ROOT,
            check=True,
        )
        super().run()
        built_kernel_dir = (
            Path(self.build_lib) / "torch_ggml_ops" / "kernels" / "gfx1151"
        )
        built_kernel_dir.mkdir(parents=True, exist_ok=True)
        expected = {path.name for path in PACKAGE_KERNEL_DIR.glob("*.hsaco")}
        for artifact in built_kernel_dir.glob("*.hsaco"):
            if artifact.name not in expected:
                artifact.unlink()
        for artifact in PACKAGE_KERNEL_DIR.glob("*.hsaco"):
            shutil.copy2(artifact, built_kernel_dir / artifact.name)

    def get_source_files(self) -> list[str]:
        # CUDAExtension eagerly rewrites ext.sources to hipify-generated files
        # on ROCm. Source distributions should contain only the canonical input.
        return [*SOURCES, *HEADER_DEPENDENCIES, *BUNDLE_BUILD_INPUTS]


stable_defines = [
    "-DTORCH_TARGET_VERSION=0x020A000000000000",
    "-DTORCH_STABLE_ONLY",
]

setup(
    packages=find_packages(),
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
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
