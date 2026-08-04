"""自定义 CUDA 快路径的延迟编译加载器。"""

from functools import lru_cache
import os
from pathlib import Path
import warnings


MAX_CUDA_FRUITS = 256


@lru_cache(maxsize=1)
def load_cuda_extension():
    """编译并缓存整步物理 CUDA Kernel。"""

    if os.name == 'nt':
        # Codex/普通 PowerShell 不一定继承 VS Developer Prompt 环境。
        # 在进程内注入 MSVC include/lib/path，使延迟编译不依赖启动方式。
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            from setuptools._distutils._msvccompiler import _get_vc_env
            vc_env = _get_vc_env('x64')
        for key, value in vc_env.items():
            os.environ[key.upper()] = value
        vc_bin = str(
            Path(vc_env['vctoolsinstalldir']) / 'bin' / 'Hostx64' / 'x64'
        )
        os.environ['PATH'] = vc_bin + os.pathsep + os.environ.get('PATH', '')
        os.environ['DISTUTILS_USE_SDK'] = '1'
        # 避免 torch 解析本地化 cl.exe 版本输出时受 OEM 代码页影响。
        os.environ['VSLANG'] = '1033'

    # 直接调用 Conda 环境的 python.exe 时 Scripts 目录不一定在 PATH。
    # torch 会通过 ``ninja --version`` 检查可用性，因此显式加入其路径。
    import ninja
    ninja_dir = str(Path(ninja.BIN_DIR))
    path_entries = os.environ.get('PATH', '').split(os.pathsep)
    if ninja_dir not in path_entries:
        os.environ['PATH'] = ninja_dir + os.pathsep + os.environ.get('PATH', '')

    import torch.utils.cpp_extension as cpp_extension
    # 本机 cl.exe 输出 UTF-8，而 torch 在 Windows 上默认按 OEM 代码页解码。
    cpp_extension.SUBPROCESS_DECODE_ARGS = ('utf-8', 'ignore')
    load = cpp_extension.load

    package_dir = Path(__file__).resolve().parent
    source_dir = package_dir / 'cuda'
    project_root = package_dir.parents[2]
    build_dir = project_root / '.torch_extensions' / 'daxigua_vector_cuda'
    build_dir.mkdir(parents=True, exist_ok=True)

    verbose = os.environ.get('DAXIGUA_CUDA_BUILD_VERBOSE') == '1'
    return load(
        name='daxigua_vector_cuda',
        sources=[
            str(source_dir / 'vector_step.cpp'),
            str(source_dir / 'vector_step_kernel.cu'),
        ],
        extra_cflags=(
            ['/O2', '/Zc:preprocessor'] if os.name == 'nt' else ['-O3']
        ),
        extra_cuda_cflags=(
            ['-O3', '-lineinfo', '-Xcompiler=/Zc:preprocessor']
            if os.name == 'nt'
            else ['-O3', '-lineinfo']
        ),
        build_directory=str(build_dir),
        with_cuda=True,
        verbose=verbose,
    )
