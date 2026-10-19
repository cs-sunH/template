import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

__version__ = "0.0.1"


def _candidate_cmake_executables():
    env_override = os.environ.get("CMAKE_EXECUTABLE")
    if env_override:
        yield env_override

    seen = set()
    discovered = []

    which_cmake = shutil.which("cmake")
    if which_cmake:
        discovered.append(which_cmake)

    discovered.extend(
        [
            "/usr/bin/cmake",
            "/bin/cmake",
        ]
    )

    for candidate in discovered:
        if not candidate:
            continue
        candidate = os.path.realpath(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        yield candidate


def resolve_cmake_executable():
    for candidate in _candidate_cmake_executables():
        try:
            subprocess.check_output([candidate, "--version"], stderr=subprocess.STDOUT)
            return candidate
        except (OSError, subprocess.CalledProcessError):
            continue
    raise RuntimeError(
        "No usable CMake executable was found. Set CMAKE_EXECUTABLE or ensure a working cmake is on PATH."
    )


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        self.cmake_executable = resolve_cmake_executable()
        print(f"Using CMake executable: {self.cmake_executable}")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep
        build_dir = os.path.join(ext.sourcedir, "build")

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = 'Debug' if debug else 'Release'

        cmake_args = [
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DCMAKE_BUILD_TYPE={cfg}',
            '-DCMAKE_POLICY_VERSION_MINIMUM=3.5',
        ]

        # Prefer vendored FetchContent sources that already exist in the repo so
        # editable installs do not depend on network access during configure.
        local_fetchcontent_sources = {
            "FETCHCONTENT_SOURCE_DIR_SPDLOG": os.path.join(ext.sourcedir, "src", "dram", "ramulator2", "ext", "spdlog"),
            "FETCHCONTENT_SOURCE_DIR_ARGPARSE": os.path.join(ext.sourcedir, "src", "dram", "ramulator2", "ext", "argparse"),
        }
        for cmake_var, source_dir in local_fetchcontent_sources.items():
            if os.path.isdir(source_dir):
                cmake_args += [f"-D{cmake_var}={source_dir}"]

        build_args = ['--config', cfg]
        build_parallel = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL") or str(int(os.cpu_count()*0.8) if int(os.cpu_count()*0.8) > 2 else 2)

        if platform.system() == "Windows":
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--parallel', build_parallel]
        else:
            build_args += ['--parallel', build_parallel]

        os.makedirs(build_dir, exist_ok=True)

        # Run CMake config
        should_config = True
        cmake_cache_file = os.path.join(build_dir, 'CMakeCache.txt')
        
        if os.path.exists(cmake_cache_file):
            should_config = False
            mtime_cache = os.stat(cmake_cache_file).st_mtime
            for root, _, files in os.walk(ext.sourcedir):
                if 'CMakeLists.txt' in files:
                    mtime_list = os.stat(os.path.join(root, 'CMakeLists.txt')).st_mtime
                    if mtime_list > mtime_cache:
                        should_config = True
                        print("Detected change in CMakeLists.txt, re-configuring...")
                        break
            if not should_config:
                cache_text = Path(cmake_cache_file).read_text(errors="ignore")
                python_cache_markers = (
                    f"PYTHON_EXECUTABLE:FILEPATH={sys.executable}",
                    f"PYTHON_EXECUTABLE:PATH={sys.executable}",
                )
                if not any(marker in cache_text for marker in python_cache_markers):
                    should_config = True
                    print("Detected change in PYTHON_EXECUTABLE, re-configuring...")

        if should_config:
             subprocess.check_call([self.cmake_executable, ext.sourcedir] + cmake_args, cwd=build_dir)

        # Reuse the shared simulator/build tree and only build the Python module target
        # (plus its dependencies). This keeps `pip install -e simulator` incremental when
        # a direct CMake build has already populated the shared build directory.
        build_cmd = [self.cmake_executable, '--build', '.', '--target', 'atlasim'] + build_args
        subprocess.check_call(build_cmd, cwd=build_dir)
        
        # Copy shared libraries (e.g. libatlasim-lib.so) to the extension directory
        lib_suffix = ".so"
        if platform.system() == "Windows":
            lib_suffix = ".dll"
        elif platform.system() == "Darwin":
            lib_suffix = ".dylib"

        # Look for built shared libraries and the extension module in the shared build tree.
        copy_targets = ("atlasim-lib", "ramulator")
        # Look for built shared libraries in the build directory
        for root, dirs, files in os.walk(build_dir):
            for file in files:
                is_python_module = file.startswith("atlasim.") and file.endswith(lib_suffix)
                is_runtime_lib = file.endswith(lib_suffix) and any(name in file for name in copy_targets)
                if is_python_module or is_runtime_lib:
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(extdir, file)
                    if not os.path.exists(dst_path) or os.stat(src_path).st_mtime > os.stat(dst_path).st_mtime:
                        print(f"Copying {src_path} to {dst_path}")
                        self.copy_file(src_path, dst_path)


setup(
    name='atlasim',
    version=__version__,
    author='ATLAS simulator Developers',
    description='Python bindings for ATLAS simulator',
    long_description='',
    packages=['atlasim'],
    package_dir={'atlasim': 'atlasim'},
    package_data={'atlasim': ['*.pyi']},
    ext_modules=[CMakeExtension('atlasim.atlasim', sourcedir='.')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    python_requires='>=3.7',
)
