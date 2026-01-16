import os
import sys
from os.path import join

from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import setup
from setuptools.extension import Extension


def _check_gcc_cpp11(cc_name):
    import subprocess

    try:
        cmd = cc_name + " -E -dM -std=c++11 -x c++ /dev/null > /dev/null"
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError:
        return False
    return True


class build_ext_subclass(build_ext):
    def build_extensions(self):
        if hasattr(self.compiler, "compiler") and len(self.compiler.compiler) > 0:
            cc_name = self.compiler.compiler[0]
            stdcpp = "-std=c++11"
            if "gcc" in cc_name and not _check_gcc_cpp11(cc_name):
                stdcpp = "-std=c++0x"
            for e in self.extensions:
                e.extra_compile_args.append(stdcpp)
                e.extra_compile_args.append("-Wno-deprecated-declarations")
                e.extra_compile_args.append("-Wno-unused-local-typedefs")
                e.extra_compile_args.append("-Wno-sign-compare")
                e.extra_compile_args.append("-Wno-unused-but-set-variable")
                e.extra_compile_args.append("-Wno-maybe-uninitialized")

        build_ext.build_extensions(self)


def extra_compile_args():
    if sys.platform.startswith("win"):
        return []
    return [
        "-Wno-comment",
        "-Wno-overloaded-virtual",
        "-Wno-unused-but-set-variable",
        "-Wno-delete-non-virtual-dtor",
        "-Wno-unused-variable",
        "-Wno-maybe-uninitialized",
    ]


def core_extension():
    import numpy as np

    def globr(root, pattern):
        import fnmatch

        matches = []
        for dirpath, _, filenames in os.walk(root):
            for filename in fnmatch.filter(filenames, pattern):
                matches.append(os.path.join(dirpath, filename))

        return matches

    def swig_opts():
        return [
            "-c++",
            "-outdir",
            join("limix_legacy", "deprecated"),
            "-I" + join("src"),
        ]

    def nlopt_files():
        src = open(join("External", "nlopt_src.files")).readlines()
        src = [join("External", "nlopt", s).strip() for s in src]
        hdr = globr(join("External", "nlopt"), "*/*.h")
        return (src, hdr)

    (src, hdr) = nlopt_files()
    src.extend(globr(join("src", "limix_legacy"), "*.cpp"))
    hdr.extend(globr(join("src", "limix_legacy"), "*.h"))

    incl = ["src", "External", join("External", "nlopt")]
    incl = [join(i) for i in incl]
    folder = join("External", "nlopt")
    incl += [join(folder, f) for f in os.listdir(folder)]
    incl = [i for i in incl if os.path.isdir(i)]
    incl.extend([np.get_include()])

    wrap_file = join("src", "interfaces", "python", "limix_wrap.cpp")
    i_file = join("src", "interfaces", "python", "limix_legacy.i")

    if os.path.exists(wrap_file):
        src.append(wrap_file)
    else:
        src.append(i_file)

    depends = src + hdr

    ext = Extension(
        "limix_legacy.deprecated._core",
        src,
        include_dirs=incl,
        extra_compile_args=extra_compile_args(),
        swig_opts=swig_opts(),
        depends=depends,
    )

    return ext


def ensemble_extension():
    import numpy as np

    src = [join("cython", "lmm_forest", "SplittingCore.pyx")]
    incl = [join("External"), np.get_include()]
    depends = src
    ext = Extension(
        "limix_legacy.ensemble.SplittingCore",
        src,
        language="c++",
        include_dirs=incl,
        extra_compile_args=extra_compile_args(),
        depends=depends,
    )

    return cythonize(ext)


# Remove -Wstrict-prototypes from compiler flags (not valid for C++)
# https://stackoverflow.com/a/29634231
import sysconfig

cfg_vars = sysconfig.get_config_vars()
for key, value in list(cfg_vars.items()):
    if isinstance(value, str):
        cfg_vars[key] = value.replace("-Wstrict-prototypes", "")


# Setup call - metadata comes from pyproject.toml
# This only defines the extension modules and custom build command
setup(
    ext_modules=[core_extension()] + ensemble_extension(),
    cmdclass={"build_ext": build_ext_subclass},
)
