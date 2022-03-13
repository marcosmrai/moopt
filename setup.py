# -*- coding: utf-8 -*-
"""
@author:  Marcos M. Raimundo
@email:   marcosmrai@gmail.com
@license: BSD 3-clause.
"""
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def read_version():
    # Default value if we cannot find the __version__ field in the init file:
    version = "0.0.1"

    # TODO: Possibly more robust way to find the directory:
    # filename = inspect.getframeinfo(inspect.currentframe()).filename
    # path = os.path.dirname(os.path.abspath(filename))

    init_file = (os.path.dirname(os.path.realpath(__file__)) +
                 "/moo/__init__.py")
    if os.path.exists(init_file):
        with open(init_file, 'r') as f:
            for line in f:
                if line.startswith("__version__"):
                    _, version = line.split("=")
                    version = version.replace('\"', '').replace('\'', '')
                    version = version.strip()
                    break

    return version


params = dict(name="moopt",
              version=read_version(),
              author="See contributors on https://github.com/marcosmrai/moopt",
              author_email="marcosmrai@gmail.com",
              maintainer="Marcos M. Raimundo",
              maintainer_email="marcosmrai@gmail.com",
              description="""MOOPT: A posteriory multi-objetive algorithms
                          in python.""",
              license="BSD 3-clause.",
              keywords="multi-objective optimization, scalarization",
              url="https://github.com/marcosmrai/moopt",
              long_description=read("README.md"),
              package_dir={"": "."},
              packages=["moopt",
                        ],
              # package_data = {"": ["README.md", "LICENSE"],
              #                 "examples": ["*.py"],
              #                 "tests": ["*.py"],
              #                },
              classifiers=["Development Status :: 3 - Alpha",
                           "Intended Audience :: Developers",
                           "Intended Audience :: Science/Research",
                           "License :: OSI Approved :: BSD 3-Clause License",
                           "Topic :: Scientific/Engineering",
                           "Topic :: Machine learning"
                           "Programming Language :: Python :: 3.7",
                           ],
              )

try:
    from setuptools import setup

    params["install_requires"] = ["pulp~=1.6.9",
                                  "numpy>=1.20.2"
                                  "scipy>=0.13.3",
                                  "mip>=1.13.0",
                                  "gurobipy>=9.1.2"
                                  ]
except ImportError:
    from distutils.core import setup

setup(**params)
