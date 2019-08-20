"""
@author:    Patrik Purgai
@copyright: Copyright 2019, dialogue-generation
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.07.03.
"""

# pylint: disable=no-name-in-module
# pylint: disable=import-error

from distutils.core import setup
from Cython.Build import cythonize

import numpy
import os

project_path = os.path.abspath(os.path.dirname(__file__))
module_path = os.path.join(project_path, 'src', 'collate.pyx')

setup(
    ext_modules=cythonize(module_path, annotate=True, language='c++'),
    include_dirs=[numpy.get_include()]
)
