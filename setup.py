import os, sys
from setuptools import setup, find_packages
from os import path
# from numpy.distutils.core import setup, Extension

# get the dependencies and installs
here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

cmd = 'gfortran ./glmnetforpython/GLMnet.f -fPIC -fdefault-real-8 -shared -o ./glmnetforpython/GLMnet.so'
os.system(cmd)

setup(name='glmnetforpython',
      version = '0.3.2',
      description = 'Python version of glmnet, adapted from Stanford University (scikit-learn style)',
      long_description="T. Moudik's Python version of glmnet (scikit-learn style)",
      url="https://github.com/thierrymoudiki/glmnetforpython",
      author = 'T. Moudiki',
      author_email = 'thierry.moudiki@gmail.com',
      license = 'GPL-2',
      packages=['glmnetforpython'],
      package_data={'glmnetforpython': ['*.so', 'glmnetforpython/*.so']},
      install_requires=all_reqs,
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Operating System :: Unix',
        ],
      keywords='glm glmnet ridge lasso elasticnet',
      )
