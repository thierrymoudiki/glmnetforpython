import os, sys
from setuptools import setup, find_packages
# from numpy.distutils.core import setup, Extension

cmd = 'gfortran ./glmnetforpython/GLMnet.f -fPIC -fdefault-real-8 -shared -o ./glmnetforpython/GLMnet.so'
os.system(cmd)

setup(name='glmnetforpython',
      version = '0.3.0',
      description = 'Python version of glmnet, adapted from Stanford University',
      long_description=open('README.md').read(),
      url="https://github.com/thierrymoudiki/glmnetforpython",
      author = 'T. Moudiki',
      author_email = 'thierry.moudiki@gmail.com',
      license = 'GPL-2',
      packages=['glmnetforpython'],
      install_requires=['joblib>=0.10.3'],
      package_data={'glmnetforpython': ['*.so', 'glmnetforpython/*.so']},
      dependencies = ['numpy', 'pandas', 'scipy', 'scikit-learn'],
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
