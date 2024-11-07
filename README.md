# Glmnet for python 

![PyPI](https://img.shields.io/pypi/v/glmnetforpython) [![PyPI - License](https://img.shields.io/pypi/l/glmnetforpython)](https://github.com/thierrymoudiki/glmnetforpython/blob/master/LICENSE) [![Downloads](https://pepy.tech/badge/glmnetforpython)](https://pepy.tech/project/glmnetforpython) 


## Install

Using pip

```bash    
pip install glmnetforpython
```

From GitHub

```bash    
pip install git+github.com/thierrymouidiki/glmnetforpython.git
```    
    
## What?

This is a python version of the popular `glmnet` library (scikit-learn style). Glmnet fits the entire lasso or elastic-net regularization path for `linear` regression, `logistic` and `multinomial` regression models, `poisson` regression and the `cox` model. 

The underlying fortran codes are the same as the `R` version, and uses a cyclical path-wise coordinate descent algorithm as described in the papers linked below. 

Currently, `glmnet` library methods for gaussian, multi-variate gaussian, binomial, multinomial, poisson and cox models are implemented for both normal and sparse matrices.

Additionally, cross-validation is also implemented for gaussian, multivariate gaussian, binomial, multinomial and poisson models. CV for cox models is yet to be implemented. 

CV can be done in both serial and parallel manner. Parallellization is done using `multiprocessing` and `joblib` libraries.

During installation, the fortran code is compiled in the local machine using `gfortran`, and is called by the python code.

## Usage

See [examples](./examples/glmnet.py)

## References:

- Friedman, J., Hastie, T. and Tibshirani, R. (2008) Regularization Paths for Generalized Linear Models via Coordinate Descent, 
http://www.jstatsoft.org/v33/i01/
*Journal of Statistical Software, Vol. 33(1), 1-22 Feb 2010*
    
- Simon, N., Friedman, J., Hastie, T., Tibshirani, R. (2011) Regularization Paths for Cox's Proportional Hazards Model via Coordinate Descent,
http://www.jstatsoft.org/v39/i05/
*Journal of Statistical Software, Vol. 39(5) 1-13*

- Tibshirani, Robert., Bien, J., Friedman, J.,Hastie, T.,Simon, N.,Taylor, J. and Tibshirani, Ryan. (2010) Strong Rules for Discarding Predictors in Lasso-type Problems,
http://www-stat.stanford.edu/~tibs/ftp/strong.pdf
*Stanford Statistics Technical Report*

## License:

This software is released under GNU General Public License v3.0 or later. 
