import matplotlib.pyplot as plt
import numpy as np
import warnings

from collections import namedtuple
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from .glmnet import glmnet
from .glmnetPlot import glmnetPlot
from .glmnetPrint import glmnetPrint
from .glmnetCoef import glmnetCoef
from .glmnetPredict import glmnetPredict
from .cvglmnet import cvglmnet
from .cvglmnetCoef import cvglmnetCoef
from .cvglmnetPlot import cvglmnetPlot
from .cvglmnetPredict import cvglmnetPredict


class GLMNet(BaseEstimator, RegressorMixin, ClassifierMixin):
    """
    A sklearn-style wrapper for the glmnet package.

    Parameters
    ----------
    alpha : float, optional
        The alpha parameter in the elastic net penalty. Default is 1.0.
    nlambda : int, optional
        The number of lambda values to use. Default is 100.
    lambdau : float, optional
        The user-defined lambda value. Default is None.
    standardize : bool, optional
        Whether to standardize the predictors. Default is True.
    thresh : float, optional
        The convergence threshold. Default is 1e-07.
    dfmax : float, optional
        The maximum number of degrees of freedom. Default is 1e+10.
    pmax : float, optional
        The maximum number of predictors. Default is 1e+10.
    exclude : array-like, optional
        The indices of predictors to exclude. Default is None.
    penalty_factor : array-like, optional
        The penalty factors for each predictor. Default is None.
    lower_lambdau : float, optional
        The lower bound for the lambda values. Default is None.
    upper_lambdau : float, optional
        The upper bound for the lambda values. Default is None.
    maxit : float, optional
        The maximum number of iterations. Default is 1e+05.
    type_measure : int, optional
        The type of measure to use. Default is 1.
    family : str, optional
        The family of the response variable. Default is 'gaussian'.
    parallel : bool, optional
        Whether to use parallel processing. Default is False.
    ncores : int, optional
        The number of cores to use. Default is -1.
    verbose : bool, optional
        Whether to print messages. Default is False.

    See also

    """

    def __init__(
        self,
        weights=None,
        alpha=1.0,
        nlambda=100,
        lambdau=None,
        standardize=True,
        thresh=1e-07,
        dfmax=1e10,
        pmax=1e10,
        exclude=None,
        penalty_factor=None,
        lower_lambdau=None,
        upper_lambdau=None,
        maxit=1e05,
        type_measure=1,
        family="gaussian",
        parallel=False,
        ncores=-1,
        verbose=False,
    ):
        self.weights = weights
        self.alpha = alpha
        self.nlambda = nlambda
        self.lambdau = lambdau
        self.standardize = standardize
        self.thresh = thresh
        self.dfmax = dfmax
        self.pmax = pmax
        self.exclude = exclude
        self.penalty_factor = penalty_factor
        self.lower_lambdau = lower_lambdau
        self.upper_lambdau = upper_lambdau
        self.maxit = maxit
        self.type_measure = type_measure
        self.family = family
        self.parallel = parallel
        self.ncores = ncores
        self.verbose = verbose
        self.model = None
        self.s = None
        self.coef_ = None

    def fit(self, X, y, s=None, exact=False, **kwargs):
        """
        Fit the model.

        Parameters
        ----------

        X : array-like
            The predictor variables.
        y : array-like
            The response variable.
        s : float, optional
            The value of lambda at which extraction is made. Default is None.
        exact : bool, optional
            Whether to use exact lambda values. Default is False.
        **kwargs : dict
            Additional arguments to pass to glmnetCoef.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.model = glmnet(
            x=X,
            y=y,
            weights=self.weights,
            alpha=self.alpha,
            nlambda=self.nlambda,
            lambdau=self.lambdau,
            standardize=self.standardize,
            thresh=self.thresh,
            dfmax=self.dfmax,
            pmax=self.pmax,
            exclude=self.exclude,
            penalty_factor=self.penalty_factor,
            lower_lambdau=self.lower_lambdau,
            upper_lambdau=self.upper_lambdau,
            maxit=self.maxit,
            type_measure=self.type_measure,
            family=self.family,
            parallel=self.parallel,
            ncores=self.ncores,
            verbose=self.verbose,
        )
        self.coef_ = glmnetCoef(self.model, s=s, exact=exact, **kwargs)
        return self

    def cvglmnet(
        self,
        X,
        y,
        family="gaussian",
        ptype="default",
        nfolds=10,
        foldid=np.empty([0]),
        parallel=1,
        keep=False,
        grouped=True,
        **kwargs
    ):
        warnings.filterwarnings("ignore")
        cvfit = cvglmnet(
            x=X,
            y=y,
            family=family,
            ptype=ptype,
            nfolds=nfolds,
            foldid=foldid,
            parallel=parallel,
            keep=keep,
            grouped=grouped,
            **kwargs
        )

        warnings.filterwarnings("default")
        best_lambda = cvfit["lambda_min"][0]
        best_coef = cvglmnetCoef(cvfit, s=best_lambda).ravel()
        DescribeResult = namedtuple(
            "DescribeResult", ["cvfit", "best_lambda", "best_coef"]
        )
        glmnet_obj = GLMNet()
        glmnet_obj.model = cvfit
        glmnet_obj.coef_ = best_coef
        return DescribeResult(glmnet_obj, best_lambda, best_coef)

    def get_coef(self, s=None, exact=False):
        """
        Get the coefficients.

        Parameters
        ----------
        s : float, optional
            The value of lambda at which extraction is made. Default is None.
        exact : bool, optional
            Whether to use exact lambda values or not. Default is False.

        Returns
        -------
        coef : array-like
            The coefficients.
        """
        if s is None:
            return self.coef_
        assert self.model is not None, "Model not fitted yet."
        return glmnetCoef(self.model, s=s, exact=exact)

    def print(self):
        """
        Print the model's characteristics.
        """
        return glmnetPrint(self.model)

    def plot(self, xvar="lambda", label=True):
        """
        Plot the model's coefficients.

        Parameters
        ----------
        xvar : str, optional
            The variable to plot ("norm" for the L1 norm of coefficients,
            "lambda" for the log-lambda value or "dev" for percentage of
            deviance explained). Default is "lambda".
        label : bool, optional
            Whether to label the plot. Default is True.
        """
        assert xvar in ("norm", "lambda", "dev"), "Invalid input for xvar."
        return glmnetPlot(self.model, xvar=xvar, label=label)

    def predict(self, X, ptype="response", s=None, exact=False, **kwargs):
        """
        Predict the response variable.

        Parameters
        ----------
        X : array-like
            The predictor variables.
        ptype : str
            The type of prediction to make.
            "response" the sames as "link" for "gaussian" family.
            "coefficients" computes the coefficients at values of s
            "nonzero" retuns a list of the indices of the nonzero coefficients for each value of s.
            Default is "response".
        s : float
            The value of lambda at which extraction is made. Default is None.
        exact : bool, optional
            Whether to use exact lambda values or not. Default is False.
        **kwargs : dict
            Additional arguments
        """
        if self.s is None:
            self.s = 0.1
        else:
            self.s = s
        assert ptype in (
            "response",
            "coefficients",
            "nonzero",
        ), "Invalid input for ptype."
        if np.isscalar(self.s):
            res = glmnetPredict(
                self.model,
                X,
                ptype=ptype,
                s=np.asarray([self.s, 0.1]),
                exact=exact,
                **kwargs
            )
            return res[:, 0]
        return glmnetPredict(
            self.model, X, ptype=ptype, s=self.s, exact=exact, **kwargs
        )

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict_log_proba(self, X):
        return self.model.predict_log_proba(X)
