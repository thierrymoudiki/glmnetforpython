from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from .glmnet import glmnet


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

    def fit(self, X, y):
        self.model = glmnet(
            x=X,
            y=y,
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
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict_log_proba(self, X):
        return self.model.predict_log_proba(X)

    def coef_(self):
        return self.model.coef_
