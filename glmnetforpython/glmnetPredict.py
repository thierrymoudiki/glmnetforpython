# -*- coding: utf-8 -*-
"""
--------------------------------------------------------------------------
 glmnetPredict.m: make predictions from a "glmnet" object.
--------------------------------------------------------------------------

 DESCRIPTION:
    Similar to other predict methods, this functions predicts fitted
    values, logits, coefficients and more from a fitted "glmnet" object.

 USAGE:
    glmnetPredict(object, newx, s, type, exact, offset)

    Fewer input arguments(more often) are allowed in the call, but must
    come in the order listed above. To set default values on the way, use
    empty matrix []. 
    For example, pred=glmnetPredict(fit,[],[],'coefficients').
   
    To make EXACT prediction, the input arguments originally passed to 
    "glmnet" MUST be VARIABLES (instead of expressions, or fields
    extracted from some struct objects). Alternatively, users should
    manually revise the "call" field in "object" (expressions or variable
    names) to match the original call to glmnet in the parent environment.

 INPUT ARGUMENTS:
 object      Fitted "glmnet" model object.
 s           Value(s) of the penalty parameter lambda at which predictions
             are required. Default is the entire sequence used to create
             the model.
 newx        scipy 2D array of new values for x at which predictions are to be
             made. Must be a 2D array; can be sparse. This argument is not 
             used for type='coefficients' or type='nonzero'.
 ptype       Type of prediction required. Type 'link' gives the linear
             predictors for 'binomial', 'multinomial', 'poisson' or 'cox'
             models; for 'gaussian' models it gives the fitted values.
             Type 'response' gives the fitted probabilities for 'binomial'
             or 'multinomial', fitted mean for 'poisson' and the fitted
             relative-risk for 'cox'; for 'gaussian' type 'response' is
             equivalent to type 'link'. Type 'coefficients' computes the
             coefficients at the requested values for s. Note that for
             'binomial' models, results are returned only for the class
             corresponding to the second level of the factor response.
             Type 'class' applies only to 'binomial' or 'multinomial'
             models, and produces the class label corresponding to the
             maximum probability. Type 'nonzero' returns a matrix of
             logical values with each column for each value of s, 
             indicating if the corresponding coefficient is nonzero or not.
 exact       If exact=false (default), then the predict function
             uses linear interpolation to make predictions for values of s
             that do not coincide with those used in the fitting
             algorithm. exact = True is not implemented.
 offset      If an offset is used in the fit, then one must be supplied
             for making predictions (except for type='coefficients' or
             type='nonzero')
 
 DETAILS:
    The shape of the objects returned are different for "multinomial"
    objects. glmnetCoef(fit, ...) is equivalent to 
    glmnetPredict(fit,np.empty([]),np.empty([]),'coefficients").

 LICENSE: GPL-2

 AUTHORS:
    Algorithm was designed by Jerome Friedman, Trevor Hastie and Rob Tibshirani
    Fortran code was written by Jerome Friedman
    R wrapper (from which the MATLAB wrapper was adapted) was written by Trevor Hasite
    The original MATLAB wrapper was written by Hui Jiang (14 Jul 2009),
    and was updated and maintained by Junyang Qian (30 Aug 2013) junyangq@stanford.edu,
    Department of Statistics, Stanford University, Stanford, California, USA.

 REFERENCES:
    Friedman, J., Hastie, T. and Tibshirani, R. (2008) Regularization Paths for Generalized Linear Models via Coordinate Descent, 
    http://www.jstatsoft.org/v33/i01/
    Journal of Statistical Software, Vol. 33(1), 1-22 Feb 2010
    
    Simon, N., Friedman, J., Hastie, T., Tibshirani, R. (2011) Regularization Paths for Cox's Proportional Hazards Model via Coordinate Descent,
    http://www.jstatsoft.org/v39/i05/
    Journal of Statistical Software, Vol. 39(5) 1-13

    Tibshirani, Robert., Bien, J., Friedman, J.,Hastie, T.,Simon, N.,Taylor, J. and Tibshirani, Ryan. (2010) Strong Rules for Discarding Predictors in Lasso-type Problems,
    http://www-stat.stanford.edu/~tibs/ftp/strong.pdf
    Stanford Statistics Technical Report

 SEE ALSO:
    glmnet, glmnetPrint, glmnetCoef, and cvglmnet.

EXAMPLES:

    x = np.random.normal(size = [100,20])
    y = np.random.normal(size = [100,1])
    g2 = np.random.choice(2, size = [100, 1])*1.0 # must be float64
    g4 = np.random.choice(4, size = [100, 1])*1.0 # must be float64
    
    fit1 = glmnet(x = x.copy(),y = y.copy());
    print( glmnetPredict(fit1,x[0:5,:],np.asarray([0.01,0.005])) )
    print( glmnetPredict(fit1, np.empty([0]), np.empty([0]), 'coefficients') )
    
    fit2 = glmnet(x = x.copy(), y = g2.copy(), family = 'binomial');
    print(glmnetPredict(fit2, x[2:5,:],np.empty([0]), 'response'))
    print(glmnetPredict(fit2, np.empty([0]), np.empty([0]), 'nonzero'))
       
    fit3 = glmnet(x = x.copy(), y = g4.copy(), family = 'multinomial');
    print(glmnetPredict(fit3, x[0:3,:], np.asarray([0.01, 0.5]), 'response'))
    
"""
import numpy as np
import scipy


def glmnetPredict(
    fit,
    newx=np.empty([0]),
    s=np.empty([0]),
    ptype="link",
    exact=False,
    offset=np.empty([0]),
):

    typebase = ["link", "response", "coefficients", "nonzero", "class"]
    indxtf = [x.startswith(ptype.lower()) for x in typebase]
    indl = [i for i in range(len(indxtf)) if indxtf[i] == True]
    ptype = typebase[indl[0]]

    if newx.shape[0] == 0 and ptype != "coefficients" and ptype != "nonzero":
        raise ValueError("You need to supply a value for " "newx" "")

    # python 1D arrays are not the same as matlab 1xn arrays
    # check for this. newx = x[0:1, :] is a python 2D array and would work;
    # but newx = x[0, :] is a python 1D array and should not be passed into
    # glmnetPredict
    if len(newx.shape) == 1 and newx.shape[0] > 0:
        raise ValueError("newx must be a 2D (not a 1D) python array")

    if exact == True and len(s) > 0:
        # It is very messy to go back into the caller namespace
        # and call glmnet again. The user should really do this at their end
        # by calling glmnet again using the correct array of lambda values that
        # includes the lambda for which prediction is sought
        raise NotImplementedError(
            "exact = True option is not implemented in python"
        )

    # we convert newx to full here since sparse and full operations do not seem to
    # be overloaded completely in np.
    if scipy.sparse.issparse(newx):
        newx = newx.todense()

    # elnet
    if fit["class"] in ["elnet", "fishnet", "lognet"]:
        if fit["class"] == "lognet":
            a0 = fit["a0"]
        else:
            a0 = np.transpose(fit["a0"])

        a0 = np.reshape(a0, [1, a0.size])  # convert to 1 x N for appending
        nbeta = np.vstack((a0, fit["beta"]))
        if np.size(s) > 0:
            lambdau = fit["lambdau"]
            lamlist = lambda_interp(lambdau, s)
            nbeta = nbeta[:, lamlist["left"]] * np.tile(
                np.transpose(lamlist["frac"]), [nbeta.shape[0], 1]
            ) + nbeta[:, lamlist["right"]] * (
                1 - np.tile(np.transpose(lamlist["frac"]), [nbeta.shape[0], 1])
            )

        if ptype == "coefficients":
            result = nbeta
            return result

        if ptype == "nonzero":
            result = nonzeroCoef(nbeta[1 : nbeta.shape[0], :], True)
            return result
        # use np.sparse.hstack instead of column_stack for sparse matrices
        result = np.dot(
            np.column_stack((np.ones([newx.shape[0], 1]), newx)), nbeta
        )

        if fit["offset"]:
            if len(offset) == 0:
                raise ValueError(
                    "No offset provided for prediction, yet used in fit of glmnet"
                )
            if offset.shape[1] == 2:
                offset = offset[:, 1]

            result = result + np.tile(offset, [1, result.shape[1]])

    # fishnet
    if fit["class"] == "fishnet" and ptype == "response":
        result = np.exp(result)

    # lognet
    if fit["class"] == "lognet":
        if ptype == "response":
            pp = np.exp(-result)
            result = 1 / (1 + pp)
        elif ptype == "class":
            result = (result > 0) * 1 + (result <= 0) * 0
            result = fit["label"][result]

    # multnet / mrelnet
    if fit["class"] == "mrelnet" or fit["class"] == "multnet":
        if fit["class"] == "mrelnet":
            if type == "response":
                ptype = "link"
            fit["grouped"] = True

        a0 = fit["a0"]
        nbeta = fit["beta"].copy()
        nclass = a0.shape[0]
        nlambda = s.size

        if len(s) > 0:
            lambdau = fit["lambdau"]
            lamlist = lambda_interp(lambdau, s)
            for i in range(nclass):
                kbeta = np.vstack((a0[i, :], nbeta[i]))
                kbeta = kbeta[:, lamlist["left"]] * np.tile(
                    np.transpose(lamlist["frac"]), [kbeta.shape[0], 1]
                ) + kbeta[:, lamlist["right"]] * (
                    1
                    - np.tile(
                        np.transpose(lamlist["frac"]), [kbeta.shape[0], 1]
                    )
                )
                nbeta[i] = kbeta
        else:
            for i in range(nclass):
                nbeta[i] = np.vstack((a0[i, :], nbeta[i]))
            nlambda = len(fit["lambdau"])

        if ptype == "coefficients":
            result = nbeta
            return result

        if ptype == "nonzero":
            if fit["grouped"]:
                result = list()
                tn = nbeta[0].shape[0]
                result.append(nonzeroCoef(nbeta[0][1:tn, :], True))
            else:
                result = list()
                for i in range(nclass):
                    tn = nbeta[0].shape[0]
                    result.append(nonzeroCoef(nbeta[0][1:tn, :], True))
            return result

        npred = newx.shape[0]
        dp = np.zeros([nclass, nlambda, npred], dtype=np.float64)
        for i in range(nclass):
            qq = np.column_stack((np.ones([newx.shape[0], 1]), newx))
            fitk = np.dot(qq, nbeta[i])
            dp[i, :, :] = dp[i, :, :] + np.reshape(
                np.transpose(fitk), [1, nlambda, npred]
            )

        if fit["offset"]:
            if len(offset) == 0:
                raise ValueError(
                    "No offset provided for prediction, yet used in fit of glmnet"
                )
            if offset.shape[1] != nclass:
                raise ValueError(
                    "Offset should be dimension %d x %d" % (npred, nclass)
                )
            toff = np.transpose(offset)
            for i in range(nlambda):
                dp[:, i, :] = dp[:, i, :] + toff

        if ptype == "response":
            pp = np.exp(dp)
            psum = np.sum(pp, axis=0, keepdims=True)
            result = np.transpose(pp / np.tile(psum, [nclass, 1, 1]), [2, 0, 1])
        if ptype == "link":
            result = np.transpose(dp, [2, 0, 1])
        if ptype == "class":
            dp = np.transpose(dp, [2, 0, 1])
            result = list()
            for i in range(dp.shape[2]):
                t = softmax(dp[:, :, i])
                result = np.append(result, fit["label"][t["pclass"]])

    # coxnet
    if fit["class"] == "coxnet":
        nbeta = fit["beta"]
        if len(s) > 0:
            lambdau = fit["lambdau"]
            lamlist = lambda_interp(lambdau, s)
            nbeta = nbeta[:, lamlist["left"]] * np.tile(
                np.transpose(lamlist["frac"]), [nbeta.shape[0], 1]
            ) + nbeta[:, lamlist["right"]] * (
                1 - np.tile(np.transpose(lamlist["frac"]), [nbeta.shape[0], 1])
            )

        if ptype == "coefficients":
            result = nbeta
            return result

        if ptype == "nonzero":
            result = nonzeroCoef(nbeta, True)
            return result

        result = np.dot(newx, nbeta)

        if fit["offset"]:
            if len(offset) == 0:
                raise ValueError(
                    "No offset provided for prediction, yet used in fit of glmnet"
                )

            result = result + np.tile(offset, [1, result.shape[1]])

        if ptype == "response":
            result = np.exp(result)

    return result


# end of glmnetPredict
# =========================================


# =========================================
# helper functions
# =========================================
def lambda_interp(lambdau, s):
    # lambda is the index sequence that is produced by the model
    # s is the new vector at which evaluations are required.
    # the value is a vector of left and right indices, and a vector of fractions.
    # the new values are interpolated bewteen the two using the fraction
    # Note: lambda decreases. you take:
    # sfrac*left+(1-sfrac*right)
    if len(lambdau) == 1:
        nums = len(s)
        left = np.zeros([nums, 1], dtype=np.int64)
        right = left
        sfrac = np.zeros([nums, 1], dtype=np.float64)
    else:
        s[s > np.amax(lambdau)] = np.amax(lambdau)
        s[s < np.amin(lambdau)] = np.amin(lambdau)
        k = len(lambdau)
        sfrac = (lambdau[0] - s) / (lambdau[0] - lambdau[k - 1])
        lambdau = (lambdau[0] - lambdau) / (lambdau[0] - lambdau[k - 1])
        coord = scipy.interpolate.interp1d(lambdau, range(k))(sfrac)
        left = np.floor(coord).astype(np.int64, copy=False)
        right = np.ceil(coord).astype(np.int64, copy=False)
        #
        tf = left != right
        sfrac[tf] = (sfrac[tf] - lambdau[right[tf]]) / (
            lambdau[left[tf]] - lambdau[right[tf]]
        )
        sfrac[~tf] = 1.0
        # if left != right:
        #    sfrac = (sfrac - lambdau[right])/(lambdau[left] - lambdau[right])
        # else:
        #    sfrac[left == right] = 1.0

    result = dict()
    result["left"] = left
    result["right"] = right
    result["frac"] = sfrac

    return result


# end of lambda_interp
# =========================================
def softmax(x, gap=False):
    d = x.shape
    maxdist = x[:, 0]
    pclass = np.zeros([d[0], 1], dtype=np.int64)
    for i in range(1, d[1], 1):
        l = x[:, i] > maxdist
        pclass[l] = i
        maxdist[l] = x[l, i]
    if gap == True:
        x = np.abs(maxdist - x)
        x[0 : d[0], pclass] = x * np.ones([d[1], d[1]])
        # gaps = pmin(x)# not sure what this means; gap is never called with True
        raise ValueError("gap = True is not implemented yet")

    result = dict()
    if gap == True:
        result["pclass"] = pclass
        # result['gaps'] = gaps
        raise ValueError("gap = True is not implemented yet")
    else:
        result["pclass"] = pclass

    return result


# end of softmax
# =========================================
def nonzeroCoef(beta, bystep=False):
    result = np.abs(beta) > 0
    if not bystep:
        result = np.any(result, axis=1)
    return result


# end of nonzeroCoef
# =========================================
