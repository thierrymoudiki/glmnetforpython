# -*- coding: utf-8 -*-
"""
Internal function called by cvglmnet. See also cvglmnet

"""
import scipy
from .glmnetPredict import glmnetPredict
from .wtmean import wtmean
from .cvcompute import cvcompute


def cvlognet(
    fit, lambdau, x, y, weights, offset, foldid, ptype, grouped, keep=False
):

    typenames = {
        "deviance": "Binomial Deviance",
        "mse": "Mean-Squared Error",
        "mae": "Mean Absolute Error",
        "auc": "AUC",
        "class": "Misclassification Error",
    }
    if ptype == "default":
        ptype = "deviance"

    ptypeList = ["mse", "mae", "deviance", "auc", "class"]
    if not ptype in ptypeList:
        print(
            "Warning: only ",
            ptypeList,
            "available for binomial models; " "deviance" " used",
        )
        ptype = "deviance"

    prob_min = 1.0e-5
    prob_max = 1 - prob_min
    nc = y.shape[1]
    if nc == 1:
        classes, sy = np.unique(y, return_inverse=True)
        nc = len(classes)
        indexes = np.eye(nc, nc)
        y = indexes[sy, :]
    else:
        classes = np.arange(nc) + 1  # 1:nc

    N = y.size
    nfolds = np.amax(foldid) + 1
    if (N / nfolds < 10) and (type == "auc"):
        print(
            "Warning: Too few (<10) observations per fold for type.measure=auc in cvlognet"
        )
        print(
            "Warning:     changed to type.measure = deviance. Alternately, use smaller value "
        )
        print("Warning:     for nfolds")
        ptype = "deviance"

    if (N / nfolds < 3) and grouped:
        print(
            "Warning: option grouped = False enforced in cvglmnet as there are < 3 observations per fold"
        )
        grouped = False

    is_offset = not (len(offset) == 0)
    predmat = np.ones([y.shape[0], lambdau.size]) * np.nan
    nfolds = np.amax(foldid) + 1
    nlams = []
    for i in range(nfolds):
        which = foldid == i
        fitobj = fit[i].copy()
        if is_offset:
            off_sub = offset[which,]
        else:
            off_sub = np.empty([0])
        preds = glmnetPredict(
            fitobj, x[which,], np.empty([0]), "response", False, off_sub
        )
        nlami = np.size(fit[i]["lambdau"])
        predmat[which, 0:nlami] = preds
        nlams.append(nlami)
    # convert nlams to scipy array
    nlams = np.asarray(nlams, dtype=np.int64)

    if ptype == "auc":
        cvraw = np.zeros([nfolds, lambdau.size]) * np.nan
        good = np.zeros([nfolds, lambdau.size])
        for i in range(nfolds):
            good[i, 0 : nlams[i]] = 1
            which = foldid == i
            for j in range(nlams[i]):
                cvraw[i, j] = auc_mat(
                    y[which,], predmat[which, j], weights[which]
                )
        N = np.sum(good, axis=0)
        sweights = np.zeros([nfolds, 1])
        for i in range(nfolds):
            sweights[i] = np.sum(weights[foldid == i], axis=0)
        weights = sweights
    else:
        ywt = np.sum(y, axis=1, keepdims=True)
        y = y / np.tile(ywt, [1, y.shape[1]])
        weights = weights * ywt
        N = y.shape[0] - np.sum(np.isnan(predmat), axis=0, keepdims=True)
        yy1 = np.tile(y[:, 0:1], [1, lambdau.size])
        yy2 = np.tile(y[:, 1:2], [1, lambdau.size])

    if ptype == "mse":
        cvraw = (yy1 - (1 - predmat)) ** 2 + (yy2 - (1 - predmat)) ** 2
    elif ptype == "deviance":
        predmat = np.minimum(np.maximum(predmat, prob_min), prob_max)
        lp = yy1 * np.log(1 - predmat) + yy2 * np.log(predmat)
        ly = np.log(y)
        ly[y == 0] = 0
        ly = np.dot(y * ly, np.asarray([1.0, 1.0]).reshape([2, 1]))
        cvraw = 2 * (np.tile(ly, [1, lambdau.size]) - lp)
    elif ptype == "mae":
        cvraw = np.abs(yy1 - (1 - predmat)) + np.abs(yy2 - (1 - predmat))
    elif ptype == "class":
        cvraw = yy1 * (predmat > 0.5) + yy2 * (predmat <= 0.5)

    if y.size / nfolds < 3 and grouped == True:
        print(
            "Option grouped=false enforced in cv.glmnet, since < 3 observations per fold"
        )
        grouped = False

    if grouped == True:
        cvob = cvcompute(cvraw, weights, foldid, nlams)
        cvraw = cvob["cvraw"]
        weights = cvob["weights"]
        N = cvob["N"]

    cvm = wtmean(cvraw, weights)
    sqccv = (cvraw - cvm) ** 2
    cvsd = np.sqrt(wtmean(sqccv, weights) / (N - 1))

    result = dict()
    result["cvm"] = cvm
    result["cvsd"] = cvsd
    result["name"] = typenames[ptype]

    if keep:
        result["fit_preval"] = predmat

    return result


# end of cvelnet
# =========================
#
# =========================
# Helper functions
# =========================
def auc_mat(y, prob, weights=None):
    if weights == None or len(weights) == 0:
        weights = np.ones([y.shape[0], 1])
    wweights = weights * y
    wweights = wweights.flatten()
    wweights = np.reshape(wweights, [1, wweights.size])
    ny = y.shape[0]
    a = np.zeros([ny, 1])
    b = np.ones([ny, 1])
    yy = np.vstack((a, b))
    pprob = np.vstack((prob, prob))
    result = auc(yy, pprob, wweights)
    return result


# =========================
def auc(y, prob, w):
    if len(w) == 0:
        mindiff = np.amin(np.diff(np.unique(prob)))
        pert = np.random.uniform(0, mindiff / 3, prob.size)
        t, rprob = np.unique(prob + pert, return_inverse=True)
        n1 = np.sum(y, keepdims=True)
        n0 = y.shape[0] - n1
        u = np.sum(rprob[y == 1]) - n1 * (n1 + 1) / 2
        result = u / (n1 * n0)
    else:
        op = np.argsort(prob)
        y = y[op]
        w = w[op]
        cw = np.cumsum(w)
        w1 = w[y == 1]
        cw1 = np.cumsum(w1)
        wauc = np.sum(w1 * (cw[y == 1] - cw1))
        sumw = cw1[-1]
        sumw = sumw * (c1[-1] - sumw)
        result = wauc / sumw
    return result


# =========================
