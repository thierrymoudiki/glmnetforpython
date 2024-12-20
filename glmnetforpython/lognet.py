# -*- coding: utf-8 -*-
"""
Internal function called by glmnet. See also glmnet, cvglmnet

"""
# import packages/methods
import numpy as np
import scipy
import ctypes
from .loadGlmLib import loadGlmLib


def lognet(
    x,
    is_sparse,
    irs,
    pcs,
    y,
    weights,
    offset,
    parm,
    nobs,
    nvars,
    jd,
    vp,
    cl,
    ne,
    nx,
    nlam,
    flmin,
    ulam,
    thresh,
    isd,
    intr,
    maxit,
    kopt,
    family,
):

    # load shared fortran library
    glmlib = loadGlmLib()

    #
    noo = y.shape[0]
    if len(y.shape) > 1:
        nc = y.shape[1]
    else:
        nc = 1

    if noo != nobs:
        raise ValueError(
            "x and y have different number of rows in call to glmnet"
        )

    if nc == 1:
        classes, sy = np.unique(y, return_inverse=True)
        nc = len(classes)
        indexes = np.eye(nc, nc)
        y = indexes[sy, :]
    else:
        classes = np.arange(nc) + 1  # 1:nc
    #
    if family == "binomial":
        if nc > 2:
            raise ValueError(
                "More than two classes in y. use multinomial family instead"
            )
        else:
            nc = 1
            y = y[:, [1, 0]]
    #
    if len(weights) != 0:
        t = weights > 0
        if ~np.all(t):
            t = np.reshape(t, (len(y),))
            y = y[t, :]
            x = x[t, :]
            weights = weights[t]
            nobs = np.sum(t)
        else:
            t = np.empty([0], dtype=np.int64)
        #
        if len(y.shape) == 1:
            mv = len(y)
            ny = 1
        else:
            mv, ny = y.shape

        y = y * np.tile(weights, (1, ny))

    #
    if len(offset) == 0:
        offset = y * 0
        is_offset = False
    else:
        if len(t) != 0:
            offset = offset[t, :]
        do = offset.shape
        if do[0] != nobs:
            raise ValueError(
                "offset should have the same number of values as observations in binominal/multinomial call to glmnet"
            )
        if nc == 1:
            if do[1] == 1:
                offset = np.column_stack((offset, -offset), 1)
            if do[1] > 2:
                raise ValueError(
                    "offset should have 1 or 2 columns in binomial call to glmnet"
                )
        if (family == "multinomial") and (do[1] != nc):
            raise ValueError(
                "offset should have same shape as y in multinomial call to glmnet"
            )
        is_offset = True

    # now convert types and allocate memory before calling
    # glmnet fortran library
    ######################################
    # --------- PROCESS INPUTS -----------
    ######################################
    # force inputs into fortran order and scipy float64
    copyFlag = False
    x = x.astype(dtype=np.float64, order="F", copy=copyFlag)
    irs = irs.astype(dtype=np.int32, order="F", copy=copyFlag)
    pcs = pcs.astype(dtype=np.int32, order="F", copy=copyFlag)
    y = y.astype(dtype=np.float64, order="F", copy=copyFlag)
    weights = weights.astype(dtype=np.float64, order="F", copy=copyFlag)
    offset = offset.astype(dtype=np.float64, order="F", copy=copyFlag)
    jd = jd.astype(dtype=np.int32, order="F", copy=copyFlag)
    vp = vp.astype(dtype=np.float64, order="F", copy=copyFlag)
    cl = cl.astype(dtype=np.float64, order="F", copy=copyFlag)
    ulam = ulam.astype(dtype=np.float64, order="F", copy=copyFlag)

    ######################################
    # --------- ALLOCATE OUTPUTS ---------
    ######################################
    # lmu
    lmu = -1
    lmu_r = ctypes.c_int(lmu)
    # a0, ca
    if nc == 1:
        a0 = np.zeros([nlam], dtype=np.float64)
        ca = np.zeros([nx, nlam], dtype=np.float64)
    else:
        a0 = np.zeros([nc, nlam], dtype=np.float64)
        ca = np.zeros([nx, nc, nlam], dtype=np.float64)
    # a0
    a0 = a0.astype(dtype=np.float64, order="F", copy=False)
    a0_r = a0.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    # ca
    ca = ca.astype(dtype=np.float64, order="F", copy=False)
    ca_r = ca.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    # ia
    ia = -1 * np.ones([nx], dtype=np.int32)
    ia = ia.astype(dtype=np.int32, order="F", copy=False)
    ia_r = ia.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    # nin
    nin = -1 * np.ones([nlam], dtype=np.int32)
    nin = nin.astype(dtype=np.int32, order="F", copy=False)
    nin_r = nin.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    # dev
    dev = -1 * np.ones([nlam], dtype=np.float64)
    dev = dev.astype(dtype=np.float64, order="F", copy=False)
    dev_r = dev.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    # alm
    alm = -1 * np.ones([nlam], dtype=np.float64)
    alm = alm.astype(dtype=np.float64, order="F", copy=False)
    alm_r = alm.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    # nlp
    nlp = -1
    nlp_r = ctypes.c_int(nlp)
    # jerr
    jerr = -1
    jerr_r = ctypes.c_int(jerr)
    # dev0
    dev0 = -1
    dev0_r = ctypes.c_double(dev0)

    #  ###################################
    #   main glmnet fortran caller
    #  ###################################
    if is_sparse:
        # sparse lognet
        glmlib.splognet_(
            ctypes.byref(ctypes.c_double(parm)),
            ctypes.byref(ctypes.c_int(nobs)),
            ctypes.byref(ctypes.c_int(nvars)),
            ctypes.byref(ctypes.c_int(nc)),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            pcs.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            irs.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            offset.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            jd.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            vp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            cl.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(ctypes.c_int(ne)),
            ctypes.byref(ctypes.c_int(nx)),
            ctypes.byref(ctypes.c_int(nlam)),
            ctypes.byref(ctypes.c_double(flmin)),
            ulam.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(ctypes.c_double(thresh)),
            ctypes.byref(ctypes.c_int(isd)),
            ctypes.byref(ctypes.c_int(intr)),
            ctypes.byref(ctypes.c_int(maxit)),
            ctypes.byref(ctypes.c_int(kopt)),
            ctypes.byref(lmu_r),
            a0_r,
            ca_r,
            ia_r,
            nin_r,
            ctypes.byref(dev0_r),
            dev_r,
            alm_r,
            ctypes.byref(nlp_r),
            ctypes.byref(jerr_r),
        )
    else:
        # call fortran lognet routine
        glmlib.lognet_(
            ctypes.byref(ctypes.c_double(parm)),
            ctypes.byref(ctypes.c_int(nobs)),
            ctypes.byref(ctypes.c_int(nvars)),
            ctypes.byref(ctypes.c_int(nc)),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            offset.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            jd.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            vp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            cl.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(ctypes.c_int(ne)),
            ctypes.byref(ctypes.c_int(nx)),
            ctypes.byref(ctypes.c_int(nlam)),
            ctypes.byref(ctypes.c_double(flmin)),
            ulam.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(ctypes.c_double(thresh)),
            ctypes.byref(ctypes.c_int(isd)),
            ctypes.byref(ctypes.c_int(intr)),
            ctypes.byref(ctypes.c_int(maxit)),
            ctypes.byref(ctypes.c_int(kopt)),
            ctypes.byref(lmu_r),
            a0_r,
            ca_r,
            ia_r,
            nin_r,
            ctypes.byref(dev0_r),
            dev_r,
            alm_r,
            ctypes.byref(nlp_r),
            ctypes.byref(jerr_r),
        )

    #  ###################################
    #   post process results
    #  ###################################

    # check for error
    if jerr_r.value > 0:
        raise ValueError(
            "Fatal glmnet error in library call : error code = ", jerr_r.value
        )
    elif jerr_r.value < 0:
        print(
            "Warning: Non-fatal error in glmnet library call: error code = ",
            jerr_r.value,
        )
        print("Check results for accuracy. Partial or no results returned.")

    # clip output to correct sizes
    lmu = lmu_r.value
    if nc == 1:
        a0 = a0[0:lmu]
        ca = ca[0:nx, 0:lmu]
    else:
        a0 = a0[0:nc, 0:lmu]
        ca = ca[0:nx, 0:nc, 0:lmu]
    ia = ia[0:nx]
    nin = nin[0:lmu]
    dev = dev[0:lmu]
    alm = alm[0:lmu]

    # ninmax
    ninmax = max(nin)
    # fix first value of alm (from inf to correct value)
    if ulam[0] == 0.0:
        t1 = np.log(alm[1])
        t2 = np.log(alm[2])
        alm[0] = np.exp(2 * t1 - t2)
    # create return fit dictionary

    if family == "multinomial":
        a0 = a0 - np.tile(np.mean(a0), (nc, 1))
        dfmat = a0.copy()
        dd = np.asarray([nvars, lmu], dtype=np.int64)
        beta_list = list()
        if ninmax > 0:
            # TODO: is the reshape here done right?
            ca = np.reshape(ca, (nx, nc, lmu))
            ca = ca[0:ninmax, :, :]
            ja = ia[0:ninmax] - 1  # ia is 1-indexed in fortran
            oja = np.argsort(ja)
            ja1 = ja[oja]
            df = np.any(np.abs(ca) > 0, axis=1)
            df = np.sum(df)
            df = np.reshape(df, (1, df.size))
            for k in range(0, nc):
                ca1 = np.reshape(ca[:, k, :], (ninmax, lmu))
                cak = ca1[oja, :]
                dfmat[k, :] = np.sum(np.abs(cak) > 0, axis=0)
                beta = np.zeros([nvars, lmu], dtype=np.float64)
                beta[ja1, :] = cak
                beta_list.append(beta)
        else:
            for k in range(0, nc):
                dfmat[k, :] = np.zeros([1, lmu], dtype=np.float64)
                beta_list.append(np.zeros([nvars, lmu], dtype=np.float64))
            #
            df = np.zeros([1, lmu], dtype=np.float64)
        #
        if kopt == 2:
            grouped = True
        else:
            grouped = False
        #
        fit = dict()
        fit["a0"] = a0
        fit["label"] = classes
        fit["beta"] = beta_list
        fit["dev"] = dev
        fit["nulldev"] = dev0_r.value
        fit["dfmat"] = dfmat
        fit["df"] = df
        fit["lambdau"] = alm
        fit["npasses"] = nlp_r.value
        fit["jerr"] = jerr_r.value
        fit["dim"] = dd
        fit["grouped"] = grouped
        fit["offset"] = is_offset
        fit["class"] = "multnet"
    else:
        dd = np.asarray([nvars, lmu], dtype=np.int64)
        if ninmax > 0:
            ca = ca[0:ninmax, :]
            df = np.sum(np.abs(ca) > 0, axis=0)
            ja = ia[0:ninmax] - 1
            # ia is 1-indexes in fortran
            oja = np.argsort(ja)
            ja1 = ja[oja]
            beta = np.zeros([nvars, lmu], dtype=np.float64)
            beta[ja1, :] = ca[oja, :]
        else:
            beta = np.zeros([nvars, lmu], dtype=np.float64)
            df = np.zeros([1, lmu], dtype=np.float64)
        #
        fit = dict()
        fit["a0"] = a0
        fit["label"] = classes
        fit["beta"] = beta
        fit["dev"] = dev
        fit["nulldev"] = dev0_r.value
        fit["df"] = df
        fit["lambdau"] = alm
        fit["npasses"] = nlp_r.value
        fit["jerr"] = jerr_r.value
        fit["dim"] = dd
        fit["offset"] = is_offset
        fit["class"] = "lognet"

    #  ###################################
    #   return to caller
    #  ###################################

    return fit


# -----------------------------------------
# end of method lognet
# -----------------------------------------
