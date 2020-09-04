import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.utils import check_X_y,check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.extmath import pinvh, safe_sparse_dot
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.optimize import fmin_l_bfgs_b
from scipy.linalg import solve_triangular
from scipy.stats import norm
from numpy.linalg import LinAlgError
from numpy import linalg as LA
import warnings
import time
from rtree import index

DEFAULT_DECISION_THRESHOLD = 0.01 # Decision threshold parameter e
UNKNOWN_PROB_PARAM = 0.0 # Param for occupancy probability in unknown regions: b


###############################################################################
#                  Helper Functions
###############################################################################

def update_precisions(Q,S,q,s,A,active,tol,n_samples, update_k_nearest = False, local_data_idx = 0):
    '''
    Selects features to add/recompute/delete to model based on effect it will have on value of log marginal likelihood.
    '''
    # initialise vector holding changes in log marginal likelihood
    deltaL = np.zeros(Q.shape[0])
    
    # identify features that can be added , recomputed and deleted in model
    theta        =  q**2 - s 
    add          =  (theta > 0) * (active == False)
    recompute    =  (theta > 0) * (active == True)
    delete       = ~(add + recompute)
    
    # compute sparsity & quality parameters corresponding to features in 
    # three groups identified above
    Qadd,Sadd      = Q[add], S[add]
    Qrec,Srec,Arec = Q[recompute], S[recompute], A[recompute]
    Qdel,Sdel,Adel = Q[delete], S[delete], A[delete]
    
    # compute new alpha's (precision parameters) for features that are 
    # currently in model and will be recomputed
    Anew           = s[recompute]**2/ ( theta[recompute] + np.finfo(np.float32).eps)
    delta_alpha    = (1./Anew - 1./Arec)
    
    # compute change in log marginal likelihood 
    deltaL[add]       = ( Qadd**2 - Sadd ) / Sadd + np.log(Sadd/Qadd**2 )
    deltaL[recompute] = Qrec**2 / (Srec + 1. / delta_alpha) - np.log(1 + Srec*delta_alpha)
    deltaL[delete]    = Qdel**2 / (Sdel - Adel) - np.log(1 - Sdel / Adel)
    deltaL            = deltaL  / n_samples


    # find feature which caused largest change in likelihood
    min_A = 0.001 # To avoid numerical instability
    if not update_k_nearest:
        feature_index = np.argmax(deltaL[:])
    else:
        feature_index = local_data_idx + np.argmax(deltaL[local_data_idx:])
        if local_data_idx > 0:
            rv_feature_index = np.argmax(deltaL[0:local_data_idx])
            A[rv_feature_index] = max(s[rv_feature_index] ** 2 / theta[rv_feature_index], min_A)
    # no deletions or additions
    same_features  = np.sum( theta[~recompute] > 0) == 0 #False #
    
    # changes in precision for features already in model is below threshold
    no_delta       = np.sum( abs( Anew[:] - Arec[:]) > tol ) == 0

    # check convergence: if no features to add or delete and small change in 
    #                    precision for current features then terminate
    converged = False
    if same_features and no_delta:
        converged = True
        print("Converge!")
        #print(sum(abs( Anew - Arec )), tol)
        return [A,converged]

    # if not converged update precision parameter of weights and return
    if theta[feature_index] > 0:
        A[feature_index] = max(s[feature_index]**2 / theta[feature_index], min_A)
        if active[feature_index] == False:
            active[feature_index] = True
    elif feature_index >= local_data_idx:
        # at least two active features
        if active[feature_index] == True and np.sum(active) >= 2:
            active[feature_index] = False
            A[feature_index]      = np.PINF
    return [A,converged]



def _gaussian_cost_grad(X,Y,w,diagA):
    '''
        Calculates cost and gradient of the log-likelihood for probit regression
    '''
    n = X.shape[0]
    Xw = np.dot(X, w)
    t = (Y-0.5)*2
    s = norm.cdf(Xw)
    cost = -(np.sum(np.log(s[Y==1] + 1e-6), 0) + \
             np.sum(np.log(1-s[Y==0]+ 1e-6), 0))
    cost = cost + 0.5*(diagA*(w**2))

    temp = norm.pdf(Xw*t)*t/norm.cdf(Xw*t)
    grad = diagA*w - np.dot(X.T, temp)
    return cost / n, grad / n


def get_kernel(X, Y, gamma, kernel):
    '''
    Calculates kernel features
    '''
    params = {"gamma": gamma}
    return pairwise_kernels(X, Y, metric=kernel,
                            filter_params=True, **params)

###############################################################################
#                  Incremental Probit Relevance Vector Classification
###############################################################################
class ProbitRVC(BaseEstimator,LinearClassifierMixin):
    '''
    Sparse Bayesian Kernel-based Map using Incremental Probit Relevance Vector Classification training

    Parameters
    ----------
    n_iter (int):
        Maximum number of iterations before termination
    tol (float):
        The algorithm terminates if change in precision parameter xi is less than tol.
    n_iter_solver (int):
        Maximum number of iterations for Laplace approximation solver
    tol_solver (float):
        Convergence threshold for Laplace approximation solver
    fixed_intercept (float):
        The fixed bias b in our paper whose \sigma(b) represents the occupancy probability of points in unknown regions.
        By default, it is set to the global variable UNKNOWN_PROB_PARAM.
    gamma (float):
        The rbf kernel parameter exp(-gamma(x-x')^2)

    Attributes
    ----------
    kernel (str):
        Kernel function, set to 'rbf'

    gamma (float):
        RBF kernel parameter k(x,y) = exp(-gamma*|x-y|^2)

    prev_X (array of size [n_samples, n_features]:
        Data points from the previous laser scan. This is used if only the delta between two consecutive laser scans is
        used for training.

    prev_y (array of size [n_samples, 1]:
       Label for the previous points prev_X from previous laser scan.
    '''
    def __init__(self, n_iter=50, tol=1e-1, n_iter_solver=15, normalize=False,
                 tol_solver=1e-2, A_thres = 10000.0, fixed_intercept = UNKNOWN_PROB_PARAM):
        self.n_iter             = n_iter
        self.tol                = tol
        print("Init ", self.tol)
        self.n_iter_solver      = n_iter_solver
        self.normalize          = normalize
        self.tol_solver         = tol_solver
        # As mentioned in the paper, the bias represent the occupancy probability of points in unknown regions and
        # and should not be trained.
        self.fixed_intercept = fixed_intercept
        self.prev_trained = False
        self.rtree_index = index.Index()
        self.relevant_vectors_dict = {}
        self.relevant_vectors_local = None
        self.A_thres = A_thres

    
    def fit(self,X,y):
        '''
        Fits Logistic Regression with ARD
        
        Parameters
        ----------
        X (array of size [n_samples, n_features]):
           Training data.
        
        y (array of size [n_samples, ])
           Labels for training data.
           
        Returns
        -------
        self : object
            self.
        '''

        X, y = check_X_y(X, y, accept_sparse = False, dtype=np.float64)

        # normalize, if required
        if self.normalize:
            self._x_mean = np.mean(X,0)
            self._x_std  = np.std(X,0)
            X            = (X - self._x_mean) / self._x_std

        # preprocess targets
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        if n_classes != 2:
            raise ValueError("Currently, we only support binary classification!")
        self.coef_, self.sigma_, self.intercept_,self.active_ = [0],[[0]],[0],[0]
        self.lambda_                                          = [0]
         
        for i in range(len(self.classes_)):
            pos_class = self.classes_[1]
            mask = (y == pos_class)
            y_bin = np.zeros(y.shape, dtype=np.float64)
            y_bin[mask] = 1
            active, lambda_ = self._fit(X,y_bin)
            self.intercept_[i] = self.fixed_intercept
            self.active_[i], self.lambda_[i] = active, lambda_
            # in case of binary classification fit only one classifier
            if n_classes == 2:
                break  
        self.intercept_ = np.asarray(self.intercept_)
        self.prev_trained = True
        return self
        
    
    def _fit(self,X,y):
        '''
        Fits binary classification
        '''
        n_samples,n_features = X.shape
        A         = np.PINF * np.ones(n_features)
        active    = np.zeros(n_features , dtype = np.bool)
        # If there is an existing model trained previously.
        if self.prev_trained:
            active[0:self.prev_rvcount] = True
            A[0:self.prev_rvcount] = self.prev_A
            active[self.prev_rvcount] = True
            A[self.prev_rvcount] = 1
        else:
            active[0] = True
            A[0] = 0.001

        warning_flag = 0
        for i in range(self.n_iter):
            Xa      =  X[:,active]
            Aa      =  A[active]

            # mean & precision of posterior distribution
            Mn,Sn,B,t_hat, cholesky = self._posterior_dist_local(Xa,y, Aa)

            if not cholesky:
                warning_flag += 1
            
            # raise warning in case cholesky failes (but only once)
            if warning_flag == 1:
                warnings.warn(("Cholesky decomposition failed ! Algorithm uses pinvh now!"))
            # compute quality & sparsity parameters
            s,q,S,Q = self._sparsity_quality(X,Xa,t_hat,B,Aa,active,Sn,cholesky)
            # update precision parameters of coefficients
            #
            local_data_idx = self.prev_rvcount if self.prev_trained else 0
            A,converged  = update_precisions(Q,S,q,s,A,active,self.tol,n_samples, local_data_idx=local_data_idx)
            # terminate if converged
            if converged or i == self.n_iter - 1:
                break
        for j in range(len(A)):
            if j >= local_data_idx and A[j] > self.A_thres:
                A[j] = np.PINF
                active[j] = False
        return active, A

    def _sparsity_quality(self, X, Xa, y, B, Aa, active, Sn, cholesky):
        '''
        Calculates sparsity & quality parameters for each feature
        '''
        XB = X.T * B
        bxx = np.matmul(B, X ** 2)
        bxy = np.matmul(XB, y)
        YB = y * B
        if cholesky:
            # Here Sn is inverse of lower triangular matrix, obtained from
            # cholesky decomposition
            XBX = np.matmul(XB, Xa)
            XBXS = np.matmul(XBX, Sn.T)
            SXBY = np.matmul(Sn, np.matmul(Xa.T,YB))
            S = bxx - np.sum(XBXS ** 2, 1)
            Q = bxy - np.sum(XBXS*SXBY.T,1)
        else:
            XSX = np.dot(np.dot(Xa, Sn), Xa.T)
            S = bxx - np.sum(np.dot(XB, XSX) * XB, 1)
            XBXSX = np.matmul(XB, XSX)

            Q = bxy - np.matmul(XBXSX, YB)
        qi = np.copy(Q)
        si = np.copy(S)
        Qa, Sa = Q[active], S[active]
        qi[active] = Aa * Qa / (Aa - Sa)
        si[active] = Aa * Sa / (Aa - Sa)
        return [si, qi, S, Q]

    def _posterior_dist_local(self, X, y, A, tol_mul = 1.0):
        '''
        Uses Laplace approximation for calculating posterior distribution for local relevance vectors
        '''
        f_full = lambda w: _gaussian_cost_grad(X, y, w, A)
        attempts = 1
        a = -2
        b = 2
        for i in range(attempts):
            w_init = a + np.random.random(X.shape[1])*(b-a)
            Mn = fmin_l_bfgs_b(f_full, x0=w_init, pgtol=tol_mul * self.tol_solver,
                               maxiter=int(self.n_iter_solver / tol_mul))[0]
            check_sign = [0 if Mn[j]*(y[j] - 0.5) >= 0 else 1 for j in range(len(Mn))]
            if sum(check_sign)/len(Mn) < 0.1:
                break

        Xm_nobias = np.dot(X, Mn)
        Xm = Xm_nobias + self.fixed_intercept
        t = (y - 0.5) * 2
        eta = norm.pdf(t * Xm) * t / norm.cdf(Xm * t) + 1e-300
        B = eta * (Xm + eta)
        S = np.dot(X.T * B, X)
        np.fill_diagonal(S, np.diag(S) + A)
        t_hat = Xm_nobias + eta / B
        cholesky = True
        # try using Cholesky , if it fails then fall back on pinvh
        try:
            R = np.linalg.cholesky(S)
            Sn = solve_triangular(R, np.eye(A.shape[0]),
                                  check_finite=False, lower=True)
        except LinAlgError:
            Sn = pinvh(S)
            cholesky = False
        return [Mn, Sn, B, t_hat, cholesky]

    def _posterior_dist_global(self, X, y, A, tol_solver, n_iter_solver):
        '''
        Uses Laplace approximation for calculating posterior distribution for all relevance vectors.
        '''
        f_full = lambda w: _gaussian_cost_grad(X, y, w, A)
        attempts = 10
        a = -2
        b = 2
        # Sometimes, fmin_l_bfgs_b fails to find a good minimizer. Retry with different initial point.
        for i in range(attempts):
            w_init = a + np.random.random(X.shape[1])*(b-a)
            Mn = fmin_l_bfgs_b(f_full, x0=w_init, pgtol=tol_solver, maxiter=n_iter_solver)[0]
            check_sign = [0 if Mn[j]*(y[j] - 0.5) >= 0 else 1 for j in range(len(Mn))]
            if sum(check_sign)/len(Mn) < 0.1:
                break
        Xm = np.dot(X, Mn) + self.fixed_intercept
        t = (y - 0.5) * 2
        eta = norm.pdf(t * Xm) * t / norm.cdf(Xm * t) + 1e-300
        S = np.matmul(X.T * eta * (Xm + eta), X) + np.diag(A)
        cholesky = True
        # try using Cholesky , if it fails then fall back on pinvh
        try:
            R = np.linalg.cholesky(S)
            Sn = solve_triangular(R, np.eye(A.shape[0]),
                                  check_finite=False, lower=True)
        except LinAlgError:
            Sn = pinvh(S)
            cholesky = False
        return [Mn, Sn, cholesky]



###############################################################################
#                  Sparse Bayesian Kernel-based Map
###############################################################################
class SBKM(ProbitRVC):
    '''
    Sparse Bayesian Kernel-based Map using Incremental Probit Relevance Vector Classification training

    Parameters
    ----------
    n_iter (int):
        Maximum number of iterations before termination
    tol (float):
        The algorithm terminates if change in precision parameter xi is less than tol.
    n_iter_solver (int):
        Maximum number of iterations for Laplace approximation solver
    tol_solver (float):
        Convergence threshold for Laplace approximation solver
    fixed_intercept (float):
        The fixed bias b in our paper whose \sigma(b) represents the occupancy probability of points in unknown regions.
        By default, it is set to the global variable UNKNOWN_PROB_PARAM.
    gamma (float):
        The rbf kernel parameter exp(-gamma(x-x')^2)

    Attributes
    ----------
    kernel (str):
        Kernel function, set to 'rbf'

    gamma (float):
        RBF kernel parameter k(x,y) = exp(-gamma*|x-y|^2)

    prev_X (array of size [n_samples, n_features]:
        Data points from the previous laser scan. This is used if only the delta between two consecutive laser scans is
        used for training.

    prev_y (array of size [n_samples, 1]:
       Label for the previous points prev_X from previous laser scan.
    '''
    def __init__(self, n_iter = 50, tol = 1e-2, n_iter_solver = 50, tol_solver = 1e-3, fixed_intercept = UNKNOWN_PROB_PARAM, gamma  = None):
        super(SBKM,self).__init__(n_iter,tol,n_iter_solver,False,tol_solver, A_thres = 100000.0, fixed_intercept=fixed_intercept)
        print("Init RVC", self.tol)
        self.kernel        = 'rbf'
        self.gamma         = gamma
        self.prev_X = None
        self.prev_y = None

    def gen_dataset(self, X, y, pres = None, nres = None):
        '''
        Generate local training dataset from laser scans on a grid. Note that this is optional and the training data
        points do not need to be on a grid for our algorithm to work.

        Parameters
        -----------
        X (array of size [n_samples, n_features]):
            Training data containing n_samples data points.

        y (array of size [n_samples, 1])
            Labels where "0" means "free" and "1" means "occupied".

        pres (float):
            Resolution for sampling positive (occupied) data points on a grid.

        nres (float):
            Resolution for sampling negative (free) data points on a grid.

        Returns
        -------
        X (array of size [n_samples, n_features]):
            Training data containing n_samples data points on a grid.

        y (array of size [n_samples, 1])
            Labels where "0" means "free" and "1" means "occupied" on a grid.
        '''
        if pres is not None and nres is not None:
            # Sample occupied (label "1") points
            X[y > 0.5] = pres * np.around(X[y > 0.5] / pres)
            # Create a dictionary of occupied data points.
            # This is used to avoid mislabeled points due to rounding the points on a grid.
            X_new_pos = np.unique(X[y > 0.5], axis=0)
            X_new_pos_int_dict = {}
            for k in range(len(X_new_pos)):
                X_new_pos_int_dict[(int(np.rint(X_new_pos[k, 0] / pres)), int(np.rint(X_new_pos[k, 1] / pres)))] = 1
            # Sample free (label "0") points
            X[y < 0.5] = nres * np.around(X[y < 0.5] / nres)
            X_new_neg = np.unique(X[y < 0.5], axis=0)
            # Avoid mislabeling positive points.
            X_new_neg_reduced = []
            for k in range(len(X_new_neg)):
                if (int(np.rint(X_new_neg[k, 0] / pres)), int(np.rint(X_new_neg[k, 1] / pres))) in X_new_pos_int_dict:
                    continue
                X_new_neg_reduced.append([X_new_neg[k, 0], X_new_neg[k, 1]])
            X_new_neg_reduced = np.array(X_new_neg_reduced)
            # Stack the negative and positive points to create a local dataset.
            X = np.vstack((X_new_pos, X_new_neg_reduced))
            y = np.hstack(
                (np.ones(len(X_new_pos), dtype=float), np.zeros(len(X_new_neg_reduced), dtype=float)))
        return X, y



    def fit(self,X,y, pos_neg_rescale = True, k_nearest = 50, use_diff = True, pres = None, nres = None):
        '''
        Update our SBKM map as a set of relevance vectors.

        Parameters
        -----------
        X (array of size [n_samples, n_features]):
            Training data containing n_samples data points.

        y (array of size [n_samples, 1])
            Labels where "0" means "free" and "1" means "occupied".

        pos_neg_rescale (bool):
            If negative (free) points dominates the training set, positive (occupied) points are less likely to be added
            to our sparse set of relevance vectors due to their smaller contribution to the data log-likelihood, and
            vice versa. This parameter allows us to rescale the number of positive and negative data points.

        k_nearest (int):
            The number of nearest relevance vectors used in our training algorithm.

        use_diff (bool):
            To improve training time, the difference between two consecutive laser scans is used to update the set of
            relevance vectors instead.

        pres (float):
            Resolution for sampling positive (occupied) data points on a grid.

        nres (float):
            Resolution for sampling negative (free) data points on a grid.

        Returns
        -------
        self (class object)
           self
        '''
        time0 = time.time()
        X, y = self.gen_dataset(X, y, pres = pres, nres = nres)
        self.cur_X = np.copy(X).tolist()
        self.cur_y = np.copy(y).tolist()

        if self.prev_trained:
            if use_diff:
                Xdiff = np.array([X[i,:] if X[i,:].tolist() not in self.prev_X else None for i in range(len(X))])
                ydiff = np.array([y[i] if X[i, :].tolist() not in self.prev_X else None for i in range(len(X))])
                Xdiff = Xdiff[ydiff != None]
                ydiff = ydiff[ydiff != None]
                Xdiff = np.concatenate(Xdiff).reshape((len(Xdiff),2))
                ydiff = ydiff.astype(np.int64)
                if len(ydiff) == 0:
                    print("Data is the same! Skip...")
                    return self
            else:
                # Use full local training dataset at the begining when our map is not trained yet.
                Xdiff = X
                ydiff = y

            # Query k nearest relevance vectors to the mean of the data points from rtree*
            # This is a local set of relevance vectors used for our training
            X_mean = np.mean(Xdiff, axis=0)
            nearest_rv = self.rtree_index.nearest((X_mean[0], X_mean[1], X_mean[0], X_mean[1]), k_nearest, objects=True)
            local_rv = []
            local_rv_label = []
            local_A = []
            for rv in nearest_rv:
                local_rv.append([rv.bbox[0], rv.bbox[1]])
                local_rv_label.append(self.relevant_vectors_dict[(rv.bbox[0], rv.bbox[1])][0])
                local_A.append(self.relevant_vectors_dict[(rv.bbox[0], rv.bbox[1])][1])
            self.relevant_vectors_local = local_rv
            self.prev_rvcount = len(local_rv)
            self.prev_A = local_A

            # Duplicate training data points to achieve fair portions for positive (occupied) and negative (free) points
            if pos_neg_rescale:
                ydiff_pos = ydiff[ydiff > 0.5]
                Xdiff_pos = Xdiff[ydiff > 0.5, :]
                ydiff_neg = ydiff[ydiff < 0.5]
                Xdiff_neg = Xdiff[ydiff < 0.5, :]
                pos_portion = sum(ydiff[ydiff > 0])
                neg_portion = len(ydiff) - pos_portion
                if (pos_portion > 0):
                    r = neg_portion / pos_portion
                    if r > 1:
                        r = int(np.rint(r))
                        for i in range(r):
                            Xdiff = np.vstack((Xdiff, Xdiff_pos))
                            ydiff = np.hstack((ydiff, ydiff_pos))
                    if 0 < r < 1:
                        r = int(np.rint(1.0 / r))
                        for i in range(r):
                            Xdiff = np.vstack((Xdiff, Xdiff_neg))
                            ydiff = np.hstack((ydiff, ydiff_neg))
            # Stack the local existing relevance vectors to the training data set.
            # The k nearest relevance vectors are used as the initial set of relevance vectors.
            X = np.vstack((local_rv, Xdiff))
            y = np.hstack((local_rv_label, ydiff))

        time1 = time.time()
        # Get feature vectors using the kernel function.
        K = get_kernel( X, X, self.gamma, self.kernel)
        # use fit method of ProbitRVC to update the relevance vectors.
        _ = super(SBKM,self).fit(K,y)

        # Update the global set of relevance vectors representing the environment.
        # Add new relevance vectors found in the current local training set
        self.relevant_  = [np.where(active==True)[0] for active in self.active_]
        if X.ndim == 1:
            self.relevant_vectors_ = [ X[relevant_] for relevant_ in self.relevant_]
            self.rv_labels = [ y[relevant_] for relevant_ in self.relevant_]
            self.rv_A = [self.lambda_[0][relevant_] for relevant_ in self.relevant_]
        else:
            self.relevant_vectors_ = [ X[relevant_,:] for relevant_ in self.relevant_ ]
            self.rv_labels = [y[relevant_] for relevant_ in self.relevant_]
            self.rv_A = [self.lambda_[0][relevant_] for relevant_ in self.relevant_]
        rv_local_list = self.relevant_vectors_[0].tolist()
        count_dict = {}
        for i in range(len(rv_local_list)):
            r = (rv_local_list[i][0], rv_local_list[i][1])
            if (r[0], r[1]) not in count_dict:
                count_dict[(r[0], r[1])] = 1
            else:
                count_dict[(r[0], r[1])] = count_dict[(r[0], r[1])] + 1
            if r not in self.relevant_vectors_dict.keys():
                self.relevant_vectors_dict[(r[0], r[1])] = (self.rv_labels[0][i], self.rv_A[0][i])
                self.rtree_index.insert(int(2*(r[0]*10000 + 2*r[1])), (r[0], r[1], r[0], r[1]))
            else:
                rv_label_A = self.relevant_vectors_dict[(r[0], r[1])]
                if self.rv_labels[0][i] != rv_label_A[0] or self.rv_A[0][i] < rv_label_A[1]:
                    #print(self.rv_labels[0][i], rv_label_A[0])
                    self.relevant_vectors_dict[(r[0], r[1])] = (self.rv_labels[0][i], self.rv_A[0][i])
        # Call Laplace approximation on the set of relevance vectors to calculate the mean and covariance of the weights
        self.global_posterior_approx()
        self.prev_X = self.cur_X
        self.prev_y = self.cur_y
        time1 = time.time()
        print("Update time:", time1 - time0)
        return self

    def global_posterior_approx(self, a_thres = -1.0):
        '''
        The GLOBAL POSTERIOR APPROXIMATION function in our paper.

        Parameters
        -----------
        a_thres (float):
            Threshold on the precision parameter xi as large value of precision means the weight is likely to be near 0
            and therefore, can be ignored.
            If it's set to negative values (-1, by default), there is no thresholding at all.

        Returns
        -------
        self (class object)
            self
        '''
        self.all_rv_X = []
        self.all_rv_y = []
        self.all_rv_A = []
        keys_list = list(self.relevant_vectors_dict.keys())
        if a_thres > 0:
            # Remove the relevance vectors with high precision xi, i.e. weight is approximately 0.
            for r in keys_list:
                if self.relevant_vectors_dict[r][1] > a_thres:
                    self.relevant_vectors_dict.pop(r)
                    self.rtree_index.delete(int(2 * (r[0] * 10000 + 2 * r[1])), (r[0], r[1], r[0], r[1]))
        for r in self.relevant_vectors_dict.keys():
            self.all_rv_X.append(np.array([r[0], r[1]]))
            self.all_rv_y.append(self.relevant_vectors_dict[r][0])
            self.all_rv_A.append(self.relevant_vectors_dict[r][1])
        all_rv_K = get_kernel(self.all_rv_X, self.all_rv_X, self.gamma, self.kernel)
        all_rv_y = np.array(self.all_rv_y)
        all_rv_A = np.array(self.all_rv_A)
        # Call Laplace approximation
        Mn, Sn, cholesky = self._posterior_dist_global(all_rv_K, all_rv_y, all_rv_A, tol_solver=0.01*self.tol_solver,
                                                       n_iter_solver=100*self.n_iter_solver)
        # in case Sn is inverse of lower triangular matrix of Cholesky decomposition
        # compute covariance using formula Sn  = np.dot(Rinverse.T , Rinverse)
        if cholesky:
            Sn = np.dot(Sn.T, Sn)
        else:
            print("Cholesky decomposition failed")
        self.Mn = Mn
        self.Sn = Sn
        return self

    def decision_function(self, X):
        '''
        Calculate the mean and variance of the score F(x) for each point x in X

        Parameters
        ----------
        X (array of size [n_samples, n_features]):
            Querying data containing n_samples points.

        Returns
        -------
        decision (array of size [n_samples_test,])
            The mean of the score F(x)
        var (array of size [n_samples_test,])
            The variance of the score F(x)
        '''
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        n_features = len(self.all_rv_X[0])
        if X.shape[1] != n_features:
            raise ValueError("X has %d features per sample; expecting %d"
                             % (X.shape[1], n_features))
        kernel = lambda rvs: get_kernel(X, rvs, self.gamma, self.kernel)
        decision = []
        K = []
        if len(self.all_rv_X) == 0:
            decision.append(np.ones(X.shape[0]) * self.intercept_)
        else:
            k = kernel(self.all_rv_X)
            decision.append(safe_sparse_dot(k, self.Mn) + self.intercept_)
            K.append(k)
        decision = np.asarray(decision).squeeze().T
        K = np.array(K[0])
        var = np.sum(np.matmul(K, self.Sn) * K, axis=1)
        return decision, var

    def get_feature_matrix(self, X):
        '''
        Calculate the feature matrix for points in X

        Parameters
        ----------
        X (array of size [n_samples, n_features]):
            Querying data containing n_samples points.

        Returns
        -------
        K (array of size [n_samples, n_rvs]):
            Feature matrix where n_rvs is the number of relevance vectors.
        '''
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        n_features = len(self.all_rv_X[0])
        if X.shape[1] != n_features:
            raise ValueError("X has %d features per sample; expecting %d"
                             % (X.shape[1], n_features))
        kernel = lambda rvs: get_kernel(X, rvs, self.gamma, self.kernel)
        K = kernel(self.all_rv_X)
        return K

    def predict_proba(self, X):
        '''
        Predicts occupancy probabilities of point x in X.

        Parameters
        ----------
        X (array of size [n_samples, n_features]):
            Querying data containing n_samples points.

        Returns
        -------
        probs (array of size [n_samples_test,]):
           Estimated occupancy probabilities

        '''
        decision, var = self.decision_function(X)
        prob = norm.cdf(decision / np.sqrt(var + 1))
        if prob.ndim == 1:
            prob = np.vstack([1 - prob, prob]).T
        prob = prob / np.reshape(np.sum(prob, axis=1), (prob.shape[0], 1))
        return prob