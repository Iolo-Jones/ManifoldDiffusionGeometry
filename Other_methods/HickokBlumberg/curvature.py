import math
from scipy.special import gamma
import numpy as np
from sklearn.manifold import Isomap
import multiprocessing as mp

class KDE:
    def __init__(self, n, X = None, D = None, kernel = None):
        '''
        Parameters
        ----------
        n: dimension of manifold
        X: (optional, but must input either X or D) N x d matrix containing N observations in R^d.
        D: (optional, but must input either X or D) Distance matrix where D[i, j] = geodesic (exact or estimated) distance or Euclidean distance between ith and jth points.
        kernel: optional. kernel function (e.g. KDE.gauss or KDE.biweight. Default is biweight)
        '''
        assert (X is not None) or (D is not None)
        self.n = n
        self.X = X
        self.D = D
        if kernel is None:
            self.kernel = KDE.biweight
        else:
            self.kernel = kernel
        if X is not None:
            self.N = X.shape[0]
        else:
            self.N = D.shape[0]
        self.h = KDE.bandwidth(self.N, n)   # Scotts's rule
            
    def __call__(self, i):
        '''
        Returns
        -------
        density: float. Kernel density estimate of density at ith point.
        '''
        if self.D is not None:
            density = sum([self.kernel(self.D[i, j]/self.h, self.n)/(self.N*math.pow(self.h, self.n)) for j in range(self.N)])
        else:
            density = sum([self.kernel(np.linalg.norm(y - self.X[i, :])/self.h, self.n)/(self.N*math.pow(self.h, self.n)) for y in self.X])
        return density
    
    def gauss(x, n):
        '''
        Returns Gaussian kernel evaluated at x, which is the distance between a pair of points
        '''
        return (1/math.pow(math.sqrt(2*math.pi), n))*math.exp(-x*x/2)
    
    def density(self):
        '''
        Returns
        -------
        density: list of length N, the number of points. density[i] is density estimate at ith point.
        '''
        with mp.Pool(mp.cpu_count()) as p:
            density = p.map(self, np.arange(self.N))
        return density
    
    def biweight(x, n):
        '''
        Returns biweight kernel evaluated at x, which is the distance between a pair of points
        '''
        if -1 < x < 1:
            s = 2*((math.pi)**(n/2))/gamma(n/2)
            normalization = s*(1/n - 2/(n+2) + 1/(n+4))
            return ((1-x**2)**2)/normalization
        else:
            return 0
    
    def epanechnikov(x, n):
        '''
        Returns Epanechnikov kernel evaluated at x, which is the distance between a pair of points
        '''
        if -1 < x < 1:
            s = 2*((math.pi)**(n/2))/gamma(n/2)
            normalization = s*(1/n - 1/(n+2))
            return (1-x**2)/normalization
        else:
            return 0
        
    def triweight(x, n):
        '''
        Returns triweight kernel evaluated at x, which is the distance between a pair of points
        '''
        if -1 < x < 1:
            s = 2*((math.pi)**(n/2))/gamma(n/2)
            normalization = s*(-1/(n+6) + 3/(n+4) - 3/(n+2) + 1/n)
            return ((1-x**2)**3)/normalization
        else:
            return 0
    
    def bandwidth(N, n):
        '''
        N: number of points in sample
        n: dimension of manifold
        
        Returns: bandwidth parameter, set according to Scott's rule
        '''
        return N**(-1/(n+4))
    
class scalar_curvature_est:
    def __init__(self, n, X = None, n_nbrs = 20, kernel = None, density = None, Rdist = None, T = None, nbr_matrix = None, verbose = True):
        '''
        Parameters
        ----------
        n: Integer. dimension of the manifold
        X: (optional, but must input either X or Rdist) N x d matrix containing N observations in R^d.
        n_nbrs: (optional) Integer, default 20. Used for geodesic-distance estimation if Rdist is None. The parameter n_nbrs is the number of neighbors to use for Isomap geodesic-distance estimation. (Isomap default is 5 but this is generally way too low.) The default works well for n = 2, but should be increased for higher dimensions.
        kernel: (optional) kernel function, default KDE.biweight. Used for kernel density estimation if density is None.
        density: (optional) list or 1D numpy array where density[i] is an estimate of the density at the ith point.
        Rdist: (optional, but must input either X or Rdist) N x N numpy array. Stores geodesic/Riemannian distances (exact or estimated distances). Takes the role of the distance matrix described in paper.
        T: (optional) N x N numpy array, where T[i, j] is distance (as given by Rdist) from ith point to its jth nearest neighbor (0th nearest nbr is itself)
        nbr_matrix: (optional) N x N numpy array, where nbr_matrix[i, j] is index of jth nearest neighbor to ith point.
        verbose: (optional) Boolean, default True. Print progress statements if True.
        '''
        assert X is not None or Rdist is not None
        self.X = X
        self.n = n
        self.n_nbrs = n_nbrs
        self.density = density
        self.Vn = (math.pi**(self.n/2))/gamma(self.n/2 + 1) # volume of Euclidean unit n-ball
        
        if X is not None:
            self.N = X.shape[0] # number of observations
            self.d = X.shape[1] # ambient dimension
        else:
            self.N = Rdist.shape[0]
            
        if Rdist is None:
            self.Rdist = scalar_curvature_est.compute_Rdist(X, n_nbrs)
            if verbose: print("computed Rdist")
        else:
            self.Rdist = Rdist
            
        self.kernel = kernel
        if density is None:
            self.density = scalar_curvature_est.compute_density(n, D = self.Rdist, kernel = self.kernel)
            if verbose: print("computed density")
        else:
            self.density = density
         
        if T is None or nbr_matrix is None:
            self.compute_nbr_matrices()
            if verbose: print("computed nearest neighbor matrices")
        else:
            self.T = T
            self.nbr_matrix = nbr_matrix
        
    def ball_ratios(self, i, rmax = None, rmin = None, max_k = None, min_k = None, avg_k = 0):
        '''
        Parameters
        ----------
        i: integer in {0, ..., self.N-1}. index of a sample point.
        rmax (optional, must input either rmax or max_k. Can't input both): positive float. This corresponds to rmax in the paper and controls the scale at which we're computing scalar curvature
        rmin (optional, can't input both rmin and min_k): nonnegative float, default 0. This corresponds to rmin in the paper.
        max_k (optional, must input either rmax or max_k. Can't input both): Positive integer. Alternative to rmax.
        min_k (optional, can't input both rmin and min_k): Positive integer. Alternative to rmin.
        avg_k (optional): Integer, default 0. Not recommended for use (strong warning: code hasn't been optimized for use).
    
        Returns
        -------
        rs: sequence of radii. Plays the role of radius sequence in paper, with notable difference that it starts at r_1 instead of r_0 = rmin.
            If rmin is given, then rs[0] is the geodesic distance (as given by Rdist) from ith point to (min_k)th nearest neighbor, where min_k is the first index s.t. Rdist[i, min_k] > rmin. If min_k is given, then rs[0] = Rdist[i, min_k].
            rs[j] = T[i, min_k + j] = geodesic distance (as given by Rdist) from ith point to its (min_k + j)st nearest neighbor.
            If rmax is given, then rs[-1] = Rdist[i, max_k] where max_k is the last index s.t. Rdist[i, max_k] <= rmax. If max_k is given, then rs[-1] = Rdist[i, max_k].
            
        ball_ratios: sequence of estimated ratios of geodesic ball volumes to euclidean ball volumes, where ball_ratios[j] = estimated ratio at radius rs[j] (for ball centered at ith point), which is computed by \hat{y} formula in paper.
        '''
        assert 0 <= i < self.N
        assert (rmax is not None and max_k is None) or (rmax is None and max_k is not None) # exactly one of these should be given
        assert rmin is None or min_k is None # only input at most one of these
        if min_k is not None:
            assert min_k >= 1
        
        rs, ball_vols = self.ball_volumes(i, rmax, max_k)
        ball_ratios = np.array([ball_vols[j]/(self.Vn*(r**self.n)) for j, r in enumerate(rs)])
        
        if max_k is None:
            max_k = len(rs)
            
        if min_k is not None:
            ball_ratios = ball_ratios[(min_k - 1):]
            rs = rs[(min_k - 1):] # recall that initially, rs[j] = Rdist[i, j + 1], hence the (min_k - 1) instead of min_k
            
        if rmin is not None:
            ball_ratios = ball_ratios[rs > rmin]
            rs = rs[rs > rmin]
            min_k = max_k + 1 - len(rs)
        
        if avg_k > 0:
            ball_ratio_sums = ball_ratios
            assert self.nbr_matrix[int(i), 0] == i
            k_nbrs = self.nbr_matrix[i, 1:(avg_k + 1)]
            assert len(k_nbrs) == avg_k
            for nbr_idx in k_nbrs:
                nbr_idx = int(nbr_idx)
                _, nbr_ball_ratios = self.ball_ratios(nbr_idx, min_k = min_k, max_k = max_k)
                for j, ratio in enumerate(nbr_ball_ratios):
                    ball_ratio_sums[j] += ratio
            ball_ratios = [ball_sum/(avg_k + 1) for ball_sum in ball_ratio_sums] # average the ball ratios at each r
        
        return rs, ball_ratios
    
    def ball_volumes(self, i, rmax = None, max_k = None):
        '''
        Parameters
        ----------
        i: integer in {0, ..., self.N-1}. index of a sample point.
        rmax: (optional, must input rmax or max_k but not both) positive float.
              This corresponds to rmax in the paper and controls the scale at which we're computing scalar curvature
        max_k: (optional, must input rmax or max_k but not both) positive integer.
               Alternative to rmax.
        
        Returns
        -------
        rs: sequence of radii where rs[j] = self.T[i, j+1] = geodesic distance (as given by Rdist) from ith point to its (j+1)st nearest neighbor. If rmax is given, then rs[-1] = Rdist[i, max_k] where max_k is the last index s.t. Rdist[i, max_k] <= rmax. If max_k is given, then rs[-1] = Rdist[i, max_k].
        
        ball_volumes: sequence of estimated geodesic ball volumes, where ball_volumes[j] = estimated volume of the ball centered at ith point with radius rs[j]. Calculated using \hat{vol} formula in paper.
        '''
        assert (rmax is not None and max_k is None) or (rmax is None and max_k is not None) # exactly one of these parameters should be given
        assert 0 <= i < self.N
        
        rs, nbr_indices = self.nbr_distances(i, rmax, max_k)
        rs = rs[1:]
        nbr_indices = nbr_indices[1:]
        
        inv_density_sums = []
        curr_sum = 0
        for j in nbr_indices:
            curr_sum += 1/self.density[int(j)]
            inv_density_sums.append(curr_sum)
        ball_volumes = [inv_density_sums[j]/(self.N -1) for j in range(len(rs))]
            
        return rs, ball_volumes
  
    def compute_density(n, X = None, D = None, kernel = None):
        '''
        Parameters
        ----------
        n: dimension of manifold
        X: (optional, but must input either X or D) N x d matrix containing N observations in R^d.
        D: (optional, but must input either X or D) N x N matrix where D_{ij} = geodesic or Euclidean distance between ith and jth points.
        kernel: optional. kernel function (e.g. KDE.gauss or KDE.biweight. Default is KDE.biweight
        
        Returns
        -------
        density: list of length N where density[i] = estimated density at point i. Computed using kernel density estimation.
        '''
        kde = KDE(n, X, D, kernel)
        density = kde.density()
        return density
    
    def compute_nbr_matrices(self):
        '''
        Computes self.T and self.nbr_matrix.
        
        self.T: N x N numpy array, where T[i, j] is distance (as given by Rdist) from ith point to its jth nearest neighbor (0th nearest nbr is itself). N is the number of points in the sample.
        self.nbr_matrix: N x N numpy array, where nbr_matrix[i, j] is index of jth nearest neighbor of ith point.
        '''
        self.T = self.get_Rdist()
        self.nbr_matrix = np.zeros((self.N, self.N))
        for i in range(self.N):
            distances = self.T[i, :]
            nbr_indices = np.argsort(distances)
            self.T[i, :] = distances[nbr_indices]
            self.nbr_matrix[i, :] = nbr_indices
        
    def compute_Rdist(X, n_nbrs = 20):
        '''
        Parameters
        ----------
        X: N x d matrix containing N observations (sample points) in R^d.
        n_nbrs: (optional) integer, default 20. Parameter to pass to isomap. (Number of neighbors in nearest neighbor graph)
        
        Returns
        -------
        Rdist: N x N numpy array, where Rdist[i, j] is estimated geodesic/Riemannian distance between ith and jth points. See Isomap paper.
        '''
        iso = Isomap(n_neighbors = n_nbrs, n_jobs = -1)
        iso.fit(X)
        Rdist = iso.dist_matrix_
        return Rdist
    
    def estimate(self, rmax, indices = None, rmin = None, version = 2, with_error = False, avg_k = 0):
        # TO DO- parallelize and allow different rmaxes for each point
        '''
        Parameters
        ----------
        rmax: positive float. This corresponds to rmax in the paper and controls the scale at which we're computing scalar curvature
        indices (optional): list or 1D numpy array, subset of {0, ..., N-1}. By default, gets set to [0, ..., N-1].
        rmin (optional): nonnegative float, default 0. This corresponds to rmin in the paper.
        version (optional): integer in {1, 2}, default 2. Corresponds to two different ways to fit the quadratic curve; version 2 is what's in the paper.
        with_error : (optional), Boolean, default False.
        avg_k (optional): Integer, default 0. Not recommended for use (strong warning: code hasn't been optimized for use).
        
        Returns
        -------
        Ss: list of length N if indices is None, or length = len(indices) otherwise.
            If indices is None, Ss[i] is the scalar curvature estimate at the ith point. Otherwise, Ss[i] is the scalar curvature estimate at the (indices[i])th point. If avg_k = 0 and version = 2, this corresponds exactly to \hat{S} in paper.
            If avg_k > 0, then the ball-volume ratio estimate at radius r_j (for each j) is averaged with the jth ball-volume estimate for the nearest avg_k neighbors. Then quadratic curve is fitted as usual.
            
        errs: list of floats, only returned if with_error is True (default False)
            errs[i] is the fit error of the ith curve, which is fitted to the (rs, ball_ratios) data points for indices[i]. For each i, the error is computed as the average difference between the curve 1 + Cr^2 and ball_ratio[r], averaged over the data points)
        '''
        if indices is None:
            indices = np.arange(self.N)
        if with_error:
            Cs = []
            errs = []
            for i in indices:
                C, err = self.fit_quad_coeff(i, rmax, rmin, version, with_error, avg_k)
                Cs.append(C)
                errs.append(err)
        else:
            Cs = [self.fit_quad_coeff(i, rmax, rmin, version, with_error, avg_k) for i in indices]
        Ss = [-6*(self.n + 2)*C for C in Cs]
        if with_error:
            return Ss, errs
        else:
            return Ss
    
    def fit_quad_coeff(self, i, rmax, rmin = None, version = 2, with_error = False, avg_k = 0):
        '''
        Parameters
        ----------
        i: integer in {0, ..., self.N-1}. index of a sample point.
        rmax: positive float. This corresponds to rmax in the paper and controls the scale at which we're computing scalar curvature
        rmin (optional): nonnegative float, default 0. Corresponds to rmin in the paper.
        version (optional): integer in {1, 2}, default 2. 
            Corresponds to two different ways to fit the quadratic curve; version 2 is what's in the paper.
        with_error : (optional), Boolean, default False.
        avg_k (optional): Integer, default 0. Not recommended for use (strong warning: code hasn't been optimized for use).

        Returns
        -------
        C: float
            The quadratic coefficient of a polynomial of form 1 + C*r^2 that we fit to the data (r_i, y_i = estimated ball ratio at radius r_i), where rs, ys are what's returned by ball_ratios function. If version = 2 and avg_k = 0, this corresponts exactly to \hat{C} in paper. If avg_k > 0, then the ball-volume ratio estimate at radius r_j (for each j) is averaged with the jth ball-volume ratio estimate for the nearest avg_k neighbors. Then quadratic curve is fitted as usual.
            
        error: float, only returned if with_error is True (default False)
               fit error between the curve 1 + Cr^2 and the (rs, ball_ratios) data points. The error is computed as the average difference between 1 + Cr^2 and ball_ratio[r], averaged over the data points.
        '''
        assert 0 <= i < self.N
        
        rs, ball_ratios = self.ball_ratios(i, rmax, rmin = rmin, avg_k = avg_k)
            
        if rmin is None:
            rmin = 0
        if version == 1:
            numerator = sum(np.array([(ball_ratios[j] - 1)*r**2 for j, r in enumerate(rs)]))
            denom = sum(np.array([r**4 for r in rs]))
            C = numerator/denom
        else:
            rs = np.append(rs, rmin) # so that r[-1] = rmin. need this for the rs[i] - rs[i-1] term below.
            numerator = sum(np.array([(r**2)*(ball_ratios[j] - 1)*(r - rs[j-1]) for j, r in enumerate(rs[:-1])]))
            denom = (rs[-2]**5 - rmin**5)/5
            C = numerator/denom
        if with_error:
            err = np.mean([abs(1 + C*(r**2) - ball_ratios[j]) for j, r in enumerate(rs[:-1])])
            return C, err
        else:
            return C
    
    def get_density(self):
        '''
        Returns
        -------
        density: list of length N where density[i] = estimated density at point i.
        '''
        if self.density is None:
            self.density = scalar_curvature_est.compute_density(self.n, self.X, self.kernel)
        return self.density
    
    def get_Rdist(self):
        '''
        Returns
        -------
        Rdist: NxN numpy array, where Rdist[i, j] is estimated geodesic/Riemannian distance between ith and jth points.
        '''
        if self.Rdist is None:
            self.Rdist = scalar_curvature_est.compute_Rdist(self.X, self.n_nbrs)
        return self.Rdist
    
    def nbr_distances(self, i, rmax = None, max_k = None):
        '''
        Parameters
        ----------
        i: integer in {0, ..., self.N-1}. index of a sample point.
        rmax: (optional, must input either rmax or max_k, but not both) positive float. 
              This corresponds to rmax in the paper and controls the scale at which we're computing scalar curvature.
        max_k: (optional, must input rmax or max_k, but not both) Positive integer.
               Alternative to rmax. Controls the scale at which we're computing scalar curvature.
        
        Returns
        -------
        distances: sorted (ascending order) list of geodesic distances (as given by Rdist) from the ith point to its set of neighbors that are within rmax (if rmax is not None), or to the 0th, 1st, ..., max_k nearest neighbors if max_k is not None.
        nbr_indices: sorted (ascending order by distance to ith point) list of neighbors' indices within rmax (if rmax given) or the indices of the 0th, 1st, ..., max_k nearest neighbors if max_k is given.
        '''
        assert 0 <= i < self.N
        assert rmax is None or max_k is None
        
        nbr_indices = self.nbr_matrix[i, :]
        distances = self.T[i, :]
        
        if rmax is not None:
            close_enough = (distances <= rmax)
            distances = distances[close_enough]
            nbr_indices = nbr_indices[close_enough]
        else:
            distances = distances[:(max_k + 1)]
            nbr_indices = nbr_indices[: (max_k + 1)]
        return distances, nbr_indices