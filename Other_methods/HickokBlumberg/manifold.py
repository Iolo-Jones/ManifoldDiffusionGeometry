# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import gamma
import math
import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d

##############################################################################
# Sphere sampling
##############################################################################

class Sphere:
    
    def Rdist(n, x1, x2):
        # x1, x2: two points on unit n-sphere
        # output: geodesic distance between x1 and x2
        dotprod = sum([x1[i]*x2[i] for i in range(n+1)])
        Rdist = np.arccos(dotprod)
        return Rdist
        
    def Rdist_array(n, X):
        # n: sphere dimension
        # X: point cloud (rows are observations)
        # output: distance matrix
        N = X.shape[0]
        Rdist = np.zeros((N, N))
        for i in range(N):
            for j in range(i):
                x1 = X[i, :]
                x2 = X[j, :]
                Rdist[i, j] = Sphere.Rdist(n, x1, x2)
                Rdist[j, i] = Rdist[i, j]
        return Rdist
    
    def sample(N, n, noise = 0, R = 1):
         # To sample a point x, let x_i ~ N(0, 1) and then rescale to have norm R. then add isotropic Gaussian noise to x with variance noise^2
        X = []
        noise_mean = np.zeros(n+1)
        noise_cov = (noise**2)*np.identity(n+1)
        for i in range(N):
            x = np.random.normal(size = n+1)
            x /= np.linalg.norm(x)
            x *= R
            x += np.random.multivariate_normal(noise_mean, noise_cov)
            X.append(x)
        return np.array(X)
    
    def S2_ball_volume(r):
        # volume of geodesic ball of radius r in unit 2-sphere
        return 4*math.pi*(math.sin(r/2)**2)
    
    def unit_volume(n):
        # returns volume of Euclidean unit n-sphere
        m = n+1
        Sn = (2*(math.pi)**(m/2))/gamma(m/2)
        return Sn


##############################################################################
# Euclidean sampling
##############################################################################

class Euclidean:
    
    def sample(N, n, R, Rsmall = None):
        # If Rsmall = None, sample N points in an n-ball of radius R
        # Otherwise, sample points in an n-ball of radius R until you get N points within an n-ball of radius Rsmall < R
        X = []
        if Rsmall is None:
            for i in range(N):
                x = np.random.normal(size = n)
                u = (R**n)*np.random.random()
                r = u**(1/n)
                x *= r/np.linalg.norm(x)
                X.append(x)
        else:
            Nsmall = 0
            while Nsmall < N:
                x = np.random.normal(size = n)
                u = (R**n)*np.random.random()
                r = u**(1/n)
                x *= r/np.linalg.norm(x) # now the norm of x is r
                X.append(x)
                if r < Rsmall: Nsmall += 1
                
        return np.array(X)

    def density(n, R):
        # density in a ball of radius R in R^n
        vn = (math.pi**(n/2))/gamma(n/2 + 1) # volume of Euclidean unit n-ball
        vol = vn*R**(n)
        return 1/vol

    def distance_array(X):
        N = X.shape[0]
        D = np.zeros((N, N))
        for i in range(N):
            x = X[i, :]
            for j in range(i):
                y = X[j, :]
                D[i, j] = np.linalg.norm(x - y)
                D[j, i] = D[i, j]
        return D
    
##############################################################################
# Torus sampling
##############################################################################

class Torus:
    
    def exact_curvatures(thetas, r, R):
        curvatures = [Torus.S_exact(theta, r, R) for theta in thetas]
        return curvatures

    def sample(N, r, R):
        psis = [np.random.random()*2*math.pi for i in range(N)]
        j = 0
        thetas = []
        while j < N:
            theta = np.random.random()*2*math.pi
            #eta = np.random.random()*2*(r/R) + 1 - (r/R)
            #if eta < 1 + (r/R)*math.cos(theta):
            eta = np.random.random()/math.pi
            if eta < (1 + (r/R)*math.cos(theta))/(2*math.pi):
                thetas.append(theta)
                j += 1
    
        def embed_torus(theta, psi):
            x = (R + r*math.cos(theta))*math.cos(psi)
            y = (R + r*math.cos(theta))*math.sin(psi)
            z = r*math.sin(theta)
            return [x, y, z]
    
        X = np.array([embed_torus(thetas[i], psis[i]) for i in range(N)])
        return X, np.array(thetas)
    
    def S_exact(theta, r, R):
        # Analytic scalar curvature
        S = (2*math.cos(theta))/(r*(R + r*math.cos(theta)))
        return S
    
    def theta_index(theta, thetas):
        # Returns index in thetas of the angle closest to theta
        err = [abs(theta_ - theta) for theta_ in thetas]
        return np.argmin(err)
    
##############################################################################
# Poincare disk sampling
##############################################################################

class PoincareDisk:
    
    def sample(N, K = -1, Rh = 1):
        # N: number of points, K: Gaussian curvature
        # Rh: hyperbolic radius of the disk
        assert K < 0, "K must be negative"
        R = 1/math.sqrt(-K)
        thetas = 2*math.pi*np.random.random(N)
        us = np.random.random(N)
        C1 = 2/math.sqrt(-K)
        C2 = np.sinh(Rh*math.sqrt(-K)/2)
        rs = [C1*np.arcsinh(C2*math.sqrt(u)) for u in us]
        ts = [R*np.tanh(r/(2*R)) for r in rs]
        X = np.array([[ts[i]*math.cos(thetas[i]), ts[i]*math.sin(thetas[i])] for i in range(N)])
        return X
    
    def sample_polar(N, K = -1):
        # N: number of points
        # Gaussian curvature is K = -1
        thetas = 2*math.pi*np.random.random(N)
        us = np.random.random(N)
        C1 = 2/math.sqrt(-K)
        C2 = np.sinh(math.sqrt(-K)/2)
        rs = [C1*np.arcsinh(C2*math.sqrt(u)) for u in us]
        X = np.array([[rs[i], thetas[i]] for i in range(N)])
        return X
    
    def cartesian_to_polar(X, K = -1):
        R = 1/math.sqrt(-K)
        N = X.shape[0]
        X = []
        for i in range(N):
            x = X[i, 0]
            y = X[i, 1]
            r = 2*R*np.arctanh(math.sqrt(x**2 + y**2)/R)
            theta = np.arccos(x/(R*np.tanh(r/(2*R))))
            X.append([r, theta])
        return X
    
    def norm(x, K = -1):
        assert K < 0, "K must be negative"
        return PoincareDisk.Rdist(np.array([0, 0]), x, K = K)
        
    def polar_to_cartesian(X, K = -1):
        R = 1/math.sqrt(-K)
        N = X.shape[0]
        rs = X[:, 0]
        thetas = X[:, 1]
        ts = [R*np.tanh(r/(2*R)) for r in rs]
        X = np.array([[ts[i]*math.cos(thetas[i]), ts[i]*math.sin(thetas[i])] for i in range(N)])
        return X
    
    def Rdist(u, v, K = -1):
        assert K < 0, "K must be negative"
        R = 1/math.sqrt(-K)
        z = u/R
        w = v/R
        #wconj = np.array([w[0], -w[1]]) # conjugate of w, thought of as a complex number
        z_wconj = np.array([z[0]*w[0] + z[1]*w[1], w[0]*z[1] - z[0]*w[1]]) # product of z and w_conj, thought of as complex numbers
        dist = 2*R*np.arctanh(np.linalg.norm(z - w)/np.linalg.norm(np.array([1, 0]) - z_wconj))
        return dist
    
    def Rdist_polar(u, v):
        # u, v: tuples. polar coordinates (r, theta)
        # Gaussian curvature is K = -1
        r1 = u[0]
        theta1 = u[1]
        r2 = v[0]
        theta2 = v[1]
        return np.arccosh(np.cosh(r1)*np.cosh(r2) - np.sinh(r1)*np.sinh(r2)*np.cos(theta2 - theta1))
    
    def Rdist_array(X, K = -1, polar = False):
        # K is the Gaussian curvature of the hyperbolic plane that X is sampled from
        if polar: assert K == -1
        N = X.shape[0]
        Rdist = np.zeros((N, N))
        for i in range(N):
            print(i)
            for j in range(i):
                x1 = X[i, :]
                x2 = X[j, :]
                if polar:
                    Rdist[i, j] = PoincareDisk.Rdist_polar(x1, x2)
                else:
                    Rdist[i, j] = PoincareDisk.Rdist(x1, x2, K)
                Rdist[j, i] = Rdist[i, j]
        return Rdist
    
    def area(Rh, K = -1):
        # Rh: hyperbolic radius, K: curvature
        assert K < 0
        return (-4*math.pi/K)*(np.sinh(Rh*math.sqrt(-K)/2)**2)


##############################################################################
# Hyperboloid sampling
##############################################################################

# TO DO- organize/clean up

class Hyperboloid:
    
    def det_g(a, c, u):
        return (a**4)*(u**2) + a**2*(u**2 + 1)*c**2
    
    def sample(N, a = 2, c = 1, B = 2, within_halfB = True):
        # if within_halfB = False, then sample N points from the hyperboloid with u in [-B, B]
        # if within_halfB = True, then sample points uniformly from u in [-B, B] until there are at least N points with u in [-.5B, .5B]
        # print(within_halfB)
        # print("a = ", a)
        # print("c = ", c)
        # print("B = ", B)
        sqrt_max_det_g = math.sqrt(Hyperboloid.det_g(a, c, B))
        us = []
        thetas = []
        i = 0
        while i < N:
            theta = 2*math.pi*np.random.random()
            u = 2*B*np.random.random() - B
            eta = sqrt_max_det_g*np.random.random()
            sqrt_det_g = math.sqrt(Hyperboloid.det_g(a, c, u))
            
            if eta < sqrt_det_g:
                us.append(u)
                thetas.append(theta)
                if (within_halfB and -.5*B <= u <= .5*B) or (not within_halfB):
                    i += 1
       
        xs = [a*math.cos(thetas[i])*math.sqrt(u**2 + 1) for i, u in enumerate(us)]
        ys = [a*math.sin(thetas[i])*math.sqrt(u**2 + 1) for i, u in enumerate(us)]
        zs = [c*u for i, u in enumerate(us)]
        X = np.array([[x, ys[i], zs[i]] for i, x in enumerate(xs)])
        return X

    def area(a, c, B):
        alpha = math.sqrt(c**2 + a**2)/(c**2)
        cBalpha = c*B*alpha
        return 2*math.pi*a*(math.sqrt(cBalpha**2 + 1)*cBalpha + np.arcsinh(cBalpha))/alpha
    
    def S(z):
        # actual scalar curvature at z when a = b = 2 and c = 1
        return -2/((5*z**2 + 1)**2)

##############################################################################
# Computing neighbor distance matrices
##############################################################################
def nbr_distance_matrices(Rdist):
    '''
    Rdist: (N x N) numpy array. (i, j)th entry is estimated (or exact) geodesic distance between ith and jth points
       
    Returns:
    T: (N x N) numpy array. (i, j)th entry = Rdist from X_i to its jth nearest neighbor (where 0th nearest neighbor is X_i)
    nbr_matrix: (N x N) numpy array. (i, j)th entry = index of jth nearest neighbor to X_i (entry (i, 0) is i for all i).
    '''
    N = Rdist.shape[0]
    nbr_matrix = np.empty((N, N))
    T = np.empty((N, N))
    for i in range(N):
        nbr_matrix[i, :] = np.argsort(Rdist[i, :])
        T[i, :] = [Rdist[i, int(j)] for j in nbr_matrix[i, :]]
        if i%1000 == 0: print(i)
    return T, nbr_matrix


##############################################################################
# 3d plotting
##############################################################################

# set_axes_equal and _set_axes_radius are functions from @Mateen Ulhaq and @karlo to make the aspect ratio equal for 3d plots

def set_axes_equal(ax):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])
    
def plot_3d(X, vals = None, s = None):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    if vals is None:
        if s is None:
            p = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c = X[:, 2]) # color according to z-coordinate
        else:
            p = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c = X[:, 2], s = s) # color according to z-coordinate
    else:
        if s is None:
            p = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c = vals)
        else:
            p = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c = vals, s = s)
    ax.set_box_aspect([1,1,1])
    set_axes_equal(ax)
    fig.colorbar(p)
    plt.show()