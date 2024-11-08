import numpy as np
from opt_einsum import contract
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from ManifoldDiffusionGeometry import *
from Visualise import *
from Other_methods.HickokBlumberg.curvature import scalar_curvature_est as HickokBlumberg
import matlab.engine

# To make the Sritharan, Wang, and Hormoz MATLAB code work you'll need to install the toolboxes:
# 1. Statistics and Machine Learning
# 2. Image Processing
# 3. Symbolic Math
# 4. Parallel Computing

eng = matlab.engine.start_matlab()
eng.cd(r'Other_methods/SritharanWangHormoz', nargout=0)
SritharanWangHormoz = eng.manifold_curvature

def LocalPCA(data, k=32):
    nbrs = NearestNeighbors(n_neighbors = k, algorithm='auto').fit(data)
    _, nbr_indices = nbrs.kneighbors(data)
    return np.array([PCA().fit(x).components_.T for x in data[nbr_indices]])

def true_normals_torus(data, R):
    # Compute the true normals for data from a torus with large radius R.
    normals = []
    for p in data:
        theta = np.arctan2(p[1], p[0])
        centre_point = R * np.array([np.cos(theta), np.sin(theta), 0])
        diff = p - centre_point
        diff /= np.linalg.norm(diff)
        normals.append(diff)
        # proj = centre_point + r * diff
    return np.array(normals)

def phi_torus(data, R):

    phis = []
    for p in data:
        theta1 = np.arctan2(p[1], p[0])
        centre_point = R * np.array([np.cos(theta1), np.sin(theta1), 0])
        diff = p - centre_point
        diff /= np.linalg.norm(diff)

        sinphi = diff[2]
        cosphi = diff[0] / np.cos(theta1)

        if sinphi > 0 and cosphi > 0:
            phi1 = np.arcsin(sinphi)
        elif cosphi < 0:
            phi1 = np.pi - np.arcsin(sinphi)
        elif sinphi < 0 and cosphi > 0:
            phi1 = 2*np.pi + np.arcsin(sinphi)

        phis.append(phi1)
        
    return np.array(phis)

def angles_between_vectors(A,B):
    angles = np.array([np.arccos(np.clip(np.absolute(np.dot(a, b)), -1., 1.)) for a,b in zip(A, B)])
    return np.degrees(angles)

def test_tangent_spaces_torus(data, R, k1, k2, n0=80):

    diffusionbundle = Tangents(data, n0 = n0)
    LPCA1 = LocalPCA(data, k = k1)
    LPCA2 = LocalPCA(data, k = k2)

    true_normals = true_normals_torus(data, R)
    diffusionnormals = diffusionbundle[:,:,2]
    localPCAnormals1 = LPCA1[:,:,2]
    localPCAnormals2 = LPCA2[:,:,2]
    angles_diffusion = angles_between_vectors(true_normals, diffusionnormals)
    angles_LPCA1 = angles_between_vectors(true_normals, localPCAnormals1)
    angles_LPCA2 = angles_between_vectors(true_normals, localPCAnormals2)

    return np.mean(angles_diffusion), np.mean(angles_LPCA1),  np.mean(angles_LPCA2)

def test_gaussian_torus(data, r, R, se1, se2, n0=40):

    ### Diffusion geometry scalar curvature
    scalar_diffusion = Scalar_Curvature(data, d=2, n0=n0)

    # ### Hickok and Blumberg scalar curvature
    # scalar_HB = HickokBlumberg(2, data, verbose=False).estimate(rmax = np.pi/2)

    ### Sritharan, Wang, and Hormoz scalar curvature
    scalar_SWH_1 = SritharanWangHormoz('', data, 2, se1)
    scalar_SWH_1 = np.array(scalar_SWH_1).flatten()
    scalar_SWH_2 = SritharanWangHormoz('', data, 2, se2)
    scalar_SWH_2 = np.array(scalar_SWH_2).flatten()

    phi = phi_torus(data, R)
    true_scalar = 2*np.cos(phi)/(r*(R + r*np.cos(phi)))

    # phi_sorted = np.sort(phi)
    # true_scalar_sorted = 2*np.cos(phi_sorted)/(r*(R + r*np.cos(phi_sorted)))
    
    error_diffusion = [np.mean(abs(scalar_diffusion - true_scalar)),
                       np.mean((scalar_diffusion - true_scalar)**2)]
    error_SWH_1 = [np.mean(abs(scalar_SWH_1 - true_scalar)),
                       np.mean((scalar_SWH_1 - true_scalar)**2)]
    error_SWH_2 = [np.mean(abs(scalar_SWH_2 - true_scalar)),
                       np.mean((scalar_SWH_2 - true_scalar)**2)]

    return error_diffusion, error_SWH_1, error_SWH_2

def Torus(r, R, n, sigma=0):
    theta, phi = np.random.uniform(0, 2*np.pi, n), np.random.uniform(0, 2*np.pi, n)
    X = (R + r * np.cos(phi)) * np.cos(theta)
    Y = (R + r * np.cos(phi)) * np.sin(theta)  
    Z = r * np.sin(phi)
    data = np.stack((X,Y,Z), axis = 1)
    data += sigma * np.random.randn(n, 3)
    return data

def Sphere(R, n, sigma=0):
    data = np.random.randn(n,3)
    data *= R/np.linalg.norm(data, axis = 1).reshape(-1,1)
    data += sigma * np.random.randn(n, 3)
    return data


