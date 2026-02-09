import numpy as np
import matplotlib.pyplot as plt

"""
Source: https://arxiv.org/pdf/1303.1702 Section 3D -> but we fit line rather than dividing by lag to compute D

File provides helper functions for inferring diffusion tensors using MSD
"""

def compute_covariance_matrix(trajectories: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the covariance matrix / MSD for each time lag
    
    Args:
        trajectories: np.ndarray (N,T,2)
            Particle trajectories 
    Returns:
        C_tensors: np.ndarray (N, T/10, 2, 2)
            Covariance matrices constructed for N particles and each time lag tau
        taus: np.ndarray (T/10,)
            Time lags used
    """

    nparticles, num_steps, d = trajectories.shape
    taus = np.arange(1, num_steps//10)
    C_tensors = np.zeros((nparticles, len(taus), d, d))

    for p in range(nparticles):
        pos = trajectories[p]
        for i, tao in enumerate(taus):
            # Compute differences for given time lag
            disp = pos[tao:] - pos[:-tao] # shape (num_steps-tao, d)

            # Compute covariance matrix for the particles at this time lag
            disp = disp - disp.mean(axis=0, keepdims=True)
            C = (disp.T @ disp) / disp.shape[0]
            C_tensors[p,i] = C

    return C_tensors, taus

def estimate_diffusion_tensor(C: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """
    Recover diffusion tensor from covariance matrices for time lags tau
    
    Args:
        C: np.ndarray (N, t, 2, 2)
            Covariance matrices constructed from trajectories and time lags
        taus: np.ndarray (t,)
            Time lags used
    Returns:
        D_tensors: np.ndarray (N,2,2)
            Diffusion tensors predicted
    """
    nparticles, _, d, _ = C.shape
    D_tensors = np.zeros((nparticles, d, d))

    for p in range(nparticles):
        for i in range(d):
            for j in range(d):
                y = C[p, :, i, j]  # covariance at each tau
                slope, _ = np.polyfit(taus, y, 1)  # linear fit
                D_tensors[p, i, j] = slope / 2.0

    return D_tensors

def diffusion_tensor_decomposition(D_tensors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs eigen-decomposition of diffusion tensors to recover principal
    diffusion coefficients and orientations
    
    Args:
        D_tensors: np.ndarray (N,2,2)
            Diffusion tensors
    Returns:
        eigenvalues: np.ndarray (N,2)
            Eigenvalues from tensors
        angles: np.ndarray (N,)
            Angles from tensors
    """
    nparticles = D_tensors.shape[0]

    eigenvalues = np.zeros((nparticles, 2))
    angles = np.zeros(nparticles)

    for p in range(nparticles):

        # Eigen decomposition
        vals, vecs = np.linalg.eig(D_tensors[p])

        # Sort eigenvalues (largest first)
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]

        eigenvalues[p] = vals

        # Orientation angle relative to x-axis
        vx = vecs[0, 0]
        vy = vecs[1, 0]
        angles[p] = np.arctan2(vy, vx)

    return eigenvalues, angles

def plotMSD(C: np.ndarray, taus: np.ndarray):
    """
    Plot MSD-based inference from trajectories
    
    Args:
        C: np.ndarray (N,t,2,2)
            Covariance matrices
        taus: np.ndarray (t,)
            Time lags
    """
    plt.figure(figsize=(4, 4))

    nparticles, nlags, _, _ = C.shape
    msd = np.zeros((nparticles, nlags))

    for p in range(nparticles):
        for i in range(nlags):
            msd[p,i] = np.trace(C[p,i])

    #plt.plot(taus, msd.mean(axis=0))
    plt.plot(taus, msd[0])
        
    # Set plot details
    plt.title("Mean Square Displacement (MSD) vs Time Lag")
    plt.xlabel("Time Lag")
    plt.ylabel(r"MSD ($\mu m^2$)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()