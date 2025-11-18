import numpy as np
import matplotlib.pyplot as plt

"""
Source: https://arxiv.org/pdf/1303.1702 Section 3D -> but we fit line rather than dividing by lag to compute D
"""

def compute_covariance_matrix(trajectories):
    """
    Computes the covariance matrix / MSD for each time lag
    """

    nparticles, num_steps, d = trajectories.shape
    taus = np.arange(1, num_steps//4)
    C_tensors = np.zeros((nparticles, len(taus), d, d))

    for p in range(nparticles):
        pos = trajectories[p]
        for i, tao in enumerate(taus):
            # Compute differences for given time lag
            disp = pos[tao:] - pos[:-tao] # shape (num_steps-tao, d)

            # Compute covariance matrix for the particles at this time lag
            C = (disp.T @ disp) / disp.shape[0]
            C_tensors[p,i] = C

    return C_tensors, taus

def estimate_diffusion_tensor(C, taus):
    """
    Recover diffusion tensor from covariance matrices for time lags tau
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

def diffusion_tensor_decomposition(D_tensors):
    """
    Performs eigen-decomposition of diffusion tensors to recover principal
    diffusion coefficients and orientations
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

def plotMSD(C, taus):
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
    plt.ylabel("MSD (pixels^2)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()