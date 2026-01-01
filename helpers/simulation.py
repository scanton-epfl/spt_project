from os import replace
import numpy as np
from skimage.measure import block_reduce
import matplotlib as plt
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML, display

def generate_diffusion_props(N, D_min, D_max, angle_max, is_isotropic=False, is_binding=False) -> np.ndarray:
    # Generate random diffusion parameters
    if is_binding:
        p1 = np.random.uniform(D_min, D_max, size=N)
    else:
        #p1 = np.random.randint(D_min, D_max+1, size=N)
        step = 0.05
        choices = np.arange(D_min, D_max + step, step)
        p1 = np.random.choice(choices, size=N, replace=True)
    
    if not is_isotropic:
        alpha = np.random.uniform(0.1, 1, size=N)
        p2 = alpha*p1
        theta = np.random.uniform(0, angle_max, size=N)
    else:
        p2 = p1
        theta = np.zeros(N)

    return np.stack((p1, p2, theta), axis=-1)

def compute_displacement_statistics(displacements: np.ndarray) -> tuple:
    """
    Compute training set displacement statistics for normalization

    Args:
        displacements: (N, nFrames, 2)
    """
    dx_mean, dy_mean = displacements.mean(axis=(0,1))
    dx_std, dy_std = displacements.std(axis=(0,1))

    return dx_mean, dy_mean, dx_std, dy_std

def normalize_displacements(displacements: np.ndarray, disp_stats: tuple) -> np.ndarray:
    """
    Normalize displacements by training set statistics
    """
    # Unpack stats
    dx_mean, dy_mean, dx_std, dy_std = disp_stats
    
    # Normalize
    dx_norm = (displacements[..., 0] - dx_mean) / dx_std
    dy_norm = (displacements[..., 1] - dy_mean) / dy_std

    return np.stack((dx_norm, dy_norm), axis=-1)

def create_multi_state_displacements(p1: np.ndarray, p2: np.ndarray, theta: np.ndarray, N: int, T: int, dt: float):
    """
    Create displacements for all time steps

    Args:
        p1: np.ndarray (N,T)
            Diffusion coefficient in first prinicipal component
        p2: np.ndarray (N,T)
            Diffusion coefficient in second prinicipal component
        theta: np.ndarray (N,T)
            Angle w/ respect to x-axis, corresponding to the orientation of the first prinicipal component
        N: int
            Number of trajectories to generate
        T: int
            Number of timesteps to simulate for each trajectory
        dt: int/float
            Timestep for simulation (now unitless and set to 1)
    Returns:
        disp: np.ndarray(N,T,2)
            Displacements for each step in the simulations  
    """
    # Create array with diffusion principal values for each trajectory
    D = np.stack((p1, p2), axis=-1) # shape: (N,T,2)

    # Generate std deviations across principal axes
    sig = np.sqrt(2.0 * dt * D) # shape: (N,T,2)

    # Create rotation matrix
    cos_t = np.cos(theta) # shape: (N,T)
    sin_t = np.sin(theta)
    R = np.stack([
            np.stack([cos_t, -sin_t], axis=-1),
            np.stack([sin_t, cos_t], axis=-1)
        ], axis=-2) # shape: (N,T,2,2)

    # Create step sizes in each direction generated from normal distribution
    steps = np.random.normal(0.0, 1.0, size=(N,T,2))

    # Create displacements
    tmp = steps * sig # shape: (N,T,2)
    disp = np.einsum('ntij,ntj->nti', R, tmp)

    return disp

def create_multi_state_trajectories(p1: np.ndarray, p2: np.ndarray, theta: np.ndarray, N: int, T: int, dt: float):
    """
    Create array of multi-state trajectories given diffusion parameters

    Args:
        p1: np.ndarray (N,T)
            Diffusion coefficient in first prinicipal component
        p2: np.ndarray (N,T)
            Diffusion coefficient in second prinicipal component
        theta: np.ndarray (N,T)
            Angle w/ respect to x-axis, corresponding to the orientation of the first prinicipal component
        fov: np.ndarray (2,)
            Window size of imaging
        N: int
            Number of trajectories to generate
        T: int
            Number of timesteps to simulate for each trajectory
        dt: int/float
            Timestep for simulation (now unitless and set to 1)
    Returns:
        pos: np.ndarray (N,T,2)
            Array containing the 2D trajectories
    """
    # Create displacements across all particles and time steps
    disp = create_multi_state_displacements(p1, p2, theta, N, T, dt)

    # Get position trajectories
    #pos_0 = np.random.uniform(fov[0]/4, 3*fov[0]/4, size=(N,2))
    pos = np.cumsum(disp, axis=1) # shape: (N,T,2)
    #pos += pos_0[:, None, :] # shift all positions by starting point

    return pos
    
def create_displacements(p1: np.ndarray, p2: np.ndarray, theta: np.ndarray, N: int, T: int, dt: float):
    """
    Create displacements for all time steps

    Args:
        p1: np.ndarray (N,)
            Diffusion coefficient in first prinicipal component
        p2: np.ndarray (N,)
            Diffusion coefficient in second prinicipal component
        theta: np.ndarray (N,)
            Angle w/ respect to x-axis, corresponding to the orientation of the first prinicipal component
        N: int
            Number of trajectories to generate
        T: int
            Number of timesteps to simulate for each trajectory
        dt: int/float
            Timestep for simulation (now unitless and set to 1)
    Returns:
        disp: np.ndarray(N,T,2)
            Displacements for each step in the simulations  
    """
    # Create array with diffusion principal values for each trajectory
    D = np.stack((p1, p2), axis=-1) # shape: (N,2)

    # Generate std deviations across principal axes
    sig = np.sqrt(2.0 * dt * D) # shape: (N,2)

    # Create rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]]) # shape: (2,2,N)
    R = np.moveaxis(R, -1, 0)                      # -> (N,2,2)

    # Create step sizes in each direction generated from normal distribution
    steps = np.random.normal(0.0, 1.0, size=(N,T,2))

    # Create displacements
    tmp = steps * sig[:, None, :] # shape: (N,T,2)
    disp = (R[:, None, :, :] @ tmp[..., None]).squeeze(-1) # shape: (N,T,2)

    return disp

def create_trajectories(p1: np.ndarray, p2: np.ndarray, theta: np.ndarray, N: int, T: int, dt: float):
    """
    Create array of trajectories given diffusion parameters

    Args:
        p1: np.ndarray (N,)
            Diffusion coefficient in first prinicipal component
        p2: np.ndarray (N,)
            Diffusion coefficient in second prinicipal component
        theta: np.ndarray (N,)
            Angle w/ respect to x-axis, corresponding to the orientation of the first prinicipal component
        fov: np.ndarray (2,)
            Window size of imaging
        N: int
            Number of trajectories to generate
        T: int
            Number of timesteps to simulate for each trajectory
        dt: int/float
            Timestep for simulation (now unitless and set to 1)
    Returns:
        pos: np.ndarray (N,T,2)
            Array containing the 2D trajectories
    """
    # Create displacements across all particles and time steps
    disp = create_displacements(p1, p2, theta, N, T, dt)

    # Get position trajectories
    #pos_0 = np.random.uniform(fov[0]/4, 3*fov[0]/4, size=(N,2))
    pos = np.cumsum(disp, axis=1) # shape: (N,T,2)
    #pos += pos_0[:, None, :] # shift all positions by starting point

    return pos
    
def create_training_set(N: int, T: int, image_props: dict, dt: float=0.001, fov: np.ndarray | None=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates new training data containing video frames of trajectories and labels for a given cycle of training

    Args:
        N: int
            Number of trajectories to generate in a simulation
        T: int
            number of timesteps for each simulation
        image_props: dict
            Parameters needed for simulation
        dt: int
            Time step to be used for the simulation
        fov: np.ndarray
            Gives fov for viewing a simulation trajectory and used for random starting point in trajectory
         
    Returns:
        videos: np.ndarray 
                    Array of images for each trajectory
        labels: np.ndarray
                    Array of labels for each trajectory generated 
    """
    # Generate random diffusion parameters
    props = generate_diffusion_props(N, image_props['D_min'], image_props['D_max'], image_props['angle_max'], is_isotropic=False)
    p1, p2, theta = props[:,0], props[:,1], props[:,2]

    # if fov is None:
    #     fov = np.array([128,128])
        
    # Create trajectories
    pos = create_trajectories(p1, p2, theta, N, T, dt)
    
    # Get labels
    labels = np.stack((p1,p2,theta), axis=-1) # shape: (N,3)

    videos = trajectories_to_videos(pos, image_props)
    
    return videos, labels

def create_training_set_w_features(N: int, T: int, image_props: dict, dt: float=0.001, fov: np.ndarray | None=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates new training data containing video frames of trajectories and labels for a given cycle of training

    Args:
        N: int
            Number of trajectories to generate in a simulation
        T: int
            number of timesteps for each simulation
        image_props: dict
            Parameters needed for simulation
        dt: int
            Time step to be used for the simulation
        fov: np.ndarray
            Gives fov for viewing a simulation trajectory and used for random starting point in trajectory
         
    Returns:
        videos: np.ndarray 
                    Array of images for each trajectory
        displacements: np.ndarray
                    Array of displacements between each pair of frames
        labels: np.ndarray
                    Array of labels for each trajectory generated 
    """
    # Generate random diffusion parameters
    props = generate_diffusion_props(N, image_props['D_min'], image_props['D_max'], image_props['angle_max'], is_isotropic=False)
    p1, p2, theta = props[:,0], props[:,1], props[:,2]

    # if fov is None:
    #     fov = np.array([0,0])
        
    # Create trajectories
    pos = create_trajectories(p1, p2, theta, N, T, dt)

    # Get labels
    labels = np.stack((p1,p2,theta), axis=-1) # shape: (N,3)

    videos, centroids = trajectories_to_videos_and_centroids(pos, image_props)

    # Calculate relative displacement using centroids
    _, nFrames, _ = centroids.shape
    displacement = centroids[:, 1:, :] - centroids [:, :nFrames-1, :]

    return videos, displacement, labels

def create_multi_state_dataset_w_features(N: int, T: int, image_props: dict, dt: float=0.001, fov: np.ndarray | None=None, binding: bool=False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates a new dataset for 2-state particle simulations

    Args:
        N: int
            Number of trajectories to generate in a simulation
        T: int
            number of timesteps for each simulation
        image_props: dict
            Parameters needed for simulation
        dt: int
            Time step to be used for the simulation
        fov: np.ndarray
            Gives fov for viewing a simulation trajectory and used for random starting point in trajectory
         
    Returns:
        videos: np.ndarray 
                    Array of images for each trajectory
        displacements: np.ndarray
                    Array of displacements between each pair of frames
        labels: np.ndarray
                    Array of labels for each trajectory generated 
    """
    # Generate random diffusion props for 1st state and 2nd state
    if not binding:
        props_0 = generate_diffusion_props(N, image_props['D_min'], image_props['D_max'], image_props['angle_max'], is_isotropic=True)
        props_1 = generate_diffusion_props(N, image_props['D_min'], image_props['D_max'], image_props['angle_max'], is_isotropic=True)
        
        # alpha = np.random.uniform(0.1, 1, size=N)
        # p1_1 = alpha*p1_0
        # p2_1 = p1_1
        # theta_1 = np.zeros(N)
    else:
        print("Creating a binding dataset")
        
        bind_min, bind_max = 0.005, 0.01
        #bind_min, bind_max = 0.5, 1
        n = N//2

        # First state
        bound_0 = generate_diffusion_props(n, bind_min, bind_max, image_props['angle_max'], is_isotropic=True, is_binding=True)
        unbound_0 = generate_diffusion_props(n, image_props['D_min'], image_props['D_max'], image_props['angle_max'], is_isotropic=False)

        # Second state
        unbound_1 = generate_diffusion_props(n, image_props['D_min'], image_props['D_max'], image_props['angle_max'], is_isotropic=False)
        bound_1 = generate_diffusion_props(n, bind_min, bind_max, image_props['angle_max'], is_isotropic=True, is_binding=True)

        props_0 = np.concatenate((bound_0, unbound_0), axis=0)
        props_1 = np.concatenate((unbound_1, bound_1), axis=0)

    p1_0, p2_0, theta_0 = props_0[:, 0], props_0[:, 1], props_0[:, 2]
    p1_1, p2_1, theta_1 = props_1[:, 0], props_1[:, 1], props_1[:, 2]
   
    # if fov is None:
    #     fov = np.array([128,128])

    # Get random point to make transition between states
    #T_0 = np.random.randint(T//6, 5*T//6, size=N)
    T_0 = np.random.randint(1, T, size=N)

    # Make time-varying diffusion properties
    p1 = np.zeros((N,T))
    p2 = np.zeros_like(p1)
    theta = np.zeros_like(p1)

    for i in range(N):
        p1[i] = np.concatenate((
                    np.tile(p1_0[i, None], (1, T_0[i])),
                    np.tile(p1_1[i, None], (1, T - T_0[i]))
                ),
                    axis=-1
            )
        p2[i] = np.concatenate((
                    np.tile(p2_0[i, None], (1, T_0[i])),
                    np.tile(p2_1[i, None], (1, T - T_0[i]))
                ),
                    axis=-1
            )
        theta[i] = np.concatenate((
                    np.tile(theta_0[i, None], (1, T_0[i])),
                    np.tile(theta_1[i, None], (1, T - T_0[i]))
                ),
                    axis=-1
            )
        
    # Generate trajectories
    pos = create_multi_state_trajectories(p1, p2, theta, N, T, dt)

    labels = np.stack((p1, p2, theta), axis=-1) # shape: (N,T,3)
    labels = labels.reshape(N, image_props['frames'], -1, 3).mean(axis=2) # shape: (N,nFrames,3)

    videos, centroids = trajectories_to_videos_and_centroids(pos, image_props)

    # Calculate relative displacement using centroids
    _, nFrames, _ = centroids.shape
    displacement = centroids[:, 1:, :] - centroids [:, :nFrames-1, :]
    
    return videos, displacement, labels

def gaussian_2d(xc, yc, sigma, grid_size, amplitude):
    """
    Generates a 2D Gaussian point spread function (PSF) centered at a specified position.

    Parameters:
    - xc, yc (float): The center coordinates (x, y) of the Gaussian within the grid.
    - sigma (float): Standard deviation of the Gaussian, controlling the spread (related to FWHM).
    - grid_size (int): Size of the output grid (grid will be grid_size x grid_size).
    - amplitude (float): Peak amplitude of the Gaussian function.

    Returns:
    - gauss (ndarray): A 2D array representing the Gaussian function centered at (xc, yc).
    """
    limit = (grid_size - 1) // 2  # Defines the range for x and y axes
    x = np.linspace(-limit, limit, grid_size)
    y = np.linspace(-limit, limit, grid_size)
    x, y = np.meshgrid(x, y)
    
    # Calculate the Gaussian function centered at (xc, yc)
    gauss = amplitude * np.exp(-(((x - xc) ** 2) / (2 * sigma ** 2) + ((y - yc) ** 2) / (2 * sigma ** 2)))
    return gauss

def get_props(image_props: dict) -> tuple:
    """
    Fetch image props from dictionary
    """
    nFrames = image_props['frames']

    resolution = image_props["resolution"]
    traj_unit = image_props["trajectory_unit"]

    output_size = image_props["output_size"]
    upsampling_factor = image_props["upsampling_factor"]
    #psf_div_factor = image_props["psf_division_factor"]
    
    # Psf diameter is computed as wavelenght/2NA according to:
    #https://www.sciencedirect.com/science/article/pii/S0005272819301380?via%3Dihub
    
    # This source says fwhm = 0.51 * wavelength / NA
    #https://www.leica-microsystems.com/science-lab/life-science/microscope-resolution-concepts-factors-and-calculation/
    
    #fwhm_psf = image_props["wavelength"] / 2 * image_props["NA"] / psf_div_factor # student's old code
    #gaussian_sigma = upsampling_factor/resolution * (fwhm_psf/2.355)
    
    # direct fwhm equation
    fwhm_psf = 0.51 * image_props['wavelength'] / image_props['NA'] # unit: m
    
    # fwhm = 2.355 * sigma for a normal distribution then we need to correct for the upsampling and resolution
    fwhm_px = fwhm_psf / resolution # units: pixel
    gaussian_sigma = upsampling_factor * (fwhm_px / 2.355) # unit: upsampled pixels
    poisson_noise = image_props["poisson_noise"]
    gaussian_noise = image_props["gaussian_noise"]
    
    particle_mean, particle_std = image_props["particle_intensity"][0],image_props["particle_intensity"][1]
    background_mean, background_std = image_props["background_intensity"][0],image_props["background_intensity"][1]
    nPosPerFrame = image_props['n_pos_per_frame']

    return (nFrames, traj_unit, resolution, output_size, gaussian_sigma, poisson_noise, 
        particle_mean, particle_std, background_mean, background_std, nPosPerFrame, upsampling_factor, gaussian_noise) 

def trajectories_to_videos(pos: np.ndarray, image_props: dict) -> np.ndarray: 
    """
    Transforms trajectory data into microscopy imagery data and centroid data

    Args:
        pos: np.ndarray (N,T,2)
            Trajectory data for N particles across T timesteps in x,y dimensions
        nPosPerFrame: int
            The number of trajectory points used to create one frame
        image_props: dict
            Dictionary containing the properties needed for image creation
    Returns:
        videos: np.ndarray (N, nFrames, output_size, output_size)

    Andi simulation code 128x128 px^2 fov, dt=0.1s, D in px^2/frame, resolution of 100nm per pixel, D on the order of 0.01 um^2/s -> pixel size and framerate tell us D=0.1 px^2/ dt
    """
    # Get trajectory dimensions
    N, _, _ = pos.shape
    
    # Invert the y axis, for video creation purposes where y-axis is inverted
    pos[:, :, 1] *= -1

    # Get imaging properties
    (nFrames, traj_unit, resolution, output_size, gaussian_sigma, poisson_noise, 
        particle_mean, particle_std, background_mean, background_std, nPosPerFrame, upsampling_factor, gaussian_noise) = get_props(image_props)
    
    # if(traj_unit !=-1 ):
    #     # put the trajectory in pixels
    #     pos = pos * traj_unit / (resolution* 1e9)

    # Convert from um to pixels assuming resolution in m
    pos = pos * 1e-6 / resolution
    
    out_videos = np.zeros((N,nFrames,output_size,output_size), np.float32)

    for n in range(N):
        generate_video_vectorized(out_videos[n,:],pos[n,:],nFrames,output_size,upsampling_factor,nPosPerFrame,
                                            gaussian_sigma,particle_mean,particle_std,background_mean,background_std, poisson_noise, gaussian_noise)
    
    videos = normalize_images(out_videos, background_mean, background_std, particle_mean + background_mean)
    
    return videos

def trajectories_to_videos_and_centroids(pos: np.ndarray, image_props: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Transforms trajectory data into microscopy imagery data and centroid data

    Args:
        pos: np.ndarray (N,T,2)
            Trajectory data for N particles across T timesteps in x,y dimensions
        nPosPerFrame: int
            The number of trajectory points used to create one frame
        image_props: dict
            Dictionary containing the properties needed for image creation
    Returns:
        videos: np.ndarray (N, nFrames, output_size, output_size)
        centroids: np.ndarray (N, nFrames, 2)
    """
    # Get trajectory dimensions
    N, _, _ = pos.shape
    
    # Invert the y axis, for video creation purposes where y-axis is inverted
    pos[:, :, 1] *= -1

    # Get imaging properties
    (nFrames, traj_unit, resolution, output_size, gaussian_sigma, poisson_noise, 
        particle_mean, particle_std, background_mean, background_std, nPosPerFrame, upsampling_factor, gaussian_noise) = get_props(image_props)

    # if(traj_unit !=-1 ):
    #     # put the trajectory in pixels
    #     pos = pos * traj_unit / (resolution* 1e9)
    
    # Convert from um to pixels assuming resolution in m
    pos = pos * 1e-6 / resolution
    
    out_videos = np.zeros((N,nFrames,output_size,output_size),np.float32)
    centroids = np.zeros((N, nFrames, 2))

    for n in range(N):
        generate_video_vectorized(out_videos[n,:],pos[n,:],nFrames,output_size,upsampling_factor,nPosPerFrame,
                                            gaussian_sigma,particle_mean,particle_std,background_mean,background_std, poisson_noise, gaussian_noise, centroids[n,:])
    
    videos = normalize_images(out_videos, background_mean, background_std, particle_mean + background_mean)
    
    return videos, centroids

def generate_video(out_video: np.ndarray, trajectory: np.ndarray, nFrames: int, output_size: int, upsampling_factor: int, nPosPerFrame: int, 
                   gaussian_sigma: float, particle_mean: float, particle_std: float, background_mean: float, background_std: float, poisson_noise: int, gaussian_noise: bool, centroid: np.ndarray | None=None):
    """Helper function of function above, all arguments documented above"""
    for f in range(nFrames):
        frame_hr = np.zeros(( output_size*upsampling_factor, output_size*upsampling_factor),np.float32)
        frame_lr = np.zeros((output_size, output_size),np.float32)

        start = f*nPosPerFrame
        end = (f+1)*nPosPerFrame
        # Center the trajectory data across a segment of nPosPerFrame
        mean = np.mean(trajectory[start:end,:],axis=0)  
        trajectory_segment = trajectory[start:end,:] - mean
        xtraj = trajectory_segment[:,0]  * upsampling_factor
        ytraj = trajectory_segment[:,1] * upsampling_factor


        # Generate frame, convolution, resampling, noise
        frame_intensity = np.random.normal(particle_mean, particle_std)
        spot_intensity = frame_intensity / nPosPerFrame
        for p in range(nPosPerFrame):
            if(particle_mean >0.0001 and particle_std > 0.0001):
                #spot_intensity = np.random.normal(particle_mean/nPosPerFrame,particle_std/nPosPerFrame)
                frame_spot = gaussian_2d(xtraj[p], ytraj[p], gaussian_sigma, output_size*upsampling_factor, spot_intensity)

                # gaussian_2d maximum is not always the wanted one because of some misplaced pixels. 
                # We can force the peak of the gaussian to have the right intensity
                spot_max = np.max(frame_spot)
                if(spot_max < 0.00001):
                    print("Particle Left the image")
                frame_hr += spot_intensity/spot_max * frame_spot
        
        frame_lr = block_reduce(frame_hr, block_size=upsampling_factor, func=np.mean)
        # Add Gaussian noise to background intensity across the image
        if gaussian_noise:
            frame_lr += np.clip(np.random.normal(background_mean, background_std, frame_lr.shape), 
                                        0, background_mean + 3 * background_std)
        
        # Add poisson noise if specified
        if poisson_noise != -1:
            frame_lr = frame_lr * np.random.poisson(poisson_noise, size=(frame_lr.shape)) / poisson_noise

        out_video[f,:] = frame_lr
        
        # Case where we want features for relative displacement between frames
        if centroid is not None:    
            centroid[f,:] = get_image_centroid(frame_lr, output_size) + mean

def generate_video_vectorized(out_video: np.ndarray, trajectory: np.ndarray, nFrames: int, output_size: int, upsampling_factor: int, nPosPerFrame: int, 
                   gaussian_sigma: float, particle_mean: float, particle_std: float, background_mean: float, background_std: float, poisson_noise: int, gaussian_noise: bool, centroid: np.ndarray | None=None):
    """
    Vectorized version of the above function
    """
    grid_size = output_size * upsampling_factor

    # Precompute coordinate grid for Gaussian generation
    limit = (grid_size - 1) // 2  # Defines the range for x and y axes
    x = np.linspace(-limit, limit, grid_size)
    y = np.linspace(-limit, limit, grid_size)
    xx, yy = np.meshgrid(x, y)

    # Center trajectory segments
    trajectory = trajectory.reshape(nFrames, nPosPerFrame, 2)
    means = trajectory.mean(axis=1, keepdims=True) # (nFrames, 1, 2)
    traj_centered = (trajectory - means) * upsampling_factor
    xc = traj_centered[..., 0]
    yc = traj_centered[..., 1]
    
#     psf_radius = 0 #1.17*gaussian_sigma
#     out_of_frame = (
#     (xc - psf_radius < -limit) |
#     (xc + psf_radius >  limit) |
#     (yc - psf_radius < -limit) |
#     (yc + psf_radius >  limit)
# )  # shape: (nFrames, nPosPerFrame)
    
#     assert np.any(out_of_frame) == False
#     print("PSF leaves frame: ", np.any(out_of_frame, axis=1).mean())
#     print("Grid half width (upsampled px):", grid_size / 2)
#     print("PSF sigma (upsampled px):", gaussian_sigma)

    # Generate random variables
    frame_intensity = np.random.normal(particle_mean, particle_std, size=(nFrames,1,1))
    spot_intensity = frame_intensity / nPosPerFrame

    # Vectorized Gaussian computation for all particles
    x_diff = xx[None, None, :, :] - xc[:, :, None, None]
    y_diff = yy[None, None, :, :] - yc[:, :, None, None]
    gaussians = np.exp(-(x_diff**2 + y_diff**2) / (2 * gaussian_sigma**2)) # (nFrames, nPosPerFrame, grid_size, grid_size)

    # Normalize each Gaussian to its own max, then scale
    gaussians /= np.max(gaussians, axis=(2, 3), keepdims=True)
    frame_hr = spot_intensity * np.sum(gaussians, axis=1)

    # Downsample (block average)
    frame_lr = np.stack([
            block_reduce(frame_hr[f], block_size=upsampling_factor, func=np.mean)
            for f in range(nFrames)
        ]) # shape (nFrames, output_size, output_size)

    # Background Gaussian noise
    if gaussian_noise:
        noise = np.random.normal(background_mean, background_std, size=frame_lr.shape)
        noise = np.clip(noise, 0, background_mean + 3 * background_std)
        frame_lr += noise

    # Optional Poisson noise
    if poisson_noise != -1:
        poisson = np.random.poisson(poisson_noise, size=frame_lr.shape) / poisson_noise
        frame_lr *= poisson

    # Case where we want features for relative displacement between frames
    if centroid is not None:
        limit = (output_size - 1) // 2
        x = np.linspace(-limit, limit, output_size)
        coords = np.stack(np.meshgrid(x,x), axis=-1) # (output_size, output_size, 2)
        total_intensity = np.sum(frame_lr, axis=(1,2))[:, None]
        centroid[...] = np.sum(coords * frame_lr[..., None], axis=(1,2)) / total_intensity
        centroid[...] += means.squeeze(1)

    out_video[...] = frame_lr

def get_image_centroid(image: np.ndarray, grid_size: int) -> np.ndarray:
    """
    Compute the intensity centroid of an image

    Args:
        image: np.ndarray
            Input image to find intensity centroid of
        grid_size: int
            Size of grid/image
    Returns:
        Centroid of image intensity
    """
    # Get grid with center (0,0)
    limit = (grid_size - 1) // 2
    x = np.linspace(-limit, limit, grid_size)
    y = np.linspace(-limit, limit, grid_size)
    coords = np.stack(np.meshgrid(x, y), axis=-1) # (grid_size, grid_size, 2)

    # Compute intensity centroid
    total_intensity = np.sum(image)
    
    return np.sum(coords*image[:,:,None], axis=(0,1)) / total_intensity

def normalize_images(images: np.ndarray, background_mean=None, background_sigma=None, theoretical_max=None) -> np.ndarray:
    """
    Normalize images according to the formula:
    im_norm = (im - (background_mean - background_sigma)) / (theoretical_max - (background_mean - background_sigma))

    Args:
        images: np.ndarray (N, number of frames, output_size, output_size)
        background_mean : float, optional
            Mean of the background. If None, computed as np.mean(images)
        background_sigma : float, optional
            Standard deviation of the background. If None, computed as np.std(images)
        theoretical_max : float, optional
            Maximum theoretical value of particle that would not move. If None, computed as np.max(images)
    Returns:
        normalized: np.ndarray (N, number of frames, output_size, output_size)
            Normalized images with the same shape as input
    """

    # Compute statistics if not provided
    if background_mean is None:
        background_mean = np.mean(images)
    
    if background_sigma is None:
        background_sigma = np.std(images)
    
    if theoretical_max is None:
        theoretical_max = np.max(images)
    
    # Apply normalization
    denominator = theoretical_max - (background_mean - background_sigma)
    
    # Avoid division by zero
    if denominator == 0:
        raise ValueError("Denominator in normalization is zero. Check your inputs.")

    normalized = (images - (background_mean - background_sigma)) / denominator
    
    return normalized

def play_video(video, figsize=(5, 5), fps=5, vmin=None, vmax=None, save_path=None):
    """
    Displays a stack of images as a video inside jupyter notebooks with consistent intensity scaling.

    Parameters
    ----------
    video : ndarray
        Stack of images.
    figsize : tuple, optional
        Canvas size of the video.
    fps : int, optional
        Video frame rate.
    vmin : float, optional
        Minimum intensity value for all frames. If None, will be automatically determined.
    vmax : float, optional
        Maximum intensity value for all frames. If None, will be automatically determined.

    Returns
    -------
    Video object
        Returns a video player with input stack of images.
    """
    fig = plt.figure(figsize=figsize)
    images = []

    if(len(video.shape) == 3):
        video = np.expand_dims(video,axis=-1)

    plt.axis("off")
    
    # If vmin/vmax not provided, compute global min/max across all frames
    if vmin is None:
        vmin = np.min([frame[:, :, 0].min() for frame in video])
    if vmax is None:
        vmax = np.max([frame[:, :, 0].max() for frame in video])
    mean = np.mean(video)

    print(f"vmin: {vmin} vmax: {vmax} mean: {mean:.2f}")

    for image in video:
        images.append([plt.imshow(image[:, :, 0], cmap="gray", vmin=vmin, vmax=vmax)])

    anim = animation.ArtistAnimation(
        fig, images, interval=1e3 / fps, blit=True, repeat_delay=0
    )

    html = HTML(anim.to_jshtml())
    display(html)

    # Save the animation if a save path is provided
    if save_path:
        if save_path.endswith('.mp4'):
            # Use FFMpegWriter for MP4 files (requires FFmpeg installed)
            writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
            anim.save(save_path, writer=writer)
            print(f"Animation saved to {save_path}")
        elif save_path.endswith('.gif'):
            # Use PillowWriter for GIF files
            writer = animation.PillowWriter(fps=fps)
            anim.save(save_path, writer=writer)
            print(f"Animation saved to {save_path}")
        else:
            print("Unsupported file format. Use .mp4 or .gif extension.")
    plt.close()