import numpy as np
from skimage.measure import block_reduce

def get_random_diffusion_params():
    pass

def create_trajectories(p1: np.ndarray, p2: np.ndarray, theta: np.ndarray, fov: np.ndarray, N: int, T: int, dt: int):
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

    # Get position trajectories
    pos_0 = np.random.uniform(fov[0]/4, 3*fov[0]/4, size=(N,2))
    disp[:,0,:] = pos_0 # Set time0 to 0 displacement
    pos = np.cumsum(disp, axis=1) # shape: (N,T,2)

    return pos
    
def create_training_set(N: int, T: int, image_props: dict, dt: int=1, fov: np.ndarray | None=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates new training data containing video frames of trajectories and labels for a given cycle of training

    Args:
        N (int): integer representing the number of trajectories to generate in a simulation
        T (int): integer representing the number of timesteps for each simulation 
         
    Returns:
        videos: np.ndarray 
                    Array of images for each trajectory
        labels: np.ndarray
                    Array of labels for each trajectory generated
    """
    
    # Generate random diffusion parameters
    p1 = np.random.randint(1, 11, size=N)
    alpha = np.random.uniform(0.1, 1, size=N)
    p2 = alpha*p1
    theta = np.random.uniform(0, np.pi, size=N)

    if fov is None:
        fov = np.array([128,128])
        
    # Create trajectories
    pos = create_trajectories(p1, p2, theta, fov, N, T, dt)
    
    # Get labels
    labels = np.stack((p1,p2,theta), axis=-1) # shape: (N,3)

    # Convert trajectories of D (pixels^2/s) to D (micro_m^2/ms)
    scaled_pos = pos / 100
    videos = trajectories_to_videos(scaled_pos, image_props)
    
    return videos, labels

def create_training_set_w_features(N: int, T: int, image_props: dict, dt: int=1, fov: np.ndarray | None=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates new training data containing video frames of trajectories and labels for a given cycle of training

    Args:
        N (int): integer representing the number of trajectories to generate in a simulation
        T (int): integer representing the number of timesteps for each simulation 
         
    Returns:
        videos: np.ndarray 
                    Array of images for each trajectory
        displacements: np.ndarray
                    Array of displacements between each pair of frames
        labels: np.ndarray
                    Array of labels for each trajectory generated
    """
    
    # Generate random diffusion parameters
    p1 = np.random.randint(1, 11, size=N)
    alpha = np.random.uniform(0.1, 1, size=N)
    p2 = alpha*p1
    theta = np.random.uniform(0, np.pi, size=N)

    if fov is None:
        fov = np.array([128,128])
        
    # Create trajectories
    pos = create_trajectories(p1, p2, theta, fov, N, T, dt)
    
    # Get labels
    labels = np.stack((p1,p2,theta), axis=-1) # shape: (N,3)

    # Convert trajectories of D (pixels^2/s) to D (micro_m^2/ms)
    scaled_pos = pos / 100
    videos, centroids = trajectories_to_videos_and_centroids(scaled_pos, image_props)

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

def get_props(T: int, image_props: dict) -> tuple:
    """
    Fetch image props from dictionary
    """
    nFrames = image_props['frames']

    resolution =image_props["resolution"]
    traj_unit = image_props["trajectory_unit"]

    output_size = image_props["output_size"]
    upsampling_factor = image_props["upsampling_factor"]
    psf_div_factor = image_props["psf_division_factor"]
    
    # Psf is computed as wavelenght/2NA according to:
    #https://www.sciencedirect.com/science/article/pii/S0005272819301380?via%3Dihub
    fwhm_psf = image_props["wavelength"] / 2 * image_props["NA"] / psf_div_factor

    
    gaussian_sigma = upsampling_factor/ resolution * fwhm_psf/2.355
    poisson_noise = image_props["poisson_noise"]
    
    particle_mean, particle_std = image_props["particle_intensity"][0],image_props["particle_intensity"][1]
    background_mean, background_std = image_props["background_intensity"][0],image_props["background_intensity"][1]
    nPosPerFrame = image_props['n_pos_per_frame']

    return (nFrames, traj_unit, resolution, output_size, gaussian_sigma, poisson_noise, 
        particle_mean, particle_std, background_mean, background_std, nPosPerFrame, upsampling_factor) 

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
    N, T, _ = pos.shape
    
    # Invert the y axis, for video creation purposes where y-axis is inverted
    pos[:, :, 1] *= -1

    # Get imaging properties
    (nFrames, traj_unit, resolution, output_size, gaussian_sigma, poisson_noise, 
        particle_mean, particle_std, background_mean, background_std, nPosPerFrame, upsampling_factor) = get_props(T, image_props)
    
    if(traj_unit !=-1 ):
        # put the trajectory in pixels
        pos = pos * traj_unit / (resolution* 1e9)

    out_videos = np.zeros((N,nFrames,output_size,output_size),np.float32)

    for n in range(N):
        generate_video(out_videos[n,:],pos[n,:],nFrames,output_size,upsampling_factor,nPosPerFrame,
                                            gaussian_sigma,particle_mean,particle_std,background_mean,background_std, poisson_noise)
    
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

    Andi simulation code 128x128 px^2 fov, dt=0.1s, D in px^2/frame, resolution of 100nm per pixel, D on the order of 0.01 um^2/s -> pixel size and framerate tell us D=0.1 px^2/ dt
    """
    # Get trajectory dimensions
    N, T, _ = pos.shape
    
    # Invert the y axis, for video creation purposes where y-axis is inverted
    pos[:, :, 1] *= -1

    # Get imaging properties
    (nFrames, traj_unit, resolution, output_size, gaussian_sigma, poisson_noise, 
        particle_mean, particle_std, background_mean, background_std, nPosPerFrame, upsampling_factor) = get_props(T, image_props)

    if(traj_unit !=-1 ):
        # put the trajectory in pixels
        pos = pos * traj_unit / (resolution* 1e9)
    
    out_videos = np.zeros((N,nFrames,output_size,output_size),np.float32)
    centroids = np.zeros((N, nFrames, 2))

    for n in range(N):
        generate_video(out_videos[n,:],pos[n,:],nFrames,output_size,upsampling_factor,nPosPerFrame,
                                            gaussian_sigma,particle_mean,particle_std,background_mean,background_std, poisson_noise,centroids[n,:])
    
    videos = normalize_images(out_videos, background_mean, background_std, particle_mean + background_mean)
    
    return videos, centroids

def generate_video(out_video: np.ndarray, trajectory: np.ndarray, nFrames: int, output_size: int, upsampling_factor: int, nPosPerFrame: int, 
                   gaussian_sigma: float, particle_mean: float, particle_std: float, background_mean: float, background_std: float, poisson_noise: float, centroid: np.ndarray | None=None):
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

        frame_intensity = np.random.normal(particle_mean, particle_std)

        # Generate frame, convolution, resampling, noise
        for p in range(nPosPerFrame):
            if(particle_mean >0.0001 and particle_std > 0.0001):
                #spot_intensity = np.random.normal(particle_mean/nPosPerFrame,particle_std/nPosPerFrame)
                spot_intensity = frame_intensity / nPosPerFrame
                frame_spot = gaussian_2d(xtraj[p], ytraj[p], gaussian_sigma, output_size*upsampling_factor, spot_intensity)

                # gaussian_2d maximum is not always the wanted one because of some misplaced pixels. 
                # We can force the peak of the gaussian to have the right intensity
                spot_max = np.max(frame_spot)
                if(spot_max < 0.00001):
                    print("Particle Left the image")
                frame_hr += spot_intensity/spot_max * frame_spot
        
        frame_lr = block_reduce(frame_hr, block_size=upsampling_factor, func=np.mean)
        # Add Gaussian noise to background intensity across the image
        frame_lr += np.clip(np.random.normal(background_mean, background_std, frame_lr.shape), 
                                    0, background_mean + 3 * background_std)
        
        # Add poisson noise if specified
        if poisson_noise != -1:
            frame_lr = frame_lr * np.random.poisson(poisson_noise, size=(frame_lr.shape)) / poisson_noise

        out_video[f,:] = frame_lr
        # Case where we want features for relative motion between frames
        if centroid is not None:    
            centroid[f,:] = get_image_centroid(frame_lr, output_size) + mean

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