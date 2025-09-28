from helpers.simulation import * # Functions to create trajectory data and their respective video frames

def main():
    # Define model hyperparameters
    patch_size = 9
    embed_dim = 64
    num_heads = 4
    hidden_dim = 128
    num_layers = 6
    dropout = 0.0


    # Image generation parameters
    traj_div_factor = 100 # need to divide trajectories because they are given in pixels/s but we want trajectories in ms domain

    nPosPerFrame = 10 
    nFrames = 30 # = Seuence length
    T = nFrames * nPosPerFrame
    # number of trajectories
    # values from Real data
    background_mean,background_sigma = 1420, 290
    part_mean, part_std = 6000 - background_mean,500

    image_props = {
        "n_pos_per_frame": nPosPerFrame,
        "frames": nFrames,
        "particle_intensity": [
            part_mean,
            part_std,
        ],  # Mean and standard deviation of the particle intensity
        "NA": 1.46,  # Numerical aperture
        "wavelength": 500e-9,  # Wavelength
        "psf_division_factor": 1.3,  
        "resolution": 100e-9,  # Camera resolution or effective resolution, aka pixelsize
        "output_size": patch_size,
        "upsampling_factor": 5,
        "background_intensity": [
            background_mean,
            background_sigma,
        ],  # Standard deviation of background intensity within a video
        "poisson_noise": -1, #100,
        "trajectory_unit" : 1200
    }
    # Simulation settings
    N = 1
    nPosFrame = 10
    nFrame = 30
    T = nPosFrame*nFrame

    # Number of cycles of generating training data and training
    cycles = 1 # 100


    # Training cycles -> one epoch but we generate training data in smaller batches
    for i in range(cycles):

        # Create training data
        all_videos, all_labels = create_training_set(N, T, image_props)

        
        # Convert to tensors
        ...

        # Create dataset and dataloader objects
        ...

        # Train


if __name__ == '__main__':
   main()
   