import numpy as np

nPosPerFrame = 10 # default value
nFrames = 30 # default value
# values from Real data
background_mean,background_sigma = 1420, 290
part_mean, part_std = 6000 - background_mean,500

BINDING_IMAGE_PROPS = dict(
    {
        "n_pos_per_frame": nPosPerFrame,
        "frames": nFrames,
        "particle_intensity": [
            part_mean,
            part_std,
        ], # Mean and standard deviation of the particle intensity
        "NA": 1.46,  # Numerical aperture
        "wavelength": 500e-9,  # Wavelength
        "resolution": 100e-9,  # Camera resolution or effective resolution, aka pixelsize
        "output_size": 11,
        "upsampling_factor": 5,
        "background_intensity": [
            background_mean,
            background_sigma,
        ],  # Standard deviation of background intensity within a video
        "poisson_noise": 100, #  -1 for no noise,
        "gaussian_noise": True,
        "D_min": 3, # um^2 / s
        "D_max": 6,
        "angle_max": np.pi,
        "D_max_norm": 6, # factor to divide by for normalization
    }
)

ISOTROPIC_PROPS = dict(
    {
    "n_pos_per_frame": nPosPerFrame,
    "frames": nFrames,
    "particle_intensity": [
        part_mean,
        part_std,
    ], # Mean and standard deviation of the particle intensity
    "NA": 1.46,  # Numerical aperture (unitless)
    "wavelength": 500e-9,  # Wavelength (m)
    "resolution": 100e-9,  # Camera resolution or effective resolution, aka pixelsize (m)
    "output_size": 11,
    "upsampling_factor": 5,
    "background_intensity": [
        background_mean,
        background_sigma,
    ],  # Standard deviation of background intensity within a video
    "poisson_noise": 100, #  -1 for no noise,
    "gaussian_noise": True,
    "D_min": 0.05, # um^2 /s 
    "D_max": 6,
    "angle_max": np.pi,
    "D_max_norm": 6, # factor to divide by for normalization
    }
)

SINGLE_STATE_PROPS = dict(
    {
    "n_pos_per_frame": nPosPerFrame,
    "frames": nFrames,
    "particle_intensity": [
        part_mean,
        part_std,
    ], # Mean and standard deviation of the particle intensity
    "NA": 1.46,  # Numerical aperture
    "wavelength": 500e-9,  # Wavelength
    "resolution": 100e-9,  # Camera resolution or effective resolution, aka pixelsize
    "output_size": 11,
    "upsampling_factor": 5,
    "background_intensity": [
        background_mean,
        background_sigma,
    ],  # Standard deviation of background intensity within a video
    "poisson_noise": 100, #  -1 for no noise,
    "gaussian_noise": True,
    "D_min": 0.05, # um^2 / s
    "D_max": 6,
    "angle_max": np.pi,
    "D_max_norm": 6, # factor to divide by for normalization
    }
)