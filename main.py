import torch
from helpers.simulation import * # Functions to create trajectory data and their respective video frames
from helpers.models import *
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_multi_modal(model: DiffusionTensorRegModelBase, N: int, T: int, props: dict, image_props: dict, device: torch.device, cycles: int, losses: list):
    """
    Training for multimodal input

    Args:
        model:
            Model to train
        N: int
            Number of particles to simulate
        T: int
            Number of timesteps in a simulation
        props: dict
            Parameters of the model
        image_props: dict
            Parameters for imaging simulation
        device: torch.device
            Device to put data on
        cycles: int
            Number of cycles of training to go through
    """
    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), props['lr'])

    model.to(device)

    # Training cycles -> one epoch but we generate training data in smaller batches
    for _ in tqdm(range(cycles), desc='Cycles of training'):
        # Create training data
        all_videos, all_displacements, all_labels = create_training_set_w_features(N, T, image_props) 
        
        # Normalize labels for better optimization
        D_max_normalization, angle_max_normalization = props['D_max_normalization'], props['angle_max_normalization']
        all_labels = all_labels / np.array([D_max_normalization, D_max_normalization, angle_max_normalization])
        
        # Convert to tensors
        all_videos = torch.Tensor(all_videos)
        all_labels = torch.Tensor(all_labels)
        all_displacements = torch.Tensor(all_displacements)

        # Create dataset and dataloader objects
        dataset = VideoMotionDataset(all_videos, all_displacements, all_labels)
        dataloader = DataLoader(dataset, batch_size=props['batch_size'], shuffle=True)

        batch_loss = []
        # Train
        for videos, displacements, labels in dataloader:
            videos = videos.to(device)
            displacements = displacements.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(videos, displacements)

            loss = log_euclidean_loss(labels, output)

            batch_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        losses.append(np.mean(batch_loss))

def train_single_mode(model: DiffusionTensorRegModelBase, N: int, T: int, props: dict, image_props: dict, device: torch.device, cycles: int, losses: list):
    """
    Training for image only input

    Args:
        model:
            Model to train
        N: int
            Number of particles to simulate
        T: int
            Number of timesteps in a simulation
        props: dict
            Parameters of the model
        image_props: dict
            Parameters for imaging simulation
        device: torch.device
            Device to put data on
        cycles: int
            Number of cycles of training to go through
    """
    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), props['lr'])
    #scheduler = 

    model.to(device)

    # Training cycles -> one epoch but we generate training data in smaller batches
    for _ in tqdm(range(cycles), desc='Cycles of training'):

        # Create training data
        all_videos, all_labels = create_training_set(N, T, image_props)
        
        # Normalize labels for better optimization
        D_max_normalization, angle_max_normalization = props['D_max_normalization'], props['angle_max_normalization']
        all_labels = all_labels / np.array([D_max_normalization, D_max_normalization, angle_max_normalization])
        
        # Convert to tensors
        all_videos = torch.Tensor(all_videos)
        all_labels = torch.Tensor(all_labels)

        # Create dataset and dataloader objects
        dataset = VideoDataset(all_videos, all_labels)
        dataloader = DataLoader(dataset, batch_size=props['batch_size'], shuffle=True)

        batch_loss = []
        # Train
        for videos, _, labels in dataloader:
            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(videos)

            loss = log_euclidean_loss(labels, output)
            
            batch_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        losses.append(np.mean(batch_loss))

def main():
    # Define model run type
    use_rel_displacement_features = False
    use_sum=True

    # Define model parameters
    model_props= {
        "D_max_normalization": 10,
        "angle_max_normalization": np.pi,
        "lr": 1e-4,
        "embed_dim": 64,
        "num_heads": 4,
        "hidden_dim": 128,
        "num_layers": 6,
        "dropout": 0.0,
        "batch_size": 16,
        "disp_feat_dim": 8
    }

    # Image parameters
    N = 1000 # number of particles to simulate
    nPosPerFrame = 10 
    nFrames = 30
    T = nFrames * nPosPerFrame
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
        "output_size": 9,
        "upsampling_factor": 5,
        "background_intensity": [
            background_mean,
            background_sigma,
        ],  # Standard deviation of background intensity within a video
        "poisson_noise": -1, #100,
        "trajectory_unit" : 1200
    }

    # Number of cycles of generating training data and training
    cycles = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Keep track of loss across training for plotting
    losses = []

    # Train model based on run type
    if use_rel_displacement_features:
        model = DiffusionTensorRegModel(model_props['embed_dim'], model_props['num_heads'], model_props['hidden_dim'], 
                model_props['num_layers'], model_props['dropout'], model_props['disp_feat_dim'], use_sum=use_sum)
        train_multi_modal(model, N, T, model_props, image_props, device, cycles, losses)
    else:
        model = DiffusionTensorRegModelBase(model_props['embed_dim'], model_props['num_heads'], model_props['hidden_dim'], 
                model_props['num_layers'], model_props['dropout'])
        train_single_mode(model, N, T, model_props, image_props, device, cycles, losses)

    # Evaluation

    # Get validation data
    data = np.load('validation_data.npz')
    videos = torch.Tensor(data['vids'])
    displacements = torch.Tensor(data['disp'])
    labels = torch.Tensor(data['labels'])
    
    D_max_normalization = model_props['D_max_normalization']
    angle_max_normalization = model_props['angle_max_normalization']
    labels = labels / np.array([D_max_normalization, D_max_normalization, angle_max_normalization])

    # Create dataset and dataloader
    if use_rel_displacement_features:
        dataset = VideoMotionDataset(videos, displacements, labels)
    else:
        dataset = VideoDataset(videos, labels)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model.eval()
    with torch.no_grad():
        loss = []
        for videos, displacements, labels in dataloader:
            videos = videos.to(device)
            labels = labels.to(device)
            
            if displacements.numel() != 0:
                displacements = displacements.to(device)
                val_predictions = model(videos, displacements)
            else:
                val_predictions = model(videos)
            val_loss = log_euclidean_loss(labels, val_predictions)
            avg_val_loss = val_loss.item()
            loss.append(avg_val_loss) # batch loss
        
        # Compute average across all labels
        avg_loss = np.mean(loss)
        print(f"Average loss across validation set: {avg_loss}")

    # Plot loss over time during training
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_title('Loss across training cycles')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Cycle')
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
   main()