import torch
from helpers.simulation import *
from helpers.models import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup

def train_multi_modal(model: DiffusionTensorRegModel, N: int, T: int, props: dict, image_props: dict, device: torch.device, cycles: int, losses: list, vlosses: list):
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
        losses: list
            List to keep track of training losses
        vlosses: list
            List to keep track of validation losses
    """
    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), props['lr'], weight_decay=props.get('weight_decay', 0.01))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*cycles, num_training_steps=0*cycles)

    model.to(device)

    # Get validation data
    val_data = np.load('validation_data.npz')
    val_videos = torch.Tensor(val_data['vids'])
    val_displacements = torch.Tensor(val_data['disp'])
    val_labels = torch.Tensor(val_data['labels'])
    D_max_normalization = image_props['D_max_norm']
    val_labels = val_labels / np.array([D_max_normalization, D_max_normalization, 1])

    val_dataset = VideoMotionDataset(val_videos, val_displacements, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Create training data
    all_videos, all_displacements, all_labels = create_training_set_w_features(N, T, image_props) 

    # Normalize labels for better optimization
    all_labels = all_labels / np.array([D_max_normalization, D_max_normalization, 1])
    
    # Convert to tensors
    all_videos = torch.Tensor(all_videos)
    all_labels = torch.Tensor(all_labels)
    all_displacements = torch.Tensor(all_displacements)

    # Create dataset and dataloader objects
    dataset = VideoMotionDataset(all_videos, all_displacements, all_labels)
    dataloader = DataLoader(dataset, batch_size=props['batch_size'], shuffle=True)

    # Training
    for epoch in tqdm(range(cycles), desc='Epochs of training'):
        model.train()

        batch_loss = []
        # Train
        for videos, displacements, labels in dataloader:
            videos = videos.to(device)
            displacements = displacements.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(videos, displacements)

            loss = props['loss_fn'](labels, output)

            batch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            #scheduler.step()

        losses.append(np.mean(batch_loss))

        # Validation
        model.eval()
        with torch.no_grad():
            batch_vloss = []
            for videos, displacements, labels in val_dataloader:
                videos = videos.to(device)
                labels = labels.to(device)
                
                if displacements.numel() != 0:
                    displacements = displacements.to(device)
                    val_predictions = model(videos, displacements)
                else:
                    val_predictions = model(videos)

                l = props['loss_fn'](labels, val_predictions)
                batch_vloss.append(l.item())
            vlosses.append(np.mean(batch_vloss))

        # Update scheduler with loss from validation
        scheduler.step(vlosses[-1])

        print(f"Epoch {epoch+1}/{cycles} | Train Loss: {losses[-1]:.4f} | Val Loss: {vlosses[-1]:.4f}")


def train_single_mode(model: DiffusionTensorRegModelBase, N: int, T: int, props: dict, image_props: dict, device: torch.device, cycles: int, losses: list, vlosses: list):
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
    optimizer = torch.optim.AdamW(model.parameters(), props['lr'])
    #scheduler = 

    model.to(device)

    # Get validation data
    val_data = np.load('validation_data.npz')
    val_videos = torch.Tensor(val_data['vids'])
    val_labels = torch.Tensor(val_data['labels'])
    D_max_normalization = image_props['D_max_norm']
    val_labels = val_labels / np.array([D_max_normalization, D_max_normalization, 1])

    val_dataset = VideoDataset(val_videos, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Create training data
    all_videos, all_labels = create_training_set(N, T, image_props) 

    # Normalize labels for better optimization
    all_labels = all_labels / np.array([D_max_normalization, D_max_normalization, 1])
    
    # Convert to tensors
    all_videos = torch.Tensor(all_videos)
    all_labels = torch.Tensor(all_labels)

    # Create dataset and dataloader objects
    dataset = VideoDataset(all_videos, all_labels)
    dataloader = DataLoader(dataset, batch_size=props['batch_size'], shuffle=True)

    # Training
    for epoch in tqdm(range(cycles), desc='Epochs of training'):
        model.train()

        batch_loss = []
        # Train
        for videos, _, labels in dataloader:
            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(videos)

            loss = props['loss_fn'](labels, output)
            
            batch_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        losses.append(np.mean(batch_loss))
        
        # Validation
        model.eval()
        with torch.no_grad():
            batch_vloss = []
            for videos, labels in val_dataloader:
                videos = videos.to(device)
                labels = labels.to(device)

                val_predictions = model(videos)

                l = props['loss_fn'](labels, val_predictions)
                batch_vloss.append(l.item())
            vlosses.append(np.mean(batch_vloss))

        print(f"Epoch {epoch+1}/{cycles} | Train Loss: {losses[-1]:.4f} | Val Loss: {vlosses[-1]:.4f}")

def train_disp_mode(model: DisplacementBasedModel, N: int, T: int, props: dict, image_props: dict, device: torch.device, cycles: int, losses: list, vlosses: list):
    """
    Training for displacement only input

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
        losses: list
            List to keep track of training losses
        vlosses: list
            List to keep track of validation losses
    """
    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), props['lr'], weight_decay=props.get('weight_decay', 0.001))

    model.to(device)

    # Get validation data
    val_data = np.load('validation_data.npz')
    val_displacements = torch.Tensor(val_data['disp'])
    val_labels = torch.Tensor(val_data['labels'])
    D_max_normalization = image_props['D_max_norm']
    val_labels = val_labels / np.array([D_max_normalization, D_max_normalization, 1])

    val_dataset = DisplacementDataset(val_displacements, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Create training data
    _, all_displacements, all_labels = create_training_set_w_features(N, T, image_props) 

    # Normalize labels for better optimization
    all_labels = all_labels / np.array([D_max_normalization, D_max_normalization, 1])
    
    # Convert to tensors
    all_labels = torch.Tensor(all_labels)
    all_displacements = torch.Tensor(all_displacements)

    # Create dataset and dataloader objects
    dataset = DisplacementDataset(all_displacements, all_labels)
    dataloader = DataLoader(dataset, batch_size=props['batch_size'], shuffle=True)

    # Training
    for epoch in tqdm(range(cycles), desc='Epochs of training'):
        model.train()

        batch_loss = []
        # Train
        for displacements, _, labels in dataloader:
            displacements = displacements.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(displacements)

            loss = props['loss_fn'](labels, output)

            batch_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        losses.append(np.mean(batch_loss))

        # Validation
        model.eval()
        with torch.no_grad():
            batch_vloss = []
            for displacements, _, labels in val_dataloader:
                labels = labels.to(device)                
                displacements = displacements.to(device)
                
                val_predictions = model(displacements)

                l = props['loss_fn'](labels, val_predictions)
                batch_vloss.append(l.item())
            vlosses.append(np.mean(batch_vloss))

        print(f"Epoch {epoch+1}/{cycles} | Train Loss: {losses[-1]:.4f} | Val Loss: {vlosses[-1]:.4f}")

def main():
    # Save model weights
    save_results = False

    # Define model run type
    multimodal = True
    use_sum=True
    cross_attn_base = True
    cross_attn_complex = True
    hierarchy_model = False
    disp_model = True

    # Define model parameters
    model_props= {
        "lr": 1e-4,
        "embed_dim": 64,
        "num_heads": 4,
        "hidden_dim": 128,
        "num_layers": 4,
        "dropout": 0.01,
        "batch_size": 16,
        "loss_fn": log_euclidean_loss,
        "weight_decay": 0.01
    }

    # Image parameters
    N = 100000 # number of particles to simulate
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
        "poisson_noise": 100, # -1 for no noise,
        "gaussian_noise": True,
        "trajectory_unit" : 1200,
        "D_min": 1, # >= 1
        "D_max": 10,
        "angle_max": np.pi,
        "D_max_norm": 1, # factor to divide by for normalization
    }

    # Number of cycles of generating training data and training
    cycles = 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    np.random.seed(150)
    
    # Keep track of loss across training for plotting
    losses = []
    vlosses = []

    # Train model based on run type
    if multimodal:
        if cross_attn_base:
            if cross_attn_complex:
                model = CrossAttentionModel(model_props['embed_dim'], model_props['num_heads'], model_props['hidden_dim'], 
                    model_props['num_layers'], model_props['dropout'], output_dim=4)
            else:
                model = CrossAttentionModelBase(model_props['embed_dim'], model_props['num_heads'], model_props['hidden_dim'], 
                    model_props['num_layers'], model_props['dropout'])
        elif hierarchy_model:
            # Model using a hierarchical (one-stream to two-streams)
            model = HierarchicalModel(model_props['embed_dim'], model_props['num_heads'], model_props['hidden_dim'], 
                model_props['num_layers'], model_props['dropout'])
        else:
            # Model using simply sum or concat of modalities
            model = DiffusionTensorRegModel(model_props['embed_dim'], model_props['num_heads'], model_props['hidden_dim'], 
                    model_props['num_layers'], model_props['dropout'], use_sum=use_sum)
        print(f"Total params of {model.__class__.__name__}: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        train_multi_modal(model, N, T, model_props, image_props, device, cycles, losses, vlosses)
    else:
        if not disp_model:
            model = DiffusionTensorRegModelBase(model_props['embed_dim'], model_props['num_heads'], model_props['hidden_dim'], 
                    model_props['num_layers'], model_props['dropout'])
            print(f"Total params of {model.__class__.__name__}: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
            train_single_mode(model, N, T, model_props, image_props, device, cycles, losses, vlosses)
        else:
            model = DisplacementBasedModel(model_props['embed_dim'], model_props['num_heads'], model_props['hidden_dim'], 
                    model_props['num_layers'], model_props['dropout'])
            print(f"Total params of {model.__class__.__name__}: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
            train_disp_mode(model, N, T, model_props, image_props, device, cycles, losses, vlosses)

    # Evaluation
    model.eval()
    for file in [f for f in os.listdir() if f.endswith('npz')]:
        # Get validation data
        data = np.load(file)
        videos = torch.Tensor(data['vids'])
        displacements = torch.Tensor(data['disp'])
        labels = torch.Tensor(data['labels'])
        
        D_max_normalization = image_props['D_max_norm']
        labels = labels / np.array([D_max_normalization, D_max_normalization, 1])

        # Create dataset and dataloader
        if multimodal:
            dataset = VideoMotionDataset(videos, displacements, labels)
        else:
            if not disp_model:
                dataset = VideoDataset(videos, labels)
            else:
                dataset = DisplacementDataset(displacements, labels)

        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        with torch.no_grad():
            loss = []
            errors = []
            for videos, displacements, labels in dataloader:
                videos = videos.to(device)
                labels = labels.to(device)
                
                if displacements.numel() != 0:
                    displacements = displacements.to(device)
                    val_predictions = model(videos, displacements)
                else:
                    val_predictions = model(videos)

                val_loss = model_props['loss_fn'](labels, val_predictions)
                loss.append(val_loss.item()) # batch loss

                angle_pred = 0.5 * torch.atan2(val_predictions[:,2], val_predictions[:,3])
                if model_props['loss_fn'] is mse_loss or mse_loss_coeff:
                    mse = torch.stack([
                        (val_predictions[:,0] - labels[:,0])**2, 
                        (val_predictions[:,1] - labels[:,1])**2,
                    ], axis=1).mean(axis=0)
                else:
                    mse = torch.stack([
                        (torch.exp(val_predictions[:,0]) - labels[:,0])**2, 
                        (torch.exp(val_predictions[:,1]) - labels[:,1])**2,
                    ], axis=1).mean(axis=0)
                rmse_batch = torch.sqrt(mse)
                angle_sim = torch.abs(torch.cos(angle_pred - labels[:,-1])).mean().unsqueeze(-1)
                errors.append(torch.cat([
                        rmse_batch.cpu() * D_max_normalization,
                        angle_sim.cpu()
                    ], axis=0))

            # Compute average across all labels
            avg_loss = np.mean(loss)
            avg_rmse = np.mean(errors, axis=0)
            print(file[:-4])
            print(f"Average loss across validation set: {avg_loss}")
            print(f"Average RMSE of diffusion coefficients and angle simularity across validation set: {avg_rmse}")
            print(50*'-')

    # Save model weights
    if save_results:
        date = datetime.now().strftime('%d_%m_%Y')
        torch.save(model.state_dict(), f'results/model_{date}.pt')
        print(f"Saved model to path: results/model_{date}.pt")

    # Plot loss over time during training
    fig, ax = plt.subplots()
    ax.plot(losses, 'r', label='Training')
    ax.plot(vlosses, 'b*', label='Validation')
    ax.set_title('Loss across training cycles')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Cycle')
    ax.legend()
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
   main()