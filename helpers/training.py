import torch
import numpy as np
from helpers.simulation import *
from helpers.models import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

def train_multi_modal(model: DiffusionTensorRegModel, N: int, T: int, props: dict, image_props: dict, device: torch.device, cycles: int, losses: list, vlosses: list, val_path: str='data/validation_data.npz'):
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
    D_max_normalization = image_props['D_max_norm']

    # Create training data
    all_videos, all_displacements, all_labels = create_training_set_w_features(N, T, image_props)
    disp_stats = compute_displacement_statistics(all_displacements)
    all_displacements = normalize_displacements(all_displacements, disp_stats)

    # Normalize labels for better optimization
    all_labels = all_labels / np.array([D_max_normalization, D_max_normalization, 1])
    
    # Convert to tensors
    all_videos = torch.Tensor(all_videos)
    all_labels = torch.Tensor(all_labels)
    all_displacements = torch.Tensor(all_displacements)

    # Create dataset and dataloader objects
    dataset = VideoMotionDataset(all_videos, all_displacements, all_labels)
    dataloader = DataLoader(dataset, batch_size=props['batch_size'], shuffle=True)

    # Get validation data
    val_data = np.load(val_path)
    val_videos = torch.Tensor(val_data['vids'])
    val_displacements = normalize_displacements(val_data['disp'], disp_stats)
    val_displacements = torch.Tensor(val_displacements)
    val_labels = torch.Tensor(val_data['labels'])
    val_labels = val_labels / np.array([D_max_normalization, D_max_normalization, 1])

    val_dataset = VideoMotionDataset(val_videos, val_displacements, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), props['lr'], weight_decay=props.get('weight_decay', 0.01))
    scheduler = CosineAnnealingLR(optimizer, T_max=cycles, eta_min=1e-5)

    model.to(device)

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

            loss = props['loss_fn'](output, labels)

            batch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()

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

                l = props['loss_fn'](val_predictions, labels)
                batch_vloss.append(l.item())
            vlosses.append(np.mean(batch_vloss))
        
        print(f"Epoch {epoch+1}/{cycles} | Train Loss: {losses[-1]:.4f} | Val Loss: {vlosses[-1]:.4f}")
    
    return disp_stats 

def train_single_mode(model: DiffusionTensorRegModelBase, N: int, T: int, props: dict, image_props: dict, device: torch.device, cycles: int, losses: list, vlosses: list, val_path: str='data/validation_data.npz'):
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
    scheduler = CosineAnnealingLR(optimizer, T_max=cycles, eta_min=1e-5)

    model.to(device)

    # Get validation data
    val_data = np.load(val_path)
    val_videos = torch.Tensor(val_data['vids'])
    val_labels = torch.Tensor(val_data['labels'])
    D_max_normalization = image_props['D_max_norm']
    val_labels = val_labels / np.array([D_max_normalization, D_max_normalization, 1])

    val_dataset = VideoDataset(val_videos, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    
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

            loss = props['loss_fn'](output, labels)
            
            batch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        
        scheduler.step()

        losses.append(np.mean(batch_loss))
        
        # Validation
        model.eval()
        with torch.no_grad():
            batch_vloss = []
            for videos, _, labels in val_dataloader:
                videos = videos.to(device)
                labels = labels.to(device)

                val_predictions = model(videos)

                l = props['loss_fn'](val_predictions, labels)
                batch_vloss.append(l.item())
            vlosses.append(np.mean(batch_vloss))

        print(f"Epoch {epoch+1}/{cycles} | Train Loss: {losses[-1]:.4f} | Val Loss: {vlosses[-1]:.4f}")

def train_disp_mode(model: DisplacementBasedModel, N: int, T: int, props: dict, image_props: dict, device: torch.device, cycles: int, losses: list, vlosses: list, val_path: str='data/validation_data.npz'):
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
    D_max_normalization = image_props['D_max_norm']

    # Create training data
    _, all_displacements, all_labels = create_training_set_w_features(N, T, image_props)
    disp_stats = compute_displacement_statistics(all_displacements)
    all_displacements = normalize_displacements(all_displacements, disp_stats)

    # Normalize labels for better optimization
    all_labels = all_labels / np.array([D_max_normalization, D_max_normalization, 1])
    
    # Convert to tensors
    all_labels = torch.Tensor(all_labels)
    all_displacements = torch.Tensor(all_displacements)

    # Create dataset and dataloader objects
    dataset = DisplacementDataset(all_displacements, all_labels)
    dataloader = DataLoader(dataset, batch_size=props['batch_size'], shuffle=True)

    # Get validation data
    val_data = np.load(val_path)
    val_displacements = normalize_displacements(val_data['disp'], disp_stats)
    val_displacements = torch.Tensor(val_displacements)
    val_labels = torch.Tensor(val_data['labels'])
    val_labels = val_labels / np.array([D_max_normalization, D_max_normalization, 1])

    val_dataset = DisplacementDataset(val_displacements, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), props['lr'], weight_decay=props.get('weight_decay', 0.01))
    scheduler = CosineAnnealingLR(optimizer, T_max=cycles, eta_min=1e-5)
    
    model.to(device)

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

            loss = props['loss_fn'](output, labels)

            batch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        
        scheduler.step()

        losses.append(np.mean(batch_loss))

        # Validation
        model.eval()
        with torch.no_grad():
            batch_vloss = []
            for displacements, _, labels in val_dataloader:
                labels = labels.to(device)                
                displacements = displacements.to(device)
                
                val_predictions = model(displacements)

                l = props['loss_fn'](val_predictions, labels)
                batch_vloss.append(l.item())
            vlosses.append(np.mean(batch_vloss))

        print(f"Epoch {epoch+1}/{cycles} | Train Loss: {losses[-1]:.4f} | Val Loss: {vlosses[-1]:.4f}")
    
    return disp_stats

def train_multi_state(model: MultiStateModel, N: int, T: int, props: dict, image_props: dict, device: torch.device, cycles: int, losses: list, vlosses: list, val_path: str, binding: bool=False, is_baseline: bool=False):
    """
    Training for multi-state input

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
    D_max_normalization = image_props['D_max_norm']

    # Create training data
    all_videos, all_displacements, all_labels = create_multi_state_dataset_w_features(N, T, image_props, binding=binding) 
    disp_stats = compute_displacement_statistics(all_displacements)
    all_displacements = normalize_displacements(all_displacements, disp_stats)

    # Get validation data
    val_data = np.load(val_path)
    val_videos = torch.Tensor(val_data['vids'])
    val_displacements = normalize_displacements(val_data['disp'], disp_stats)
    val_displacements = torch.Tensor(val_displacements)
    val_labels = torch.Tensor(val_data['labels'])
    val_labels = val_labels / np.array([D_max_normalization, D_max_normalization, 1])

    val_dataset = VideoMotionDataset(val_videos, val_displacements, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    
    # Normalize labels for better optimization
    all_labels = all_labels / np.array([D_max_normalization, D_max_normalization, 1])
    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), props['lr'], weight_decay=props.get('weight_decay', 0.01))
    scheduler = CosineAnnealingLR(optimizer, T_max=cycles, eta_min=1e-5)

    model.to(device)

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

            if not is_baseline:
                output = model(videos, displacements)
            else:
                output = model(videos)
                
            loss = props['loss_fn'](output, labels)

            batch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()

        losses.append(np.mean(batch_loss))

        # Validation
        model.eval()
        with torch.no_grad():
            batch_vloss = []
            for videos, displacements, labels in val_dataloader:
                videos = videos.to(device)
                displacements = displacements.to(device)
                labels = labels.to(device)                

                if not is_baseline:
                    val_predictions = model(videos, displacements)
                else:
                    val_predictions = model(videos)

                l = props['loss_fn'](val_predictions, labels)
                batch_vloss.append(l.item())
            vlosses.append(np.mean(batch_vloss))

        print(f"Epoch {epoch+1}/{cycles} | Train Loss: {losses[-1]:.4f} | Val Loss: {vlosses[-1]:.4f}")
    
    return disp_stats