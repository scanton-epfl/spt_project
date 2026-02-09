"""
File containing functions for evaluation on different types of models / data
"""
from types import LambdaType
import torch
import numpy as np
from helpers.simulation import normalize_displacements
from helpers.models import VideoMotionDataset, VideoDataset
from torch.utils.data import DataLoader

def convert_and_get_tensors(all_videos: np.ndarray, all_displacements: np.ndarray | None, og_labels: np.ndarray, disp_stats: tuple | None, D_max_normalization: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize data and create tensors
    
    Args:
        all_videos: np.ndarray (N, num_frames, H, W)
            Simulated video data
        all_displacements: np.ndarray (N, num_frames-1, 2) | None
            Displacements between frame pixel centroids for each video
            If None, displacements aren't used
        og_labels: np.ndarray (N, 3) or (N, num_frames, 3)
            Labels for each video or each frame for pointwise predictions (multi-state model) 
        disp_stats: tuple | None
            Displacement statistics from the training set
            If None, displacements aren't used
        D_max_normalization: float
            Number to divide diffusion coefficients by for scaling
    Returns:
        all_videos: torch.Tensor (N, num_frames, H, W)
        all_displacements: torch.Tensor (N, num_frames-1, 2) | None
        all_labels: torch.Tensor (N, 3) or (N, num_frames, 3)
    """
    all_videos = torch.Tensor(all_videos)
    if all_displacements is not None:
        all_displacements = normalize_displacements(all_displacements, disp_stats)
        all_displacements = torch.Tensor(all_displacements)

    all_labels = og_labels / np.array([D_max_normalization, D_max_normalization, 1])
    all_labels = torch.Tensor(all_labels)
    
    return all_videos, all_displacements, all_labels

def get_dataloader(all_videos: np.ndarray, all_displacements: np.ndarray | None, og_labels: np.ndarray, disp_stats: dict | None, D_max_normalization: float) -> DataLoader:
    """
    Helper function for creating a DataLoader
    
    Args:
        all_videos: np.ndarray (N, num_frames, H, W)
            Simulated video data
        all_displacements: np.ndarray (N, num_frames-1, 2) | None
            Displacements between frame pixel centroids for each video
            If None, displacements aren't used
        og_labels: np.ndarray (N, 3) or (N, num_frames, 3)
            Labels for each video or each frame for pointwise predictions (multi-state model) 
        disp_stats: dict | None
            Displacement statistics from the training set
            If None, displacements aren't used
        D_max_normalization: float
            Scaling for diffusion coefficients
    Returns:
        dataloader: DataLoader
            Torch DataLoader object for batching data
    """
    # Get tensors and normalize displacements
    all_videos, all_displacements, all_labels = convert_and_get_tensors(all_videos, all_displacements, og_labels, disp_stats, D_max_normalization)

    # Create dataset and dataloader
    if all_displacements is not None:
        dataset = VideoMotionDataset(all_videos, all_displacements, all_labels)
    else:
        dataset = VideoDataset(all_videos, all_labels)
        
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    return dataloader
    
def single_state_eval(all_videos: np.ndarray, all_displacements: np.ndarray | None, og_labels: np.ndarray, disp_stats: tuple | None, model, loss_fn: LambdaType, D_max_normalization: float, device: torch.device) -> np.float32:
    """
    Evaluation for single state predictions
    
    Args:
        all_videos: np.ndarray (N, num_frames, H, W)
            Simulated video data
        all_displacements: np.ndarray (N, num_frames-1, 2) | None
            Displacements between frame pixel centroids for each video
            If None, displacements aren't used
        og_labels: np.ndarray (N, 3) or (N, num_frames, 3)
            Labels for each video or each frame for pointwise predictions (multi-state model) 
        disp_stats: tuple | None
            Displacement statistics from the training set
            If None, displacements aren't used
        model: Any single state model object (e.g. CrossAttentionModel or Pix2D)
            Model to use for evaluation
        loss_fn: LambdaType
            Loss function
        D_max_normalization: float
            Scaling for diffusion coefficients
        device: torch.device
            Device where data and model is located
    Returns:
        avg_loss: np.float32
            Average loss from evaluation
    """
    # Get dataloader from test data
    dataloader = get_dataloader(all_videos, all_displacements, og_labels, disp_stats, D_max_normalization)

    model.eval()
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

            val_loss = loss_fn(val_predictions, labels)
            loss.append(val_loss.item()) # batch loss

            angle_pred = 0.5 * torch.atan2(val_predictions[...,2], val_predictions[...,3])
            
            mae = (val_predictions[...,:-2] - labels[...,:-1]).abs().mean(axis=((0,1) if labels.dim()==3 else 0))

            # cos(alpha)*cos(beta) + sin(alpha)*sin(beta) = cos(alpha - beta) -> [-1, 1]
            angle_sim = torch.abs(torch.cos(angle_pred - labels[...,-1])).mean().unsqueeze(-1)
            errors.append(torch.cat([
                    mae.cpu() * D_max_normalization,
                    angle_sim.cpu()
                ], axis=0))
            
        # Compute average across all labels
        avg_loss = np.mean(loss)
        avg_error = np.mean(errors, axis=0)
        print(f"Average loss across validation set: {avg_loss}")
        print(f"Average MAE of diffusion coefficients and angle simularity across validation set: {avg_error}")
        print(50*'-')

    return avg_loss

def multi_state_eval_isotropic(all_videos: np.ndarray, all_displacements: np.ndarray | None, og_labels: np.ndarray, disp_stats: tuple | None, model, loss_fn: LambdaType, D_max_normalization: float, device: torch.device) -> np.float32:
    """"
    Evaluation for a multi-state model with isotropic states

    Args:
        all_videos: np.ndarray (N, num_frames, H, W)
            Simulated video data
        all_displacements: np.ndarray (N, num_frames-1, 2) | None
            Displacements between frame pixel centroids for each video
            If None, displacements aren't used
        og_labels: np.ndarray (N, 3) or (N, num_frames, 3)
            Labels for each video or each frame for pointwise predictions (multi-state model) 
        disp_stats: tuple | None
            Displacement statistics from the training set
            If None, displacements aren't used
        model: Any single state model object (e.g. CrossAttentionModel or Pix2D)
            Model to use for evaluation
        loss_fn: LambdaType
            Loss function
        D_max_normalization: float
            Scaling for diffusion coefficients
        device: torch.device
            Device where data and model is located
    Returns:
        avg_loss: np.float32
            Average loss from evaluation
    """
    # Get dataloader from test data
    dataloader = get_dataloader(all_videos, all_displacements, og_labels, disp_stats, D_max_normalization)

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

            val_loss = loss_fn(val_predictions, labels)
            loss.append(val_loss.item()) # batch loss

            mae = (val_predictions[...,:2] - labels[...,:-1]).abs().mean(axis=((0,1) if labels.dim()==3 else 0))

            errors.append(mae.cpu() * D_max_normalization)
            
        # Compute average across all labels
        avg_loss = np.mean(loss)
        avg_error = np.mean(errors, axis=0)
        print(f"Average loss across validation set: {avg_loss}")
        print(f"Average MAE of diffusion coefficients and angle simularity across validation set: {avg_error}")
        print(50*'-')

    return avg_loss

def multi_state_eval(all_videos: np.ndarray, all_displacements: np.ndarray | None, og_labels: np.ndarray, disp_stats: tuple | None, model, loss_fn: LambdaType, D_max_normalization: float, device: torch.device):
    """"
    Evaluation for a multi-state model with binding/unbinding (functional the same as single_state_eval -> one could be deleted or kept separate in case a difference arises later)

    Args:
        all_videos: np.ndarray (N, num_frames, H, W)
            Simulated video data
        all_displacements: np.ndarray (N, num_frames-1, 2) | None
            Displacements between frame pixel centroids for each video
            If None, displacements aren't used
        og_labels: np.ndarray (N, 3) or (N, num_frames, 3)
            Labels for each video or each frame for pointwise predictions (multi-state model) 
        disp_stats: tuple | None
            Displacement statistics from the training set
            If None, displacements aren't used
        model: Any single state model object (e.g. CrossAttentionModel or Pix2D)
            Model to use for evaluation
        loss_fn: LambdaType
            Loss function
        D_max_normalization: float
            Scaling for diffusion coefficients
        device: torch.device
            Device where data and model is located
    Returns:
        avg_loss: np.float32
            Average loss from evaluation
    """
    # Get dataloader from test data
    dataloader = get_dataloader(all_videos, all_displacements, og_labels, disp_stats, D_max_normalization)

    model.eval()
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

            val_loss = loss_fn(val_predictions, labels)
            loss.append(val_loss.item()) # batch loss
            
            angle_pred = 0.5 * torch.atan2(val_predictions[...,2], val_predictions[...,3])
            
            mae = (val_predictions[...,:2] - labels[...,:-1]).abs().mean(axis=((0,1) if labels.dim()==3 else 0))

            # cos(alpha)*cos(beta) + sin(alpha)*sin(beta) = cos(alpha - beta) -> [-1, 1]
            angle_sim = torch.abs(torch.cos(angle_pred - labels[...,-1])).mean().unsqueeze(-1)
            errors.append(torch.cat([
                    mae.cpu() * D_max_normalization,
                    angle_sim.cpu()
                ], axis=0))
            
        # Compute average across all labels
        avg_loss = np.mean(loss)
        avg_error = np.mean(errors, axis=0)
        print(f"Average loss across validation set: {avg_loss}")
        print(f"Average MAE of diffusion coefficients and angle simularity across validation set: {avg_error}")
        print(50*'-')

    return avg_loss

def single_state_sliding_window_eval(all_videos: np.ndarray, all_displacements: np.ndarray | None, og_labels: np.ndarray, disp_stats: tuple | None, model, loss_fn: LambdaType, D_max_normalization: float, device: torch.device, window_size: int=5) -> np.float32:
    """
    Evaluation using single state model for multi-state predictions - binding (or in general a model that predicts the angle)
    
    Args:
        all_videos: np.ndarray (N, num_frames, H, W)
            Simulated video data
        all_displacements: np.ndarray (N, num_frames-1, 2) | None
            Displacements between frame pixel centroids for each video
            If None, displacements aren't used
        og_labels: np.ndarray (N, 3) or (N, num_frames, 3)
            Labels for each video or each frame for pointwise predictions (multi-state model) 
        disp_stats: tuple | None
            Displacement statistics from the training set
            If None, displacements aren't used
        model: Any single state model object (e.g. CrossAttentionModel or Pix2D)
            Model to use for evaluation
        loss_fn: LambdaType
            Loss function
        D_max_normalization: float
            Scaling for diffusion coefficients
        device: torch.device
            Device where data and model is located
        window_size: int
            Number of frames to use for window-based prediction
    Returns:
        avg_loss: np.float32
            Average loss from evaluation
    """
    # Get dataloader from test data
    dataloader = get_dataloader(all_videos, all_displacements, og_labels, disp_stats, D_max_normalization)

    model.eval()
    with torch.no_grad():
        loss = []
        errors = []
        for videos, displacements, labels in dataloader:
            videos = videos.to(device)
            labels = labels.to(device)
            
            batch_loss = []
            batch_errors = []
            # Loop over non-overlapping windows
            for i in range(0, videos.shape[1], window_size):
                end_index = i + window_size if i + window_size < videos.shape[1] else videos.shape[1]
                l = labels[:, i:end_index].mean(dim=1) # take average label over window
                
                if displacements.numel() != 0:
                    d = displacements[:, i:end_index].to(device)
                    val_predictions = model(videos[:,i:end_index], d)
                else:
                    val_predictions = model(videos[:,i:end_index])

                val_loss = loss_fn(val_predictions, l)
                batch_loss.append(val_loss.item()) # batch loss

                angle_pred = 0.5 * torch.atan2(val_predictions[...,2], val_predictions[...,3])
                
                mae = (val_predictions[...,:-2] - l[...,:-1]).abs().mean(axis=((0,1) if l.dim()==3 else 0))

                angle_sim = torch.abs(torch.cos(angle_pred - l[...,-1])).mean().unsqueeze(-1)
                
                batch_errors.append(torch.cat([
                    mae.cpu() * D_max_normalization,
                    angle_sim.cpu()
                ], axis=0))
                
            loss.append(np.mean(batch_loss))
            errors.append(np.mean(batch_errors, axis=0))
            
        # Compute average across all labels
        avg_loss = np.mean(loss)
        avg_error = np.mean(errors, axis=0)
        print(f"Average loss across validation set: {avg_loss}")
        print(f"Average MAE of diffusion coefficients and angle simularity across validation set: {avg_error}")
        print(50*'-')

    return avg_loss

def single_state_sliding_window_eval_iso(all_videos: np.ndarray, all_displacements: np.ndarray | None, og_labels: np.ndarray, disp_stats: tuple | None, model, loss_fn: LambdaType, D_max_normalization: float, device, window_size: int=5) -> np.float32:
    """
    Evaluation using single state model for multi-state predictions - isotropic
    
    Args:
        all_videos: np.ndarray (N, num_frames, H, W)
            Simulated video data
        all_displacements: np.ndarray (N, num_frames-1, 2) | None
            Displacements between frame pixel centroids for each video
            If None, displacements aren't used
        og_labels: np.ndarray (N, 3) or (N, num_frames, 3)
            Labels for each video or each frame for pointwise predictions (multi-state model) 
        disp_stats: tuple | None
            Displacement statistics from the training set
            If None, displacements aren't used
        model: Any single state model object (e.g. CrossAttentionModel or Pix2D)
            Model to use for evaluation
        loss_fn: LambdaType
            Loss function
        D_max_normalization: float
            Scaling for diffusion coefficients
        device: torch.device
            Device where data and model is located
        window_size: int
            Number of frames to use for window-based prediction
    Returns:
        avg_loss: np.float32
            Average loss from evaluation
    """
    # Get dataloader from test data
    dataloader = get_dataloader(all_videos, all_displacements, og_labels, disp_stats, D_max_normalization)

    model.eval()
    with torch.no_grad():
        loss = []
        errors = []
        for videos, displacements, labels in dataloader:
            videos = videos.to(device)
            labels = labels.to(device)
            
            batch_loss = []
            batch_errors = []
            # Loop over non-overlapping windows
            for i in range(0, videos.shape[1], window_size):
                end_index = i + window_size if i + window_size < videos.shape[1] else videos.shape[1]
                l = labels[:, i:end_index].mean(dim=1) # take average label over window
                
                if displacements.numel() != 0:
                    d = displacements[:, i:end_index].to(device)
                    val_predictions = model(videos[:,i:end_index], d)
                else:
                    val_predictions = model(videos[:,i:end_index])

                val_loss = loss_fn(val_predictions, l)
                batch_loss.append(val_loss.item()) # batch loss

                mae = (val_predictions[...,:-2] - l[...,:-1]).abs().mean(axis=((0,1) if l.dim()==3 else 0))

                batch_errors.append(torch.cat([
                    mae.cpu() * D_max_normalization,
                ], axis=0))
                
            loss.append(np.mean(batch_loss))
            errors.append(np.mean(batch_errors, axis=0))
            
        # Compute average across all labels
        avg_loss = np.mean(loss)
        avg_error = np.mean(errors, axis=0)
        print(f"Average loss across validation set: {avg_loss}")
        print(f"Average MAE of diffusion coefficients across validation set: {avg_error}")
        print(50*'-')

    return avg_loss