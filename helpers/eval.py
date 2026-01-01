import torch
import numpy as np
from helpers.simulation import normalize_displacements
from helpers.models import VideoMotionDataset, DataLoader, mse_similarity, VideoDataset

def convert_and_get_tensors(all_videos: np.ndarray, all_displacements: np.ndarray | None, og_labels: np.ndarray, disp_stats: dict | None, D_max_normalization: float):
    all_videos = torch.Tensor(all_videos)
    if all_displacements is not None:
        all_displacements = normalize_displacements(all_displacements, disp_stats)
        all_displacements = torch.Tensor(all_displacements)

    all_labels = og_labels / np.array([D_max_normalization, D_max_normalization, 1])
    all_labels = torch.Tensor(all_labels)
    
    return all_videos, all_displacements, all_labels

def single_state_eval(all_videos: np.ndarray, all_displacements: np.ndarray | None, og_labels: np.ndarray, model, disp_stats: dict | None, model_props: dict, image_props: dict, device) -> np.float32:
    """
    Evaluation for single state predictions
    """
    # Get tensors and normalize displacements
    D_max_normalization = image_props['D_max_norm']
    all_videos, all_displacements, all_labels = convert_and_get_tensors(all_videos, all_displacements, og_labels, disp_stats, D_max_normalization)

    # Create dataset and dataloader
    if all_displacements is not None:
        dataset = VideoMotionDataset(all_videos, all_displacements, all_labels)
    else:
        dataset = VideoDataset(all_videos, all_labels)
        
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.eval()
    with torch.no_grad():
        loss = []
        errors = []
        mse_sims = []
        for videos, displacements, labels in dataloader:
            videos = videos.to(device)
            labels = labels.to(device)
            
            if displacements.numel() != 0:
                displacements = displacements.to(device)
                val_predictions = model(videos, displacements)
            else:
                val_predictions = model(videos)

            val_loss = model_props['loss_fn'](val_predictions, labels)
            loss.append(val_loss.item()) # batch loss

            angle_pred = 0.5 * torch.atan2(val_predictions[...,2], val_predictions[...,3])
            
            sims = mse_similarity(val_predictions, labels)

            #mse = ((val_predictions[...,:-2] - labels[...,:-1])**2).mean(axis=((0,1) if labels.dim()==3 else 0))
            mae = (val_predictions[...,:-2] - labels[...,:-1]).abs().mean(axis=((0,1) if labels.dim()==3 else 0))

            angle_sim = torch.abs(torch.cos(angle_pred - labels[...,-1])).mean().unsqueeze(-1)
            errors.append(torch.cat([
                    mae.cpu() * D_max_normalization,
                    angle_sim.cpu()
                ], axis=0))
            mse_sims.append(sims.cpu())
            
        # Compute average across all labels
        avg_loss = np.mean(loss)
        avg_error = np.mean(errors, axis=0)
        avg_sim = np.mean(mse_sims)
        print(f"Average loss across validation set: {avg_loss}")
        print(f"Average MAE of diffusion coefficients and angle simularity across validation set: {avg_error}")
        print(f"Average L2-based similarity of diffusion tensors across validation set: {avg_sim}")
        print(50*'-')

    return avg_loss, avg_sim

def multi_state_eval_isotropic(all_videos: np.ndarray, all_displacements: np.ndarray | None, og_labels: np.ndarray, disp_stats: dict | None, model, model_props: dict, image_props: dict, device):
    """"
    Evaluation for a multi-state model with isotropic states
    """
    # Get tensors and normalize displacements
    D_max_normalization = image_props['D_max_norm']
    all_videos, all_displacements, all_labels = convert_and_get_tensors(all_videos, all_displacements, og_labels, disp_stats, D_max_normalization)

    # Create dataset and dataloader
    if all_displacements is not None:
        dataset = VideoMotionDataset(all_videos, all_displacements, all_labels)
    else:
        dataset = VideoDataset(all_videos, all_labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    with torch.no_grad():
        loss = []
        errors = []
        mse_sims = []
        for videos, displacements, labels in dataloader:
            videos = videos.to(device)
            labels = labels.to(device)
            
            if displacements.numel() != 0:
                displacements = displacements.to(device)
                val_predictions = model(videos, displacements)
            else:
                val_predictions = model(videos)

            val_loss = model_props['loss_fn'](val_predictions, labels)
            loss.append(val_loss.item()) # batch loss

            sims = mse_similarity(val_predictions, labels)

            mae = (val_predictions[...,:2] - labels[...,:-1]).abs().mean(axis=((0,1) if labels.dim()==3 else 0))

            errors.append(mae.cpu() * D_max_normalization)
            mse_sims.append(sims.cpu())
            
        # Compute average across all labels
        avg_loss = np.mean(loss)
        avg_error = np.mean(errors, axis=0)
        avg_sim = np.mean(mse_sims)
        print(f"Average loss across validation set: {avg_loss}")
        print(f"Average MAE of diffusion coefficients and angle simularity across validation set: {avg_error}")
        print(f"Average L2-based similarity of diffusion tensors across validation set: {avg_sim}")
        print(50*'-')

    return avg_loss, avg_sim

def multi_state_eval(all_videos: np.ndarray, all_displacements: np.ndarray | None, og_labels: np.ndarray, disp_stats: dict | None, model, model_props: dict, image_props: dict, device):
    """"
    Evaluation for a multi-state model with isotropic states
    """
    # Get tensors and normalize displacements
    D_max_normalization = image_props['D_max_norm']
    all_videos, all_displacements, all_labels = convert_and_get_tensors(all_videos, all_displacements, og_labels, disp_stats, D_max_normalization)

    # Create dataset and dataloader
    if all_displacements is not None:
        dataset = VideoMotionDataset(all_videos, all_displacements, all_labels)
    else:
        dataset = VideoDataset(all_videos, all_labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    with torch.no_grad():
        loss = []
        errors = []
        mse_sims = []
        for videos, displacements, labels in dataloader:
            videos = videos.to(device)
            labels = labels.to(device)
            
            if displacements.numel() != 0:
                displacements = displacements.to(device)
                val_predictions = model(videos, displacements)
            else:
                val_predictions = model(videos)

            val_loss = model_props['loss_fn'](val_predictions, labels)
            loss.append(val_loss.item()) # batch loss
            
            angle_pred = 0.5 * torch.atan2(val_predictions[...,2], val_predictions[...,3])
            
            sims = mse_similarity(val_predictions, labels)

            mae = (val_predictions[...,:2] - labels[...,:-1]).abs().mean(axis=((0,1) if labels.dim()==3 else 0))

            angle_sim = torch.abs(torch.cos(angle_pred - labels[...,-1])).mean().unsqueeze(-1)
            errors.append(torch.cat([
                    mae.cpu() * D_max_normalization,
                    angle_sim.cpu()
                ], axis=0))
            mse_sims.append(sims.cpu())  
                      
        # Compute average across all labels
        avg_loss = np.mean(loss)
        avg_error = np.mean(errors, axis=0)
        avg_sim = np.mean(mse_sims)
        print(f"Average loss across validation set: {avg_loss}")
        print(f"Average MAE of diffusion coefficients and angle simularity across validation set: {avg_error}")
        print(f"Average L2-based similarity of diffusion tensors across validation set: {avg_sim}")
        print(50*'-')

    return avg_loss, avg_sim