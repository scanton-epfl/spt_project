import torch
import numpy as np
from helpers.training import *
import matplotlib.pyplot as plt
from datetime import datetime
import os

def main():
    # Save model weights
    save_results = False

    # Define model run type
    multistate = True
    multimodal = True
    use_sum=True
    cross_attn_base = True
    cross_attn_complex = True
    hierarchy_model = False
    disp_model = False

    # Define model parameters
    model_props= {
        "lr": 1e-4,
        "embed_dim": 64,
        "num_heads": 4,
        "hidden_dim": 128,
        "num_layers": 6,
        "dropout": 0.2,
        "batch_size": 16,
        "loss_fn": mse_loss,
        "weight_decay": 0.01,
        "use_pos_embed": False,
        "use_segment_embed": True,
        "use_rotary": True
    }

    if model_props['loss_fn'] is mse_loss_coeff:
        output_size = 2
    else: 
        output_size = 4

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
        ], # Mean and standard deviation of the particle intensity
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
        "poisson_noise": 100, #  -1 for no noise,
        "gaussian_noise": True,
        "trajectory_unit" : 1200,
        "D_min": 6, # >= 1
        "D_max": 10,
        "angle_max": np.pi,
        "D_max_norm": 10, # factor to divide by for normalization
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
    if multistate:
        model = MultiStateModel(model_props['embed_dim'], model_props['num_heads'], model_props['hidden_dim'], 
                    model_props['num_layers'], model_props['dropout'], output_dim=output_size, use_pos_embed=model_props['use_pos_embed'],
                    use_segment_embed=model_props['use_segment_embed'], use_rotary=model_props['use_rotary'])
        print(f"Total params of {model.__class__.__name__}: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        train_multi_state(model, N, T, model_props, image_props, device, cycles, losses, vlosses)
    elif multimodal:
        if cross_attn_base:
            if cross_attn_complex:
                model = CrossAttentionModel(model_props['embed_dim'], model_props['num_heads'], model_props['hidden_dim'], 
                    model_props['num_layers'], model_props['dropout'], output_dim=output_size, use_pos_embed=model_props['use_pos_embed'],
                    use_segment_embed=model_props['use_segment_embed'], use_rotary=model_props['use_rotary'])
            else:
                model = CrossAttentionModelBase(model_props['embed_dim'], model_props['num_heads'], model_props['hidden_dim'], 
                    model_props['num_layers'], model_props['dropout'], output_dim=output_size, use_pos_embed=model_props['use_pos_embed'],
                    use_segment_embed=model_props['use_segment_embed'], use_rotary=model_props['use_rotary'])
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
    for file in [f for f in os.listdir('data') if f.startswith('multi_state')]:
        # Get validation data
        data = np.load('data/' + file)
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

                angle_pred = 0.5 * torch.atan2(val_predictions[...,2], val_predictions[...,3])
                if model_props['loss_fn'] is mse_loss or mse_loss_coeff:
                    #mse = ((val_predictions[...,:-2] - labels[...,:-1])**2).mean(axis=((0,1) if labels.dim()==3 else 0))
                    mae = (val_predictions[...,:-2] - labels[...,:-1]).abs().mean(axis=((0,1) if labels.dim()==3 else 0))
                else:
                    # mse = torch.stack([
                    #     (torch.exp(val_predictions[:,0]) - labels[:,0])**2, 
                    #     (torch.exp(val_predictions[:,1]) - labels[:,1])**2,
                    # ], axis=1).mean(axis=0)
                    mae = (torch.exp(val_predictions[...,:-2]) - labels[...,:-1]).abs().mean(axis=((0,1) if labels.dim()==3 else 0))

                #rmse_batch = torch.sqrt(mae)
                angle_sim = torch.abs(torch.cos(angle_pred - labels[...,-1])).mean().unsqueeze(-1)
                errors.append(torch.cat([
                        mae.cpu() * D_max_normalization,
                        angle_sim.cpu()
                    ], axis=0))

            # Compute average across all labels
            avg_loss = np.mean(loss)
            avg_error = np.mean(errors, axis=0)
            print(file[:-4])
            print(f"Average loss across validation set: {avg_loss}")
            print(f"Average MAE of diffusion coefficients and angle simularity across validation set: {avg_error}")
            print(50*'-')

    # Save model weights
    if save_results:
        date = datetime.now().strftime('%d_%m_%Y')
        torch.save(model.state_dict(), f'results/multi_state_model_{date}.pt')
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