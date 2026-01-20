import torch
import numpy as np
from helpers.training import *
import matplotlib.pyplot as plt
import os
from settings import *

def main():
    # Save model weights
    save_results = True

    # Define model run type
    multistate = True
    multimodal = True
    disp_model = False
    
    # Options for multimodal
    if multimodal:
        cross_attn_base = True
        cross_attn_complex = True
        hierarchy_model = False
        use_sum=True
        
    #val_name = 'validation_data_11.npz'
    val_name = 'multi_state_binding_11.npz'

    # Define model parameters
    model_props= {
        "lr": 1e-4,
        "embed_dim": 64,
        "num_heads": 4,
        "hidden_dim": 128,
        "num_layers": 6,
        "dropout": 0.05,
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
    N = 275000 # number of particles to simulate
    nPosPerFrame = 10 
    nFrames = 30
    T = nFrames * nPosPerFrame

    image_props = BINDING_IMAGE_PROPS

    # Number of cycles of generating training data and training
    cycles = 35
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
        disp_stats = train_multi_state(model, N, T, model_props, image_props, device, cycles, losses, vlosses, val_path='data/'+val_name, binding=True)
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
        disp_stats = train_multi_modal(model, N, T, model_props, image_props, device, cycles, losses, vlosses, val_path='data/'+val_name)
    else:
        if not disp_model:
            model = DiffusionTensorRegModelBase(model_props['embed_dim'], model_props['num_heads'], model_props['hidden_dim'], 
                    model_props['num_layers'], model_props['dropout'])
            print(f"Total params of {model.__class__.__name__}: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
            train_single_mode(model, N, T, model_props, image_props, device, cycles, losses, vlosses, val_path='data/'+val_name)
        else:
            model = DisplacementBasedModel(model_props['embed_dim'], model_props['num_heads'], model_props['hidden_dim'], 
                    model_props['num_layers'], model_props['dropout'])
            print(f"Total params of {model.__class__.__name__}: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
            disp_stats = train_disp_mode(model, N, T, model_props, image_props, device, cycles, losses, vlosses, val_path='data/'+val_name)

    # Evaluation
    model.eval()
    for file in [f for f in os.listdir('data') if f.startswith(val_name)]:
        # Get validation data
        data = np.load('data/' + file)
        videos = torch.Tensor(data['vids'])
        if multimodal or disp_model:
            displacements = normalize_displacements(data['disp'], disp_stats)
            displacements = torch.Tensor(displacements)
        labels = torch.Tensor(data['labels'])
        
        D_max_normalization = image_props['D_max_norm']
        labels = labels / np.array([D_max_normalization, D_max_normalization, 1])

        # Create dataset and dataloader
        if multimodal or multistate:
            dataset = VideoMotionDataset(videos, displacements, labels)
        else:
            if not disp_model:
                dataset = VideoDataset(videos, labels)
            else:
                dataset = DisplacementDataset(displacements, labels)

        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

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

                val_loss = model_props['loss_fn'](val_predictions, labels)
                loss.append(val_loss.item()) # batch loss

                angle_pred = 0.5 * torch.atan2(val_predictions[...,2], val_predictions[...,3])
                
                abs_error = (val_predictions[...,:2] - labels[...,:-1]).abs()
                mae = abs_error.mean(axis=((0,1) if labels.dim()==3 else 0))

                # cos(alpha)*cos(beta) + sin(alpha)*sin(beta) = cos(alpha - beta) -> [-1, 1]
                angle_sim = torch.abs(torch.cos(angle_pred - labels[...,-1])).mean().unsqueeze(-1) 
                errors.append(torch.cat([
                        mae.cpu() * D_max_normalization,
                        angle_sim.cpu()
                    ], axis=0))
                #errors.append(mae.cpu() * D_max_normalization)
                
            # Compute average across all labels
            avg_loss = np.mean(loss)
            avg_error = np.mean(errors, axis=0)
            print(file[:-4])
            print(f"Average loss across validation set: {avg_loss}")
            print(f"Average MAE of diffusion coefficients and angle simularity across validation set: {avg_error}")
            print(50*'-')

    # Save model weights
    if save_results:
        path = f'results_final/multi_state_binding_{N}_{cycles}_{model_props["embed_dim"]}_{model_props["lr"]}_{model_props["dropout"]}.pt'
        if multimodal or disp_model:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "norm_stats": disp_stats
            }
        else:
            checkpoint = {
                "model_state_dict": model.state_dict()
            }
        torch.save(checkpoint, path)
        print(f"Saved model to path: {path}")

    # Plot loss over time during training
    fig, ax = plt.subplots()
    ax.plot(losses, 'r', label='Training')
    ax.plot(vlosses, 'b*', label='Validation')
    ax.set_title('Loss across training cycles')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Cycle')
    ax.legend()
    fig.tight_layout()
    plt.savefig('training.png')

if __name__ == '__main__':
   main()