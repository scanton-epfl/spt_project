"""
Main script for training a multimodal "cross transformer", multi-state version of it, and an image only version

The following information should be updated for controlling the training setup:
    - Save weights:
        Set save_results = True if you want to save the weights of the training
    - Model type:
        Set multistate = True if you want to train a two-state multimodal model
        Set multimodal = True and multistate = False if you want to train a single-state multimodal model
        Set both options to False if you want to train an image only transformer model
    - Validation path:
        Set val_path to the path of your validation set
    - Model settings:
        Use model_props to adjust the model and training parameters. 
    - Simulation settings:
        Adjust number of particles to simulate, timesteps, nPosPerFrame, and image_props dictionary (inside settings.py)
    - Save path:
        At bottom of file adjust save path if saving weights
"""
import torch
import numpy as np
from helpers.training import *
from helpers.eval import *
import matplotlib.pyplot as plt
from settings import *

def main():
    ######################################################################################################
    # Training settings
    ######################################################################################################
    
    # Save model weights
    save_results = False

    # Define model run type
    multistate = False
    multimodal = True
    if multistate:
        is_isotropic = False # whether to do isotropic of binding case
      
    val_path = 'data/validation_data_11.npz'
    #val_path = 'data/multi_state_binding_11.npz'
    #val_path = 'data/multi_state_iso_11.npz'
    
    # Define model parameters
    model_props= {
        "lr": 1e-4,
        "embed_dim": 64,
        "num_heads": 4,
        "hidden_dim": 128,
        "num_layers": 6,
        "dropout": 0.05,
        "batch_size": 16,
        "loss_fn": models.mse_loss, # leave as is unless different loss function developped
        "weight_decay": 0.01,
        "use_pos_embed": False, # leave as False unless experimenting with sinusoidal embeddings
        "use_segment_embed": True, # leave as True (used inside multimodal and multistate versions, ignored for image-only version)
    }
    
    # Image parameters
    N = 1000 # number of particles to simulate
    nPosPerFrame = 10 
    nFrames = 30
    T = nFrames * nPosPerFrame

    image_props = BINDING_IMAGE_PROPS
    image_props["n_pos_per_frame"] = nPosPerFrame
    image_props["frames"] = nFrames

    # Number of epochs of training
    epochs = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    ######################################################################################################
    
    # Name of model being saved (ignore unless want to change naming scheme which is defined at bottom of file where weights are saved)
    if multistate:
        if is_isotropic:
            model_name = 'multi_state_iso'
        else:
            model_name = 'multi_state_binding'
    elif multimodal:
        model_name = 'model'
    else:
        model_name = 'image_only'
    
    # Settings altered for multistate
    if multistate:
        model_props['use_rotary'] = True
        if is_isotropic:
            model_props['loss_fn'] = models.mse_loss_coeff
    else:
        model_props['use_rotary'] = False

    # Alter output size for isotropic case
    if model_props['loss_fn'] is models.mse_loss_coeff:
        output_size = 2
    else: 
        output_size = 4

    np.random.seed(150)
    
    # Keep track of loss across training for plotting
    losses = []
    vlosses = []

    # Train model based on run type
    if multistate:
        model = models.MultiStateModel(model_props['embed_dim'], model_props['num_heads'], model_props['hidden_dim'], 
                    model_props['num_layers'], model_props['dropout'], output_dim=output_size, use_pos_embed=model_props['use_pos_embed'],
                    use_segment_embed=model_props['use_segment_embed'], use_rotary=model_props['use_rotary'])
        print(f"Total params of {model.__class__.__name__}: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        disp_stats = train_multi_state(model, N, T, model_props, image_props, device, epochs, losses, vlosses, val_path=val_path, binding=not is_isotropic)
    elif multimodal:
        model = models.CrossAttentionModel(model_props['embed_dim'], model_props['num_heads'], model_props['hidden_dim'], 
            model_props['num_layers'], model_props['dropout'], output_dim=output_size, use_pos_embed=model_props['use_pos_embed'],
            use_segment_embed=model_props['use_segment_embed'], use_rotary=model_props['use_rotary'])
        print(f"Total params of {model.__class__.__name__}: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        disp_stats = train_multi_modal(model, N, T, model_props, image_props, device, epochs, losses, vlosses, val_path=val_path)
    else:
        model = models.DiffusionTensorRegModelBase(model_props['embed_dim'], model_props['num_heads'], model_props['hidden_dim'], 
                model_props['num_layers'], model_props['dropout'])
        print(f"Total params of {model.__class__.__name__}: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        train_single_mode(model, N, T, model_props, image_props, device, epochs, losses, vlosses, val_path=val_path)

    # Evaluation
    val_data = np.load(val_path)
    if multistate:
        if image_props == BINDING_IMAGE_PROPS:
            multi_state_eval(val_data['vids'], val_data['disp'], val_data['labels'], disp_stats, model, model_props['loss_fn'], image_props['D_max_norm'], device)
        else:
            multi_state_eval_isotropic(val_data['vids'], val_data['disp'], val_data['labels'], disp_stats, model, model_props['loss_fn'], image_props['D_max_norm'], device)
    elif multimodal:
        single_state_eval(val_data['vids'], val_data['disp'], val_data['labels'], disp_stats, model, model_props['loss_fn'], image_props['D_max_norm'], device)
    else:
        single_state_eval(val_data['vids'], None, val_data['labels'], None, model, model_props['loss_fn'], image_props['D_max_norm'], device)
        
    # Save model weights
    if save_results:
        save_path = f'results/{model_name}_{N}_{epochs}_{model_props["embed_dim"]}_{model_props["lr"]}_{model_props["dropout"]}.pt'
        if multimodal:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "norm_stats": disp_stats
            }
        else:
            checkpoint = {
                "model_state_dict": model.state_dict()
            }
        torch.save(checkpoint, save_path)
        print(f"Saved model to path: {save_path}")

    # Plot loss over time during training
    fig, ax = plt.subplots()
    ax.plot(losses, 'r', label='Training')
    ax.plot(vlosses, 'b*', label='Validation')
    ax.set_title('Loss across training epochs')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend()
    fig.tight_layout()
    plt.savefig('training.png')

if __name__ == '__main__':
   main()