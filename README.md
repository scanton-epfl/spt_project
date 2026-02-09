# Estimating Diffusion Tensors from Single Particle Tracking Data
This repository contains the necessary tools for training and evaluating models built for 3 different cases of diffusion:
- Anisotropic diffusion
- Two-state binding/unbinding
- Two-state isotropic diffusion

## Table of Contents
- [Environment](#)
- [Training](#training)
- [Evaluation](#evaluation)
- [Repository Layout](#repository-layout)
    - [helpers](#helpers)
    - [experiments](#experiments)
    - [main.py](#mainpy)
    - [settings.py](#settingspy)
- [Contact Information](#contact)

# Environment
All codes were run and developed on a Linux machine. To reproduce results, you can create the necessary conda environment via `conda env create -f environment.yml`. It is also necessary to download the correct version of PyTorch for the device being used. Visit [PyTorch installation](https://pytorch.org/get-started/locally/) for more information.

# Training
To train one of our main models (multimodal cross transformer or two-state verisions), we use the `main.py` script. This script allows us to control the type of model and settings used for training such a model. This section will provide a tutorial for users on how to train a model via interaction with `main.py` and `settings.py`. Note: Inside `main.py` there rows of '#' signifying the section that can/should be altered by the user. Code beyond this section shouldn't be altered unless further work conducted. 

## 1. Model Selection
The first step is selecting the model you wish to train. There are three options:
- Image-only Transformer `DiffusionTensorRegModelBase`
- Multimodal, Single-state Cross Transformer `CrossAttentionModel`
- Multistate Cross Transformer `MultiStateModel`

The first option is a single mode baseline implemented to compare our proposed multimodal model with. The second option is used for single state predictions, where the model predicts a single diffusion tensor for the whole input sequence. The third option is used for multi-state predictions, where the model predicts a diffusion tensor for each frame in the input video sequence. Both cross transformer models are given as input a sequence of images and a sequence of displacements between frames. 

Inside `main.py`, there are the variables `multistate` and `multimodal` which are used to control which type of model to train. By setting `multistate=True`, the `MultiStateModel` will be trained. This option will trigger the generation of a two-state diffusion dataset for training. `MultiStateModel` can be used to train two different models: one on two-state isotropic diffusion data and one on two-state binding/unbinding data. To control which type of data to train on, you must set the `is_isotropic` variable to `True` for isotropic diffusive states and `False` otherwise.

If you want single state data, you must set `multistate` to `False`. By setting `multimodal=True`, the `CrossAttentionModel` will be trained. This option will trigger the generation of a multimodal dataset. If you want to train an image-only model, set `multistate` and `multimodal` to `False`. This will result in training a `DiffusionTensorRegModelBase` model. 

## 2. Model Parameters
Once a model is selected, it is necessary to control the model size and training parameters. These values are adjusted through the dictionary `model_props`. Here is a simple overview of each option. 
- `lr`: Learning rate for AdamW optimizer. All models were trained with the value `1e-4`.
- `embed_dim`: Embedding dimension for encoded images and displacements, as well as internally in transformers. Set to `64` for all models.
- `num_heads`: Number of attention heads for the attention mechanism inside each transformer block. Set to `4` for all models.
- `hidden_dim`: Embedding dimension inside hidden layers of MLPs. Set to `128` for all models.
- `num_layers`: Number of transformer blocks for each transformer model. Set to `6` for all models.
- `dropout`: Dropout probablity used throughout the model architecture. Each model was trained with values found through experimentation.
- `batch_size`: Batch size for training. Set to `16` due to memory limitations. Larger value (e.g. 32) might lead to more stable training.
- `loss_fn`: Loss function to use. No need to change this unless a new loss function is developed.
- `weight_decay`: Weight decay for AdamW optimizer. Set to `0.01` for all models.
- `use_pos_embed`: Specifies whether to use sinusoidal positional embeddings or not. Set to `False` for all models (we use rotary embeddings instead).
- `use_segment_embed`: Specifies whether to train embeddings unique to each modality. Helps model differentiate between modalities. Set to `True` for all models. This option is ignored for single modality training.

## 3. Simulation Settings
Each training begins by generating a training dataset. You can control the settings of the simulation to generate the desired dataset. Some settings are set within `main.py` and others are set through `settings.py`. 

Within `main.py`, you can control the size of the dataset to generate via the variables:
- `N`: Number of particles to simulate
- `nPosPerFrame`: Number of timesteps within a simulation to use for generating a frame. The simulation first generates trajectories then simulates the imaging process. This involves creating a sequence of frames from the trajectory. This parameter controls how many positions in the trajectory are used to generate one frame. This variable is directly linked to the frame rate of the imaging system. For a fixed timestep in the trajectory simulations, a larger frame rate corresponds to fewer positions per frame and vice versa. Using 10 positions per frame and a timestep of 1 ms corresponds to using a 100 Hz frame rate.
- `nFrames`: Number of frames to generate per trajectory

Additionally, you utilize the `image_props` variable to set the parameters used for generating images and videos. `settings.py` contains three pre-defined dictionaries that can be used. Each of these dictionaries correspond to one of our models:
- `BINDING_IMAGE_PROPS`: Properties to use for the two-state binding/unbinding dataset/model.
- `ISOTROPIC_PROPS`: Properties to use for the two-state isotropic diffusion dataset/model.
- `SINGLE_STATE_PROPS`: Properties to use for the single-state dataset/model. 

If desired, you can create new imaging settings by defining a new property dictionary within `settings.py` and updating `image_props` to dictionary name. Otherwise, you can just update `image_props` to one of pre-defined values.

Each image properties dictionary has the following fields:
- `n_pos_per_frame`: Number of positions per frame generated
- `frames`: Number of frames to generate
- `particle_intensity`: Experimental values for mean and standard deviation of particle intensity
- `NA`: Numerical aperture
- `wavelength`: Wavelength in meters
- `resolution`: Pixel resolution in meters
- `output_size`: Size of frame to generate in pixels (square frame)
- `upsampling_factor`: Upsampling factor used during frame generation to first generate a higher resolution frame before downsampling to desired output size
- `background_intensity`: Experimental values for mean and standard deviation of background noise intensity
- `poisson_noise`: Expected number of events occurring in a fixed-time interval, must be >= 0. We use `100` for all simulations. Can set to `-1` to request no Poisson noise. 
- `gaussian_noise`: Boolian specifying whether to add background Gaussian noise
- `D_min`: Minimum diffusion coefficient to sample in $\frac{\mu m^2}{s}$
- `D_max`: Maximum diffusion coefficient to sample in $\frac{\mu m^2}{s}$
- `angle_max`: Maximum angle to sample in radians
- `D_max_norm`: Scaling factor for diffusion coefficients (set to `D_max`)


## 4. Miscellaneous Settings
Additionally you control three other variables:
- `save_results`: Boolian specifying whether to save the model weights or not. Weights are saved to a predefined path depending on the training settings. The naming convention takes the following form:
```python
f'results/{model_name}_{N}_{epochs}_{model_props["embed_dim"]}_{model_props["lr"]}_{model_props["dropout"]}.pt'
```
where `model_name` is set based on the type of model being trained. This naming convention allows you to distinguish different trained models.
- `val_path`: Path to the validation data for use during training. A set of pre-defined ones are available for each model. New validation sets can be created through the `create_data.ipynb` notebook. 
- `epochs`: Number of epochs to train. 

## 5. Running the script
Once you have selected the model, model parameters, and simulation parameters, you can start training by running the script.

```bash
python main.py
```
Updates will be printed in the terminal at each epoch. At the end of training, the model will be evaluated on the validation set and results will be printed. The training and validation loss across epochs will be plotted and saved in `training.png`.

# Evaluation
To evaluate a trained model, you can use one of two notebooks in the `experiments` directory. For single state models, `DiffusionTensorRegModelBase` and `CrossAttentionModel`, you can evaluate them using the `evaluation.ipynb` notebook. For `MultiStateModel` models, you can evaluate them using the `multistate.ipynb` notebook. 

## Single-state Models
Within `evaluation.ipynb`, there is a section labeled `Single state predictions`. Within this section, there are subsections named `Single mode` and `Multi-modal`. For either section the procedure is the same, 3 cells in the section need to be run:

1. Update `model_props` to match your training settings and instantiate the model.  
2. Update `model_path` to match the path where model weights were saved during training. 
3. Run the third cell to evaluate on a newly generated test set (can update the loop to average results over multiple evaluations).

## Multi-state Models
Within `multistate.ipynb`, there is a section for evaluating each version of `MultiStateModel`: `Multi-state: Isotropic` and `Multi-state: Binding`. The procedure is for evaluation is identical to the previous cases:

1. Update `model_props` to match your training settings and instantiate the model.  
2. Update `model_path` to match the path where model weights were saved during training. 
3. Run the third cell to evaluate on a newly generated test set (can update the loop to average results over multiple evaluations).

For the two-state models, we can additionally evaluate how well our model captures the point when the transition of states occurs. After loading the weights for one of the `MultiStateModel` models, you can run all the cells under the section `Changepoint Detection` to evaluate and visualize the performance for this metric.

# Repository Layout
The repository has the following structure:

<pre>
├── helpers/
│   ├── simulation.py 
│   ├── models.py 
│   ├── training.py
│   ├── eval.py 
│   ├── helpersMSD.py
│   ├── create_data.ipynb 
│   └── image_generation.ipynb 
├── experiments/ 
│   ├── baselines.ipynb
│   ├── evaluation.ipynb
│   └── multistate.ipynb
├── data/
│   ├── validation_data_11.npz
│   ├── multi_state_binding_11.npz
│   └── multi_state_iso_11.npz
├── results/
│   ├── Model weights
├── main.py
└── settings.py
</pre>

Below is a small description for each file. Function-level documentation can be found in the respective files.

## helpers
The **helpers** directory contains useful python files containing model implementations, simulation code, training/evaluation codes, and visualization tools. 

### simulation.py
This python file contains all functions related to simulating trajectories and video generation. The functions used for creating datasets comes from this file. Inside we support simulations of anisotropic diffusion, isotropic diffusion, two-state isotropic diffusion, and two-state binding/unbinding. 

### models.py
This python file contains all functions for building our models. This includes PyTorch model implementations, PyTorch dataset classes, loss functions, and positional embeddings. Notably, we have implementations for our cross transformer `CrossAttentionModel`, multistate version `MultiStateModel`, and baselines `Pix2D` and `LSTM`. 

### training.py
This python file contains a training script for each type of model trained: single-mode models, multimodal models, and multistate models.

### eval.py
This python file contains functions for evaluation on different types of models. 

### helpersMSD.py
This python file contains functions for using mean squared displacement (MSD) to infer diffusion tensors. This acts as a baseline method.

### create_data.ipynb
This python notebooks contains single cell calls for creating validation sets for single state models and two-state models. Additionally, there are useful visualizations available for the created data. 

### image_generation.ipynb
This python notebook looks to visualize data samples under varying conditions of noise and PSF size.

## experiments
The **experiments** directory contains a few python notebooks for testing models. 

### baselines.ipynb
This notebook contains testing for each baseline model as well as testing on a sliding window approach for the multistate case of Pix2D. 

### evaluation.ipynb
This notebook contains testing for our proposed single state models. Additionally, we do experiments with varying noise levels and PSF sizes to study how capable the model.

### multistate.ipynb
This notebook contains testing for our proposed multistate models for the isotropic diffusion case and binding/unbinding case. 

## main.py
This python file is the main script for training models. Users will mainly interact with this file for training. 

## settings.py
This python file allows a user to define a set of parameters for the model to use for training/evaluation. Mainly you can control the imaging and simulation parameters. 


# Contact Information
This work was conducted as a master's semester project worth 8 ECTS. Questions regarding the code can be directed towards Simon Anton, simon.anton@epfl.ch. 
