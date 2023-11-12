# MDWGAN
This is the source code for paper Multi-objective Generative Design of Three-Dimensional Material Structures. Notice that the default parameter is the parameter used in our training procedure.

## Code Download
Download the code by

    git clone --recursive https://github.com/FurionChang/MDWGAN/tree/master.git

## Dataset
Dataset is under './dataset/'.

## Pre-train the surrogate model
**Quick start**

    python train_surrogate.py
 
**Otherwise**

    python train_surrogate.py --lr [learning rate] --bsz [batch size] --epochs [number of epochs] --gamma [gamma for scheduler] --weight_decay [weight decay] --model_path [saved path for surrogate]

## Test the surrogate model

    python test_surrogate.py --model_path [saved path for test model] --output_path [saved path for output]

## Train MDWGAN
**Quick start**

    python train_wgan.py

**Otherwise**

    python train_wgan.py --lr [learning rate] --bsz [batch size] --epochs [number of epochs] --G_channels [channels for the generator] --D_channels [channels for the discriminator] --alpha [multiplier for loss G2] --beta [multiplier for loss G3] --obj [goal for training] --G_path [saved path for generator] --D_path [saved path for discriminator] --dataset [type of training dataset (symm/random)] --surr_path [saved path for surrogate]

## Generate samples by MDWGAN

    python wgan_generate_sample.py --num [number of generated samples] --G_path [saved path for generator] --D_path [saved path for discriminator] --dataset [type of training dataset (symm/random)] --S_path [path for output samples]

## Deal with output samples
Please follow the instructions of sample_demo.ipynb. It provide functions for plotting the surrogate results for output and provide functions for plotting the structure of chosen samples.
