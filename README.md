# MDWGAN
This is the source code for paper Multi-objective Generative Design of Three-Dimensional Material Structures. Notice that the default parameter is the parameter used in our training procedure.

# Dataset
Dataset is under './dataset/'.

# Pre-train the surrogate model
Quick start

  python train_surrogate.py
Otherwise

  python train_surrogate.py --lr [learning rate] --bsz [batch size] --epochs [number of epochs] --gamma [gamma for scheduler] --weight_decay [weight decay] --model_path [saved path for surrogate]

# Test the surrogate model

  python test_surrogate.py --model_path [saved path for test model] --output_path [saved path for output]
