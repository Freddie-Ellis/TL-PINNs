# TL-PINNs: Transfer Learning with Physics-Informed Neural Networks

A third-year individual research project exploring the integration of **Transfer Learning (TL)** into **Physics-Informed Neural Networks (PINNs)** for fluid mechanics applications.

---

## Project Overview

This project investigates how pre-trained PINNs can be efficiently adapted to new flow regimes using transfer learning techniques. The models are trained to solve PDEs governing fluid flow (e.g., Navier-Stokes) using limited data, and retrained for new Reynolds numbers or boundary conditions.

---

## Requirements

Make sure the following dependencies and environments are met:

| Package       | Version   |
|---------------|-----------|
| TensorFlow    | 2.7.0     |
| NumPy         | 1.24.4    |
| SciPy         | 1.10.1    |
| Matplotlib    | 3.2.2     |
| CUDA Toolkit  | 11.2      |
| cuDNN         | 8.1       |

> Ensure that your GPU environment supports CUDA 11.2 and cuDNN 8.1 for optimal training performance.

---

## Workflow

### Training a Baseline PINN

1. **Preprocess Source Data** for training case A  
2. **Train PINN Model A** on source data  
3. **Predict and Plot Flow Fields** (e.g., \( u, v, p \))  
4. **Compute Errors**: MSE, relative L2 error, PDE residuals  
5. **Visualize** prediction quality through plots and animations

---

### Transfer Learning with PINNs

1. **Load Pretrained Model A**  
2. **Modify Hyperparameters** (e.g., learning rate, λ-weighting, data resolution)  
3. **Preprocess Target Data** for new case (e.g., different Reynolds number)  
4. **Retrain Model A** on the new data  
5. **Generate Predictions** and compare to ground truth  
6. **Evaluate and Visualize** performance (error metrics, FFTs, plots)

---

data/  
│  
├──cyl100/  
├──cyl150/  

TL-PINNs/  
│  
├── frames/             # Snapshots of flow fields for report writing  
├── models/             # Saved model checkpoints (.npz files)  
├── out/                # HPC out files  
├── pinn/               # This has the pre and post processing, the model and the FFT code  
├── plots/              # Plots   
├── temp/               # Temporary processed data storage, included in gitignore as files too large  
├── evaluate.py         # Script to compute errors and visualizations   
├── plotter.py          # Collocation points visualisation for report  
├── README.md           # This file  
├── Tl.py               # Transfer learning script  
├── training.py         # Base model training script  
