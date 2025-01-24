# TL-PINNs
Third year individual project on Transfer Learning (TL) integrated PINNs for fluid mechanics.
I'm currently connecting this repository to MobaXterm for HPC usage.

Workflow is as follows:

Make change in VS -> Push to GitHub -> pull on Moba using 'git pull origin main' from within cd

Make change in Moba -> Push to GitHub ('make the change' -> git status -> git add . -> git commit -m "update message" -> 
git push origin main) -> Pull on local machine in GitHub Desktop

# Requirements

TensorFlow == 2.5.0  
NumPy == 1.19.2  
Cuda == 11.2  
Cudnn == 8.1

# Naming Convention

A_B_C_D_E

A - TL or Mod (Transfer Learning or Model)  
B - @ + 'No It' (No Iterations (e+3) for standard training)  
C - TL + 'No It' (No Iterations (e+3) for Transfer Learning)  
D - '123' (Frozen Layers Numbers)  
E - 'Re1 - Re2' (Reynolds Number Change)  

# Process

For training and verifying PINN model:
Clean and process source data for training case A -> Train model A on said data -> Clean and process test data -> Make and plot flow predictions -> Compare predictions to true values and compute errors

For TL:
Change hyper paramters of model A -> process target data -> re-train model on target data -> Make and plot flow predictions -> Compare predictions to true values and compute errors 