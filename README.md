# TL-PINNs
Third year individual project on Transfer Learning (TL) integrated PINNs for fluid mechanics.
I'm currently connecting this repository to MobaXterm for HPC usage.

Workflow is as follows:

Make change in VS -> Push to GitHub -> pull on Moba using 'git reset --hard -> git clean -fd -> git pull origin main' from within cd 

Make change in Moba -> Push to GitHub ('make the change' -> git status -> git add . -> git commit -m "update message" -> git push origin main) -> Pull on local machine in GitHub Desktop

# Requirements

TensorFlow == 2.7.0  
NumPy == 1.24.4  
Cuda == 11.2  
Cudnn == 8.1  
matplotlib == 3.2.2  
Scipy == 1.10.1  

# Naming Convention
{run_ID}_nIter

n_Iter is number of iterations

run_ID ->
A_B_C_D:    
A is the Re of the finished model,  
B is PINN for a PINN model and TL for a transfer learned model  
C is the frozen layers if applicable  
D is something else


# Process

For training and verifying PINN model:
Clean and process source data for training case A -> Train model A on said data -> Clean and process test data -> Make and plot flow predictions -> Compare predictions to true values and compute errors

For TL:
Change hyper paramters of model A -> process target data -> re-train model on target data -> Make and plot flow predictions -> Compare predictions to true values and compute errors 

Strouhal number converges on 0.2 for Re<2e5 (I think)
for Re=100, St=0.16  ==> T=6.25
for Re=150, St=0.173  ==> T=5.78
for Re=100000, St=0.19996 ==> T=5
Therefore we converge on a period of 5s, we need a snapshot every 5/3 Seconds.

Change save it -> change print it -> change save type for Tl -> change config stuff -> change slurm file

The checkpointed models for the TL saved as the same name as the loaded pre trained model