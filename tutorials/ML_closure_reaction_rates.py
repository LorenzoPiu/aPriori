#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:50:44 2024

@author: lorenzo piu
"""

import os
import aPriori as ap
from aPriori.DNS import Field

# Uncomment the following line if you did not download the dataset yet
# ap.download(dataset='h2_lifted')

directory = os.path.join('.', 'Lifted_H2_subdomain') # change this with your path to the data folder

T_path = os.path.join(directory,'data', 'T_K_id000.dat')
print(f"\nChecking the path \'{T_path}\' is correct...")
if not os.path.exists(T_path):
    print(f"The path '{T_path}' does not exist in your system. Downloading the dataset from Github...")
    ap.download(dataset='h2_lifted')

# Initialize the DNS field
DNS_field = Field(directory)

# Compute the reaction rates
DNS_field.compute_reaction_rates()

# # Compute the strain rate module on the DNS data without saving the tensor components
DNS_field.compute_strain_rate(save_tensor=False)

# FILTERING
filter_size = 16

# filter DNS field and initialize filtered field
filtered_field = Field(DNS_field.filter_favre(filter_size))

# compute the reaction rates on the filtered field
filtered_field.compute_reaction_rates()

# compute the strain rate on the filtered field
filtered_field.compute_strain_rate(save_tensor=True)

# compute the residual dissipation rate with Smagorinsky model
filtered_field.compute_residual_dissipation_rate(mode='Smag')

# compute residual kinetic energy
filtered_field.compute_residual_kinetic_energy()

# compute chemical timescale with SFR, FFR and Chomiak model 
filtered_field.compute_chemical_timescale(mode='SFR')
filtered_field.fuel = 'H2'
filtered_field.ox = 'O2'
filtered_field.compute_chemical_timescale(mode='Ch')
filtered_field.compute_mixing_timescale(mode='Kolmo')

###############################################################################
#                        Machine Learning Closure
###############################################################################
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# DATA PROCESSING
T      = filtered_field.T.reshape_column() # extract the valeue of Temperature from the filtered field and rehape it as a column vector
S      = filtered_field.S_LES.reshape_column()
Tau_c  = filtered_field.Tau_c_SFR.reshape_column()

HRR_LFR = filtered_field.HRR_LFR.reshape_column()
HRR_DNS = filtered_field.HRR_DNS.reshape_column()

# Data scaling
T      = (T-np.min(T))/(np.max(T)-np.min(T))
S      = np.log10(S)
S      = (S-np.min(S))/(np.max(S) - np.min(S))
Tau_c  = np.log10(Tau_c)
Tau_c  = (Tau_c-np.min(Tau_c)) / (np.max(Tau_c)-np.min(Tau_c))

# Build the training data matrix
X = np.hstack([T, S, Tau_c])

# Divide between train and test data
X_train, X_test, HRR_LFR_train, HRR_LFR_test, HRR_DNS_train, HRR_DNS_test = train_test_split(
    X, HRR_LFR, HRR_DNS, test_size=0.9, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
HRR_LFR_train = torch.tensor(HRR_LFR_train, dtype=torch.float32)
HRR_LFR_test  = torch.tensor(HRR_LFR_test,  dtype=torch.float32)
HRR_DNS_train = torch.tensor(HRR_DNS_train, dtype=torch.float32)
HRR_DNS_test  = torch.tensor(HRR_DNS_test,  dtype=torch.float32)


# NN class definition
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList() # initialize the layers list as an empty list using nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size)) # Add the first input layer. The layer takes as input <input_size> neurons and gets as output <hidden_size> neurons
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size)) # Add hidden layers
        self.layers.append(nn.Linear(hidden_size, output_size)) # add output layer

    def forward(self, x):    # Function to perform forward propagation
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x
    
# NN architecture
input_size = 3
num_layers = 6
hidden_size = 64
output_size = 1
model = NeuralNetwork(input_size, hidden_size, output_size, num_layers)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters()) # here we are using the Adam optimizer, to optimize model.parameters, but what is there inside this attribute?

# transfer on GPU
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    
model = model.to(device)
X_train = X_train.to(device)
X_test = X_test.to(device)
HRR_LFR_train = HRR_LFR_train.to(device)
HRR_DNS_train = HRR_DNS_train.to(device)
HRR_LFR_test = HRR_LFR_test.to(device)
HRR_DNS_test = HRR_DNS_test.to(device)

# Move the optimizer's state to the same device
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)

# Lists to store training loss and testing loss
train_loss_list = []
test_loss_list = []

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    output = model(X_train)

    # compute loss on training data
    loss = criterion(output*HRR_LFR_train, HRR_DNS_train)

    # Compute loss on testing data. NOTE: we aren't gonna use the test loss for optimization!!!
    with torch.no_grad():
        output_test = model(X_test)
        loss_test = criterion(output_test*HRR_LFR_test, HRR_DNS_test)

    # Backprop and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_loss_list.append(loss.item()) # Save the losses
    test_loss_list.append(loss_test.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test loss: {loss_test.item():.4f}')

# Plot training and testing loss
plt.plot(train_loss_list, label='Training Loss')
plt.plot(test_loss_list, label='Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Testing Loss')
plt.yscale('log')
plt.legend()
plt.show()

# PLOTTING
with torch.no_grad():
    gamma = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()

# Visualize the results
f = ap.parity_plot(HRR_DNS, HRR_LFR, density=True, 
               x_name=r'$\dot{Q}_{DNS}$',
               y_name=r'$\dot{Q}_{LFR}$',
               cbar_title=r'$\rho_{KDE}/max(\rho_{KDE})$',
               )
f = ap.parity_plot(HRR_DNS, gamma*HRR_LFR,density=True, 
               x_name=r'$\dot{Q}_{DNS}$',
               y_name=r'$\dot{Q}_{ML}$',
               cbar_title=r'$\rho_{KDE}/max(\rho_{KDE})$',
               )

gamma_2D = gamma.reshape(filtered_field.shape)[:,:,filtered_field.shape[2]//2] # extract the z midplane of gamma
HRR_LFR_2D = HRR_LFR.reshape(filtered_field.shape)[:,:,filtered_field.shape[2]//2]# extract the z midplane
HRR_ML_2D = gamma_2D * HRR_LFR_2D
HRR_DNS_2D = HRR_DNS.reshape(filtered_field.shape)[:,:,filtered_field.shape[2]//2]# extract the z midplane

f = ap.contour_plot(filtered_field.mesh.X_midZ*1000,   # Extract x mesh on the z midplane
                    filtered_field.mesh.Y_midZ*1000,   # Extract y mesh on the z midplane
                    np.abs(HRR_LFR_2D-HRR_DNS_2D),
                    vmax=1.5e10,
                    colormap='Reds',
                    x_name='x [mm]',
                    y_name='y [mm]',
                    title=r'$|\dot{Q}_{LFR}-\dot{Q}_{DNS}|$'
                    )

f = ap.contour_plot(filtered_field.mesh.X_midZ*1000,   # Extract x mesh on the z midplane
                    filtered_field.mesh.Y_midZ*1000,   # Extract y mesh on the z midplane
                    np.abs(HRR_ML_2D-HRR_DNS_2D),
                    vmax=1.5e10,
                    colormap='Reds',
                    x_name='x [mm]',
                    y_name='y [mm]',
                    title=r'$|\dot{Q}_{LFR}-\dot{Q}_{DNS}|$'
                    )

# Visualize the NN output
f = ap.contour_plot(filtered_field.mesh.X_midZ*1000,   # Extract x mesh on the z midplane
                    filtered_field.mesh.Y_midZ*1000,   # Extract y mesh on the z midplane
                    gamma_2D,
                    colormap='viridis',
                    x_name='x [mm]',
                    y_name='y [mm]',
                    title=r'$\gamma_{NN}$'
                    )
