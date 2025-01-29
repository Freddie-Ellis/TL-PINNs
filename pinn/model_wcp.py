# pinn/model.py

import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from pinn.config import run_ID, LEARNING_RATE
from pinn.utils import save_model

class PhysicsInformedNN_wcp:
    def __init__(self, x, y, t, u, v, layers, pretrain_path=None, layers_to_freeze=None, num_collocation=100):
        X = np.concatenate([np.atleast_2d(x), np.atleast_2d(y), np.atleast_2d(t)], axis=1)
        self.lb = X.min(0)
        self.ub = X.max(0)

        self.x = tf.convert_to_tensor(X[:, 0:1], dtype=tf.float32)
        self.y = tf.convert_to_tensor(X[:, 1:2], dtype=tf.float32)
        self.t = tf.convert_to_tensor(X[:, 2:3], dtype=tf.float32)

        self.u = tf.convert_to_tensor(u, dtype=tf.float32)
        self.v = tf.convert_to_tensor(v, dtype=tf.float32)

        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
        self.lambda_1 = tf.Variable(0.0, dtype=tf.float32, name="lambda_1")
        self.lambda_2 = tf.Variable(0.0, dtype=tf.float32, name="lambda_2")
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        self.loss_history = []

        # Generate collocation points 
        self.x_f = tf.convert_to_tensor(np.random.uniform(self.lb[0], self.ub[0], (num_collocation, 1)), dtype=tf.float32)
        self.y_f = tf.convert_to_tensor(np.random.uniform(self.lb[1], self.ub[1], (num_collocation, 1)), dtype=tf.float32)
        self.t_f = tf.convert_to_tensor(np.random.uniform(self.lb[2], self.ub[2], (num_collocation, 1)), dtype=tf.float32)

        # Load weights if provided
        if pretrain_path is not None:
            self.load_model(pretrain_path)

        # Track frozen layers
        self.frozen_indices = layers_to_freeze if layers_to_freeze else []
        print("Frozen Layers:", self.frozen_indices)

    def initialize_NN(self, layers):
        weights, biases = [], []
        for l in range(len(layers) - 1):
            W = tf.Variable(self.xavier_init([layers[l], layers[l + 1]]), dtype=tf.float32)
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32))
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim, out_dim = size
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev)

    def neural_net(self, X, weights, biases):
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for W, b in zip(weights[:-1], biases[:-1]):
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        return tf.add(tf.matmul(H, weights[-1]), biases[-1])
    
    def net_NS(self, x, y, t):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
    
        # Use a single GradientTape to compute both first and second derivatives
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y, t])
            
            # Forward pass through the neural network
            psi_and_p = self.neural_net(tf.concat([x, y, t], 1), self.weights, self.biases)
            psi = psi_and_p[:, 0:1]  # Stream function
            p = psi_and_p[:, 1:2]    # Pressure
    
            # First-order derivatives
            u = tape.gradient(psi, y)  # u = d(psi)/dy
            v = -tape.gradient(psi, x)  # v = -d(psi)/dx
            
            # Print first-order derivatives to debug
            #print(f"u: {u}, v: {v}")
    
            if u is None or v is None:
                raise ValueError("Error: First-order derivatives u or v are None.")
    
            # Second-order derivatives
            u_t = tape.gradient(u, t)
            u_x = tape.gradient(u, x)
            u_y = tape.gradient(u, y)
            v_t = tape.gradient(v, t)
            v_x = tape.gradient(v, x)
            v_y = tape.gradient(v, y)
    
            # Second-order spatial derivatives
            u_xx = tape.gradient(u_x, x)
            u_yy = tape.gradient(u_y, y)
            v_xx = tape.gradient(v_x, x)
            v_yy = tape.gradient(v_y, y)
            p_x = tape.gradient(p, x)
            p_y = tape.gradient(p, y)
    
            # Debug second-order gradients
            #print(f"u_xx: {u_xx}, u_yy: {u_yy}, v_xx: {v_xx}, v_yy: {v_yy}")
    
            if u_xx is None or u_yy is None or v_xx is None or v_yy is None:
                raise ValueError("Error: Second-order derivatives are None.")
    
        # Free up memory by deleting tape
        del tape
    
        # Navier-Stokes equations
        f_u = u_t + lambda_1 * (u * u_x + v * u_y) + p_x - lambda_2 * (u_xx + u_yy)
        f_v = v_t + lambda_1 * (u * v_x + v * v_y) + p_y - lambda_2 * (v_xx + v_yy)
    
        return u, v, p, f_u, f_v

    @tf.function
    def loss(self):
        # Compute supervised loss (data loss)
        u_pred, v_pred, _, _, _ = self.net_NS(self.x, self.y, self.t)
        loss_data = tf.reduce_mean(tf.square(self.u - u_pred)) + \
                    tf.reduce_mean(tf.square(self.v - v_pred))

        # Compute physics loss using collocation points
        _, _, _, f_u_f_pred, f_v_f_pred = self.net_NS(self.x_f, self.y_f, self.t_f)
        loss_physics = tf.reduce_mean(tf.square(f_u_f_pred)) + tf.reduce_mean(tf.square(f_v_f_pred))

        # Combine losses
        loss_value = loss_data + loss_physics
        return loss_value
        
    @tf.function
    def train_step(self):
        with tf.GradientTape() as tape:
            loss_value = self.loss()
            
        # Get gradients for all trainable variables
        trainable_vars = []
        for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            if idx not in self.frozen_indices:  # Skip frozen layers
                trainable_vars.extend([w, b])
            
        trainable_vars.extend([self.lambda_1, self.lambda_2])  # Always include these
            
        gradients = tape.gradient(loss_value, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return loss_value
    
    def train(self, nIter):
        start_time = time.time()
        self.lambda_1_history = []  # Store lambda_1 values
        self.lambda_2_history = []  # Store lambda_2 values
        self.loss_data_history = []  # Store data loss values
        self.loss_physics_history = []  # Store physics loss values

        for it in range(nIter):
            loss_value = self.train_step()
            self.loss_history.append(loss_value.numpy())
            self.lambda_1_history.append(self.lambda_1.numpy())  # Append current lambda_1
            self.lambda_2_history.append(self.lambda_2.numpy())  # Append current lambda_2

            # Print the progress
            if it % 100 == 0:
                elapsed = time.time() - start_time
                lambda_1_value = self.lambda_1.numpy()
                lambda_2_value = self.lambda_2.numpy()
                loss_data = tf.reduce_mean(tf.square(self.u - self.net_NS(self.x, self.y, self.t)[0])) + \
                            tf.reduce_mean(tf.square(self.v - self.net_NS(self.x, self.y, self.t)[1]))

                loss_physics = tf.reduce_mean(tf.square(self.net_NS(self.x_f, self.y_f, self.t_f)[3])) + \
                            tf.reduce_mean(tf.square(self.net_NS(self.x_f, self.y_f, self.t_f)[4]))
                self.loss_data_history.append(loss_data.numpy())
                self.loss_physics_history.append(loss_physics.numpy())

                print(f'It: {it}, Loss: {loss_value:.3e}, l1: {lambda_1_value:.3f}, l2: {lambda_2_value:.5f}, Time: {elapsed:.2f}')
                start_time = time.time()

            if it % 1000 == 0 and it > 0:
                save_model(self, f'models/{run_ID}/{run_ID}_{it}.npz')
                    
    def plot_loss_history(self, save_path=f'plots/savehistory_{run_ID}'):
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, label='Total Loss History')
        plt.plot(self.loss_data_history, label='Data Loss History')
        plt.plot(self.loss_physics_history, label='Physics Loss History')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.title('Loss History During Training')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        print(f"Loss history plot saved to {save_path}")
        
    def plot_lambda_history(self, save_path=f'plots/lambda_history_{run_ID}'):
        plt.figure(figsize=(12, 8))
        
        # Plot lambda_1 history
        plt.subplot(2, 1, 1)
        plt.plot(self.lambda_1_history, label='Lambda 1', color='blue')
        plt.xlabel('Iteration')
        plt.ylabel('Lambda 1 Value')
        plt.title('Lambda 1 History During Training')
        plt.grid(True)
        plt.legend()
        
        # Plot lambda_2 history
        plt.subplot(2, 1, 2)
        plt.plot(self.lambda_2_history, label='Lambda 2', color='orange')
        plt.xlabel('Iteration')
        plt.ylabel('Lambda 2 Value')
        plt.title('Lambda 2 History During Training')
        plt.grid(True)
        plt.legend()

        # Save the plot
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Lambda history plot saved to {save_path}")

    def predict(self, x_star, y_star, t_star):
        x_star = tf.convert_to_tensor(x_star, dtype=tf.float32)
        y_star = tf.convert_to_tensor(y_star, dtype=tf.float32)
        t_star = tf.convert_to_tensor(t_star, dtype=tf.float32)
        
        u_pred, v_pred, p_pred, f_u_pred, f_v_pred= self.net_NS(x_star, y_star, t_star)
        return u_pred.numpy(), v_pred.numpy(), p_pred.numpy(), f_u_pred.numpy(), f_v_pred.numpy()

    def load_model(self, path, load_weights=True, load_params=True):
            data = np.load(path, allow_pickle=True)
                
            if load_weights:
                for i in range(len(self.weights)):
                    weight_key = f'weight_{i}'
                    bias_key = f'bias_{i}'
                        
                    if weight_key in data and bias_key in data:
                        self.weights[i].assign(tf.convert_to_tensor(data[weight_key]))
                        self.biases[i].assign(tf.convert_to_tensor(data[bias_key]))
                
            if load_params:
                if 'lambda_1' in data and 'lambda_2' in data:
                    self.lambda_1.assign(tf.convert_to_tensor(data['lambda_1']))
                    self.lambda_2.assign(tf.convert_to_tensor(data['lambda_2']))

            print(f"Model loaded from {path}")
