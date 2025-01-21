import time
import tensorflow as tf
from pinn.config import run_ID

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
    for it in range(nIter):
        loss_value = self.train_step()
        self.loss_history.append(loss_value.numpy())
            
        # Print the progress
        if it % 10 == 0:
            elapsed = time.time() - start_time
            lambda_1_value = self.lambda_1.numpy()
            lambda_2_value = self.lambda_2.numpy()
            print(f'It: {it}, Loss: {loss_value:.3e}, l1: {lambda_1_value:.3f}, l2: {lambda_2_value:.5f}, Time: {elapsed:.2f}')
            start_time = time.time()
            if it % 5000 == 0 and it > 0:
                self.save_model(f'models/checkpoints/{run_ID}_checkpoint_{it}.npz')