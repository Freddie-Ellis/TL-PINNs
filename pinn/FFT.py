from training import model
import numpy as np
import matplotlib.pyplot as plt
from pinn.pre_processing import process_reynolds_data
from scipy.interpolate import griddata

def FFT(save_dir, model_path, Re_TL, x=1, y=0):
    model.load_model(model_path)

    data = process_reynolds_data(Re_TL, 0, 250, 250)

    # Extract components
    x_real = data['x_train'].squeeze()
    y_real = data['y_train'].squeeze()
    t_data = data['t_train'].squeeze()
    v_real = data['v_train'].squeeze()

    # Define the target point for interpolation
    target_point = np.array([[1, 0]])  # x=1, y=0

    # Interpolate for each time step
    unique_times = np.unique(t_data)
    v_interp = []

    for t in unique_times:
        # Get data for the current time step
        mask = t_data == t
        points = np.column_stack((x_real[mask], y_real[mask]))  # Existing (x, y) points
        values = v_real[mask]  # Corresponding velocity values

        # Interpolate at (x=1, y=0)
        v_at_x1_y0 = griddata(points, values, target_point, method='cubic')
        v_interp.append(v_at_x1_y0[0])  # Store interpolated velocity

    # Convert to numpy array for plotting
    v_interp = np.array(v_interp)
    unique_times = np.linspace(0, 250, len(v_interp)) * 0.08

    t_values = np.linspace(0, np.max(t_data), 2000)  # Time values from 0 to 20 seconds, 2000 samples

    # Prepare arrays for prediction
    x_user = np.full((len(t_values), 1), x)
    y_user = np.full((len(t_values), 1), y)
    t_user = t_values.reshape(-1, 1)

    # Get predictions from the trained PINN model
    _, v_pred_user, _, _, _ = model.predict(x_user, y_user, t_user)

    # Choose a signal for FFT (e.g., velocity component u)
    signal = v_pred_user.flatten() - np.mean(v_pred_user)  # Normalize the signal

    # Perform FFT
    dt = t_values[1] - t_values[0]  # Time step
    fs = 1 / dt  # Sampling frequency
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), dt)
    amplitude = np.abs(fft_result) / len(signal)  # Normalize amplitude

    # Plot the frequency spectrum with a log scale on the y-axis
    plt.figure(figsize=(8, 6))
    plt.semilogy(frequencies[:len(frequencies)//2], amplitude[:len(amplitude)//2], label="v-velocity", color='blue')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (log scale)")
    plt.grid(True, which="both", linestyle='--', alpha=0.6)
    plt.xlim(0, 1)  # Set x-axis range from 0 to 5
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/log_FFT", dpi=300)
    plt.close()

    # Plot the frequency spectrum on a linear scale
    plt.figure(figsize=(8, 6))
    plt.plot(frequencies[:len(frequencies)//2], amplitude[:len(amplitude)//2], label="v-velocity")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Frequency Spectrum of Predicted v at (x=1, y=0)")
    plt.grid()
    plt.xlim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/FFT", dpi=300)
    plt.close()


    # Plot the time-domain signal
    plt.figure(figsize=(8, 6))
    plt.plot(unique_times, v_interp, label="Interpolated Velocity at (x=1, y=0)", color='b')
    plt.plot(t_values, v_pred_user, label="NN Prediction", color='r', linestyle='solid')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Time-Domain Signal of Predicted v at (x=1, y=0)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/signal.png", dpi=300)
    plt.close()
    

    print('FFT Complete')