from training import model
import numpy as np
import matplotlib.pyplot as plt

def FFT(save_dir, model_path, x = 1, y = 0):
    model.load_model(model_path)

    # Define the time range for sampling
    t_values = np.linspace(0, 20, 2000)  # Time values from 0 to 20 seconds, 2000 samples

    # Prepare arrays for prediction
    x_user = np.full((len(t_values), 1), x)
    y_user = np.full((len(t_values), 1), y)
    t_user = t_values.reshape(-1, 1)

    # Get predictions from the trained PINN model
    u_pred_user, v_pred_user, p_pred_user, _, _ = model.predict(x_user, y_user, t_user)

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
    plt.plot(t_values, signal, label="v-velocity (time-domain)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Time-Domain Signal of Predicted v at (x=1, y=0)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/signal", dpi=300)
    plt.close()

    print('FFT Complete')