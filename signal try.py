import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal
import pandas as pd

def read_audio_file(file_path):
    # Read audio file
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    return audio_data, sample_rate

def plot_time_domain(signal, sample_rate, title="Time Domain Signal"):
    # Plot the time domain signal
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(signal, sr=sample_rate)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

def plot_frequency_domain(signal, sample_rate, title="Frequency Domain Signal"):
    # Calculate the Fourier Transform
    frequencies = np.fft.fftfreq(len(signal), d=1/sample_rate)
    fft_result =np.fft.fft(signal)

    # Plot the frequency domain signal
    plt.figure(figsize=(12, 4))
    plt.plot(frequencies, np.abs(fft_result))
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.show()

def filter(data: np.ndarray,filter_type='lowpass',cutoff_frequency=1000, sample_rate: float=44100, poles: int = 1):
    if filter_type=='lowpass':
        sos = scipy.signal.butter(poles, cutoff_frequency, 'lowpass', fs=sample_rate, output='sos')
        filtered_data = scipy.signal.sosfiltfilt(sos, data)
    elif filter_type=='highpass':
        sos = scipy.signal.butter(poles, cutoff_frequency, 'highpass', fs=sample_rate, output='sos')
        filtered_data = scipy.signal.sosfiltfilt(sos, data)
    else:
        raise ValueError("Invalid filter type. Use 'Ideal' or 'NonIdeal'.")
    return filtered_data

def apply_filter(signal, filter_type='lowpass', cutoff_frequency=1000, sample_rate=44100,Gain=1,order =1,type_f="NonIdeal"):
    # Apply a filter (lowpass or highpass)
    if type_f=="NonIdeal":
        if filter_type == 'lowpass':
            filter_mask = np.zeros_like(signal)
            filter_mask= 1/np.sqrt(1+(np.fft.fftfreq(len(signal), d=1/sample_rate)/cutoff_frequency)**(2*order))
            print(filter_mask)
        elif filter_type == 'highpass':
            filter_mask = np.zeros_like(signal)
            filter_mask= 1/np.sqrt(1+(cutoff_frequency/np.fft.fftfreq(len(signal), d=1/sample_rate))**(2*order))
        else:
            raise ValueError("Invalid filter type. Use 'lowpass' or 'highpass'.")
    elif type_f=="Ideal":
        if filter_type == 'lowpass':
            filter_mask = np.zeros_like(signal)
            filter_mask[np.fft.fftfreq(len(signal), d=1/sample_rate)<cutoff_frequency]=1
        elif filter_type == 'highpass':
            filter_mask = np.zeros_like(signal)
            filter_mask[np.fft.fftfreq(len(signal), d=1/sample_rate)>cutoff_frequency]=1
        else:
            raise ValueError("Invalid filter type. Use 'lowpass' or 'highpass'.")
    else:
        raise ValueError("Invalid filter type. Use 'Ideal' or 'NonIdeal'.")

    # Apply the filter in the frequency domain
    filtered_fft = np.fft.fft(signal)* filter_mask * Gain
    # Calculate the inverse Fourier Transform to get the time-domain filtered signal
    filtered_signal = np.fft.ifft(filtered_fft)

    return np.real(filtered_signal)

def save_audio_file(file_path, signal, sample_rate):
    # Save the signal as an audio file
    sf.write(file_path, signal, sample_rate)

# Names_of_team_members=["Mohamed Hussien Mansour Sayed","Mohamed Sayed Ahmed ElSayed","Yousef Ahmed Mohamed","Mahmoud Hazem","Yousef Mahmoud Mohamed Amin"]

# for i in range(5):

# Specify the path to your audio file
audio_file_path = r"D:\Signal team/Voice_team signal.mp3"

# Read the original audio file
original_signal, sample_rate = read_audio_file(audio_file_path)

# Task 1: Plot the original audio signal in the time domain
plot_time_domain(original_signal, sample_rate, title=f"Original Time Domain Signal of Voice team signal")

# Task 2: Plot the frequency domain representation of the original signal
plot_frequency_domain(original_signal, sample_rate, title=f"Original Frequency Domain Signal of Voice team signal")

# Task 3: Apply a lowpass filter to the original signal and plot the filtered signal in the frequency domain
filtered_signal = apply_filter(original_signal, filter_type='lowpass', cutoff_frequency=1000, sample_rate=sample_rate)

plot_frequency_domain(filtered_signal, sample_rate, title=f"Filtered Frequency Domain Signal of Voice team signal")

# Task 4: Find the corresponding signal in time domain for the filtered signal and plot it
plot_time_domain(filtered_signal, sample_rate, title=f"Filtered Time Domain Signal Voice team signal")

# Task 5: Save the filtered signal in time domain as an audio file
filtered_audio_file_path = r"D:\Signal team/Voice_team signal_filtered.mp3"
save_audio_file(filtered_audio_file_path, filtered_signal, sample_rate)

# Optionally, you can play the original and filtered signals to hear the difference.
