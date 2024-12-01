from scipy.signal import firwin, lfilter, freqz
from scipy.io.wavfile import write, read
import numpy as np
import matplotlib.pyplot as plt

# Upload the audio file
sample_rate, audio_data = read("vazado_3.wav")

# Normalize the original audio
audio_data = audio_data / np.max(np.abs(audio_data))

low_cutoff = 500  
high_cutoff = 1650
num_taps = 201  # Number of FIR filter coefficients

# Create the bandpass FIR filter
fir_coeff = firwin(
    num_taps, [low_cutoff, high_cutoff], pass_zero="bandpass", fs=sample_rate
)

# View the frequency response of the filter
w, h = freqz(fir_coeff, worN=8000, fs=sample_rate)
plt.figure(figsize=(10, 4))
plt.plot(w, 20 * np.log10(np.abs(h)))
plt.title("Resposta em Frequência do Filtro FIR (500 Hz - 1650 Hz)")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid()
plt.show()

# Apply FIR filter to audio
filtered_audio = lfilter(fir_coeff, 1.0, audio_data)

# Normalize filtered audio to avoid clipping
filtered_audio = filtered_audio / np.max(np.abs(filtered_audio))

amplification_factor = 2.0  

# Amplify the audio
amplified_audio = filtered_audio * amplification_factor

# Ensure audio does not exceed limits [-1, 1]
amplified_audio = np.clip(amplified_audio, -1, 1)

# Save the amplified audio
write("filtered_audio.wav", sample_rate, np.int16(amplified_audio * 32767))
