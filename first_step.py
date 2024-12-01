import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Carregar o arquivo de áudio
sample_rate, audio_data = wavfile.read("vazado_3.wav")

# Normalizar o áudio (se necessário)
audio_data = audio_data / np.max(np.abs(audio_data))

# Estatísticas básicas do áudio
mean_amplitude = np.mean(audio_data)
std_amplitude = np.std(audio_data)

print(f"Média da Amplitude: {mean_amplitude}")
print(f"Desvio Padrão da Amplitude: {std_amplitude}")

# Plotar o áudio no domínio do tempo
plt.figure(figsize=(10, 4))
plt.plot(audio_data)
plt.title("Sinal no Domínio do Tempo")
plt.xlabel("Amostras")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# Espectro de frequência usando FFT
freqs = np.fft.rfftfreq(len(audio_data), 1 / sample_rate)
fft_magnitude = np.abs(np.fft.rfft(audio_data))

# Plotar espectro de frequência com escala logarítmica
plt.figure(figsize=(10, 4))
plt.semilogx(freqs, fft_magnitude, label="Espectro de Frequência")
plt.axvline(500, color='r', linestyle='--', label='500 Hz (Corte Inferior)')
plt.axvline(1650, color='g', linestyle='--', label='1650 Hz (Corte Superior)')
plt.title("Espectro de Frequência")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

# Zoom na faixa de interesse (escala linear)
plt.figure(figsize=(10, 4))
plt.plot(freqs, fft_magnitude, label="Espectro de Frequência")
plt.axvline(500, color='r', linestyle='--', label='500 Hz (Corte Inferior)')
plt.axvline(1650, color='g', linestyle='--', label='1650 Hz (Corte Superior)')
plt.xlim(0, 2000)  # Limitar a exibição a 0-2000 Hz
plt.title("Espectro de Frequência (Zoom na Faixa de Interesse)")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

# Normalizar o espectro de frequência para destacar componentes menos intensas
normalized_fft = fft_magnitude / np.max(fft_magnitude)

# Plotar espectro de frequência normalizado
plt.figure(figsize=(10, 4))
plt.plot(freqs, normalized_fft, label="Espectro Normalizado")
plt.axvline(500, color='r', linestyle='--', label='500 Hz (Corte Inferior)')
plt.axvline(1650, color='g', linestyle='--', label='1650 Hz (Corte Superior)')
plt.xlim(0, 2000)  # Limitar a exibição a 0-2000 Hz
plt.title("Espectro de Frequência Normalizado")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Amplitude Normalizada")
plt.legend()
plt.grid()
plt.show()