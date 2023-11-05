import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import time

"""
Anton Brekke / antonabr@uio.no
Making a class so one can make several objects
with same properties. 
"""

class Freq_Analysis:
    def __init__(self, sampled_signal, fs, t):
        self.sampled_signal = sampled_signal
        self.fs = fs    # Sample frequency
        self.t = t
        self.T = t[-1] - t[0]      # Total sample-time
        self.dt = 1 / fs    # Time betweem sampling-points
        self.N = len(t)      # Number of sampling-points

    def FourierTransform(self):
        x_n = self.sampled_signal
        X_k = np.fft.fft(x_n)   # FT of sampled frequency
        FT_freq = np.fft.fftfreq(self.N, self.dt)     # FT-frequency

        return FT_freq, X_k

    # Fouriertransform of Morlet-wavelet
    def FT_wavelet(self, omega, omega_a, K):
        w = 2 * (np.exp(-(K * (omega - omega_a)/omega_a)**2) - np.exp(-K**2) * np.exp(-(K*omega/omega_a)**2))
        return w

    # Fast algortihm of wavelet-transform using convolution theorem
    def faster_wavelet_diagram(self, omega_a, K):
        tk = self.t.copy()
        omega_a_mesh, tk_mesh = np.meshgrid(omega_a, tk, indexing='ij')
        omega_0 = np.fft.fftfreq(self.N, self.dt) * 2*np.pi
        x_n = self.sampled_signal.copy()
        x_nFT = np.fft.fft(x_n)
        N = len(tk)
        M = len(omega_a)
        WT = np.zeros([M, N], dtype=np.complex128)
        for i in range(M):
            WT[i, :] = np.fft.ifft(x_nFT * self.FT_wavelet(omega_0, omega_a[i], K)) # Convolution-theorem
        return tk_mesh, omega_a_mesh, WT


fs, data = wavfile.read('tawny_owl.wav')    # data is the sampled signal
print(f'Sample frequency: {fs}')
data = data[:,0]  # Returns two channels from stereo, almost identical
N = len(data)   # Number of data-points
dt = 1 / fs     # Time between sample-points
T = N*dt        # Total time of sample
t = np.linspace(0, T, N)

analyse = Freq_Analysis(data, fs, t)        # Making object
FT_freq, X_k = analyse.FourierTransform()       # Getting FT-signal


# Plotting Fourier-transform to find interesting/relevant domain of frequency
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.plot(t, data)
ax1.set_xlabel('t [s]', weight='bold')
ax1.set_ylabel('A [m]', weight='bold')
ax1.set_title('Incoming signal', weight='bold')
ax2.plot(FT_freq, abs(X_k))
ax2.set_xlabel('freq [Hz]', weight='bold')
ax2.set_ylabel('X_k', weight='bold')
ax2.set_title('Fourier Transform', weight='bold')

ax2in = ax2.inset_axes([0.05, 0.4, 0.2, 0.5])
ax2in.plot(FT_freq, abs(X_k))
ax2in.set_xlim(0, 4000)
ax2in.set_ylim(0, 3300)
ax2.indicate_inset_zoom(ax2in, edgecolor="black")

fig.tight_layout()
plt.show()

# FT-plot : freq [500 - 1000], time [1s - 2.5s]

omega_a = np.logspace(np.log10(2*np.pi*500), np.log10(2*np.pi*1000), 500)
K = 40

# Plotting Wavelet-Transform
fig = plt.figure()
fig.suptitle('Faster Wavelet Transform', weight='bold')
ax = fig.add_subplot()

# Measuring run-time of the code
time_start = time.time()
t_mesh, omega_a_mesh, WT = analyse.faster_wavelet_diagram(omega_a, K)
time_end = time.time()
time_tot = time_end - time_start
print(f'faster_wavelet_diagram ran in {time_tot}')

# Plotting every 10'th data point
p = ax.contourf(t_mesh[::10, ::10], omega_a_mesh[::10, ::10] / (2*np.pi), abs(WT)[::10, ::10], levels=300, cmap='hot')
cbar_ax = fig.colorbar(p, ax=ax)

cbar_ax.set_label('Amplitude', weight='bold')
ax.set_xlabel('t [s]', weight='bold'); ax.set_ylabel('freq [1/s]', weight='bold')
ax.set_title(f'K = {K}', weight='bold')

fig.tight_layout()
plt.show()

