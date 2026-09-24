import numpy as np, lameenc

# Parameters
c = 343.0      # speed of sound (m/s)
v = 40.0       # source speed (m/s) ~ 144 km/h
f0 = 440.0     # emitted frequency (Hz)
d = 5.0        # closest distance to observer (m)
T = 8.0        # duration (s)
sr = 44100     # sample rate

# Source moves along x at speed v, passing the observer (at origin) at emission time te = T/2
te = np.linspace(-1, T + 1, 4_000_000)          # dense grid of emission times
x = v * (te - T / 2)
r = np.sqrt(x**2 + d**2)                        # source-observer distance at emission
t_arrival = te + r / c                          # when that sound reaches the observer

# Invert: for each observer sample time t, find the emission time te (exact retarded time)
t = np.arange(int(T * sr)) / sr
te_of_t = np.interp(t, t_arrival, te)
r_of_t = np.sqrt((v * (te_of_t - T / 2))**2 + d**2)

# The source emits sin(2*pi*f0*te); amplitude falls as 1/r. Doppler shift comes out automatically.
signal = np.sin(2 * np.pi * f0 * te_of_t) / r_of_t
signal *= 0.9 / np.abs(signal).max()

pcm = (signal * 32767).astype(np.int16)
enc = lameenc.Encoder()
enc.set_bit_rate(128); enc.set_in_sample_rate(sr); enc.set_channels(1); enc.set_quality(2)
with open("doppler.mp3", "wb") as f:
    f.write(enc.encode(pcm.tobytes()) + enc.flush())

print("Approaching: %.1f Hz | Receding: %.1f Hz" % (f0 * c / (c - v), f0 * c / (c + v)))
