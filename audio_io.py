import wave
import struct
import numpy as np

def read_wav(filename):
    """Read WAV file and return samples with sample rate."""
    wf = wave.open(filename, 'rb')
    n_channels = wf.getnchannels()
    sample_rate = wf.getframerate()
    n_samples = wf.getnframes()
    raw_data = wf.readframes(n_samples)
    wf.close()
    samples = struct.unpack('<' + 'h'*n_samples*n_channels, raw_data)
    if n_channels > 1:
        samples = samples[::n_channels]
    return np.array(samples, dtype=float), sample_rate

def save_wav(filename, samples, sample_rate):
    """Save samples as WAV file."""
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    samples = np.nan_to_num(samples, nan=0.0, posinf=32767, neginf=-32767)
    samples = np.clip(samples, -32767, 32767)
    samples = samples.astype(np.int16)
    raw_data = struct.pack('<' + 'h'*len(samples), *samples)
    wf.writeframes(raw_data)
    wf.close()
