import numpy as np

class NoiseReducer:
    """
    Advanced noise reduction using multiple filtering techniques.
    Implements spectral subtraction, Wiener filtering, and multi-band processing.
    """
    def __init__(self, fs=44100):
        self.fs = fs
        self.window_size = 2048
        self.overlap = 0.75
        self.hop_length = int(self.window_size * (1 - self.overlap))

    def compute_stft(self, signal):
        """Compute Short-Time Fourier Transform."""
        N = len(signal)
        num_windows = (N - self.window_size) // self.hop_length + 1
        frequencies = np.arange(self.window_size // 2) * self.fs / self.window_size
        stft_matrix = np.zeros((self.window_size // 2, num_windows), dtype=complex)
        hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(self.window_size) / (self.window_size - 1))
        for i in range(num_windows):
            start = i * self.hop_length
            end = start + self.window_size
            if end <= N:
                window = signal[start:end] * hann
                fft_result = np.fft.fft(window)
                stft_matrix[:, i] = fft_result[:self.window_size // 2]
        return frequencies, np.arange(num_windows) * self.hop_length / self.fs, stft_matrix
    
    def compute_istft(self, stft_matrix, frequencies, times):
        """Compute inverse Short-Time Fourier Transform."""
        _, num_windows = stft_matrix.shape
        output_length = (num_windows - 1) * self.hop_length + self.window_size
        output = np.zeros(output_length)
        hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(self.window_size) / (self.window_size - 1))
        for i in range(num_windows):
            full_spectrum = np.zeros(self.window_size, dtype=complex)
            full_spectrum[:self.window_size // 2] = stft_matrix[:, i]
            for k in range(1, self.window_size // 2):
                full_spectrum[self.window_size - k] = np.conj(full_spectrum[k])
            time_domain = np.fft.ifft(full_spectrum).real
            time_domain = time_domain * hann
            start = i * self.hop_length
            end = start + self.window_size
            output[start:end] += time_domain
        return output
    
    def estimate_noise_profile(self, stft_matrix, num_noise_frames=30):
        """Estimate noise profile from initial frames."""
        noise_frames = stft_matrix[:, :num_noise_frames]
        noise_magnitude = np.abs(noise_frames)
        noise_profile = np.percentile(noise_magnitude, 75, axis=1)
        return noise_profile
    
    def apply_wiener_filter(self, stft_matrix, noise_profile, alpha=1.2, beta=0.5, gamma=0.05):
        """Apply adaptive Wiener filter."""
        magnitude = np.abs(stft_matrix)
        phase = np.angle(stft_matrix)
        filtered_magnitude = np.zeros_like(magnitude)
        for t in range(magnitude.shape[1]):
            frame_mag = magnitude[:, t]
            snr = frame_mag / (noise_profile + 1e-10)
            wiener_gain = snr**alpha / (snr**alpha + 1)
            aggressive_mask = snr < 1.5
            wiener_gain[aggressive_mask] *= beta
            filtered_magnitude[:, t] = frame_mag * wiener_gain
        return filtered_magnitude * np.exp(1j * phase)
    
    def apply_spectral_subtraction(self, stft_matrix, noise_profile, alpha=1.2, beta=0.5):
        """Apply enhanced spectral subtraction."""
        magnitude = np.abs(stft_matrix)
        phase = np.angle(stft_matrix)
        filtered_magnitude = np.zeros_like(magnitude)
        for t in range(magnitude.shape[1]):
            frame_mag = magnitude[:, t]
            snr_estimate = frame_mag / (noise_profile + 1e-10)
            adaptive_alpha = alpha * (1 + np.exp(-snr_estimate + 2))
            cleaned = frame_mag - adaptive_alpha * noise_profile
            cleaned = np.maximum(cleaned, beta * frame_mag)
            filtered_magnitude[:, t] = cleaned
        return filtered_magnitude * np.exp(1j * phase)
    
    def apply_multiband_filter(self, stft_matrix):
        """
        Apply multi-band filtering with frequency-specific coefficients.
        Preserves speech frequencies while attenuating noise bands.
        """
        magnitude = np.abs(stft_matrix)
        phase = np.angle(stft_matrix)
        f = self.frequencies
        mask = np.ones_like(magnitude)
        mask[(f <= 60)] *= 0.1
        mask[(f > 60) & (f <= 200)] *= 0.3
        mask[(f > 200) & (f <= 500)] *= 0.8
        mask[(f > 300) & (f <= 3400)] *= 1.0
        mask[(f > 500) & (f <= 2000)] *= 1.0
        mask[(f > 1000) & (f <= 3000)] *= 1.1
        mask[(f > 2000) & (f <= 4000)] *= 0.8
        mask[(f > 4000) & (f <= 8000)] *= 0.5
        mask[f > 8000] *= 0.2
        return magnitude * mask * np.exp(1j * phase)
    
    def reduce_noise(self, signal, method='wiener'):
        """
        Main noise reduction method.
        Supports different filtering strategies: wiener, spectral, multi_band, combined.
        """
        print("Computing STFT...")
        self.frequencies, times, stft = self.compute_stft(signal)
        print("Extracting noise profile...")
        noise_profile = self.estimate_noise_profile(stft, num_noise_frames=30)
        print(f"Applying filtering ({method})...")
        if method == 'wiener':
            stft_filtered = self.apply_wiener_filter(stft, noise_profile)
        elif method == 'spectral':
            stft_filtered = self.apply_spectral_subtraction(stft, noise_profile)
        elif method == 'multi_band':
            stft_filtered = self.apply_multiband_filter(stft)
        elif method == 'combined':
            stft_temp = self.apply_spectral_subtraction(stft, noise_profile * 0.7)
            stft_filtered = self.apply_wiener_filter(stft_temp, noise_profile * 0.3)
        else:
            stft_filtered = stft
        print("Computing ISTFT...")
        clean_signal = self.compute_istft(stft_filtered, self.frequencies, times)
        return clean_signal
    
    def analyze_quality(self, original, cleaned):
        """
        Analyze quality metrics for noise reduction.
        Calculates RMS and SNR values.
        """
        print("\nKalite Analizi:")
        print("=" * 30)
        rms_original = np.sqrt(np.mean(original**2))
        rms_cleaned = np.sqrt(np.mean(cleaned**2))
        print(f"RMS Orijinal: {rms_original:.4f}")
        print(f"RMS Temizlenmiş: {rms_cleaned:.4f}")
        print(f"RMS Oranı: {rms_cleaned/rms_original:.2f}")
        noise_samples = int(0.5 * self.fs)
        noise_rms = np.sqrt(np.mean(original[:noise_samples]**2))
        signal_rms = np.sqrt(np.mean(original[noise_samples:]**2))
        snr_before = 20 * np.log10(signal_rms / (noise_rms + 1e-10))
        noise_rms_after = np.sqrt(np.mean(cleaned[:noise_samples]**2))
        signal_rms_after = np.sqrt(np.mean(cleaned[noise_samples:]**2))
        snr_after = 20 * np.log10(signal_rms_after / (noise_rms_after + 1e-10))
        print(f"Tahmini SNR Öncesi: {snr_before:.1f} dB")
        print(f"Tahmini SNR Sonrası: {snr_after:.1f} dB")
        print(f"SNR İyileştirmesi: {snr_after - snr_before:.1f} dB")

def suppress_hum(signal, fs):
    """
    Suppress electrical hum and harmonics at specific frequencies.
    Targets 50Hz power line noise and its harmonics.
    """
    N = len(signal)
    fft_data = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, 1/fs)
    target_frequencies = [49.9, 50.0, 50.1, 100.0, 150.0, 200.0]
    for i, f in enumerate(freqs):
        absf = abs(f)
        for target in target_frequencies:
            if abs(absf - target) < 0.5:
                fft_data[i] *= 0.3
        if absf < 80:
            fft_data[i] *= 0.7
        elif 80 <= absf < 500:
            fft_data[i] *= 1.0
        if absf > 9000:
            fft_data[i] *= 0.5
    cleaned = np.fft.ifft(fft_data).real
    return cleaned

def process_audio(signal, fs, method='combined'):
    """
    Process audio signal with optimized noise reduction.
    Normalizes signal, applies filters, and suppresses electrical hum.
    """
    signal = np.array(signal, dtype=float)
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val
    reducer = NoiseReducer(fs)
    clean_signal = reducer.reduce_noise(signal, method=method)
    clean_signal = clean_signal * max_val
    clean_signal = suppress_hum(clean_signal, fs)
    reducer.analyze_quality(signal * max_val, clean_signal)
    return clean_signal
