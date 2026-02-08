# Audio Noise Reduction

A Python-based audio noise reduction system that applies advanced signal processing techniques to clean noisy audio recordings. This project implements custom algorithms for spectral subtraction, Wiener filtering, and multi-band processing.

## Features

- Optimized spectral subtraction
- Adaptive Wiener filtering
- Multi-band frequency filtering
- Electrical hum suppression
- Support for multiple audio formats (mp3, aac, m4a, wma, ogg, flac)
- Detailed analysis plots
- Quality metrics (RMS, SNR)

## Technical Approach

This project combines several signal processing techniques to effectively reduce noise while preserving speech quality. The implementation uses standard libraries for mathematical operations and file I/O, while the core noise reduction algorithms are custom-developed.

### Implemented Algorithms

1. **STFT (Short-Time Fourier Transform)**
   - FFT-based implementation with Hann windowing
   - Optimized window size and overlap parameters
   - Minimizes spectral leakage

2. **Wiener Filter**
   - Based on classic Wiener filtering theory
   - Dynamic parameter adjustment based on SNR
   - Preserves phase information

3. **Spectral Subtraction**
   - Adaptive implementation with dynamic alpha values
   - SNR-dependent threshold adjustment
   - Reduces musical noise artifacts

4. **Multi-Band Filtering**
   - Frequency-specific gain coefficients
   - Preserves speech frequencies (300-3400 Hz)
   - Attenuates low and high frequency noise

5. **Hum Suppression**
   - Targets 50 Hz electrical interference and harmonics
   - Reduces low frequency rumble (below 80 Hz)
   - Attenuates high frequency noise (above 9 kHz)

### Libraries Used

The project uses the following libraries for specific tasks:

- **numpy**: Array operations and mathematical computations
- **pydub & ffmpeg**: Audio format conversions only
- **matplotlib**: Analysis plots and visualization

The core noise reduction algorithms are custom implementations that don't rely on pre-built filtering functions from these libraries.

## Installation

1. Clone or download the project

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. FFmpeg Installation:
   
   This project uses FFmpeg for audio format conversions. Install it for your operating system:

   **Windows**:
   - Download from https://ffmpeg.org/download.html
   - Extract the zip file
   - Add the `bin` folder to your system PATH
   - Or copy `ffmpeg.exe` from the `bin` folder to the project directory

   **macOS**:
   ```bash
   brew install ffmpeg
   ```

   **Linux (Ubuntu/Debian)**:
   ```bash
   sudo apt-get update
   sudo apt-get install ffmpeg
   ```

   **Linux (Fedora)**:
   ```bash
   sudo dnf install ffmpeg
   ```

   Test the installation:
   ```bash
   ffmpeg -version
   ```

## Usage

1. Run the program:
```bash
python main.py
```

2. Enter the audio file name when prompted

3. The program will automatically:
   - Convert the file to WAV format if needed
   - Apply noise reduction
   - Generate analysis plots
   - Save the cleaned audio

All outputs are saved in the `outputs` folder:
- Cleaned audio: `clean_enhanced_combined_[filename].wav`
- Analysis plots: `noise_reduction_analysis_[filename].png`

## Technical Details

### Quality Metrics

The program calculates these metrics after processing:
- RMS (Root Mean Square) values
- SNR (Signal-to-Noise Ratio) in dB
- Noise reduction amount in dB

## Results

Typical results on test recordings:
- 3-4 dB SNR improvement on average
- Noticeable improvement in speech clarity
- Significant reduction in background noise
- Up to 70% reduction in electrical hum

## Project Structure

```
audio_noise_reduction/
├── main.py              # Main program entry point
├── audio_converter.py   # Audio format conversion
├── filters.py           # Noise reduction algorithms
├── audio_io.py          # WAV file I/O operations
├── analyzer.py          # Visualization and plotting
├── requirements.txt     # Python dependencies
├── ffmpeg.exe           # FFmpeg executable (Windows)
└── outputs/             # Output directory for processed files
```
