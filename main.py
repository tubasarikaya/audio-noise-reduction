from audio_converter import AudioConverter
from filters import process_audio
from audio_io import read_wav, save_wav
from analyzer import plot_analysis
import os
import numpy as np

def main():
    """
    Main program function for audio noise reduction.
    Takes audio file input, converts it, applies noise reduction, and saves results.
    """
    print("ADVANCED NOISE REDUCTION SYSTEM (COMBINED METHOD)")
    print("=" * 50)
    print("Features:")
    print("   • Optimized spectral subtraction + Wiener + multi-band + hum suppression")
    print("=" * 50)
    
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    converter = AudioConverter()
    print("Supported audio formats:", ", ".join(converter.list_supported_formats()))
    filename = input("Audio file name: ")
    
    try:
        print("\nPreparing file...")
        recording_file = converter.convert(filename)
        if recording_file is None:
            print("Conversion failed!")
            return
        print("\nReading file...")
        samples, fs = read_wav(recording_file)
        print(f"File information:")
        print(f"   Sample rate: {fs} Hz")
        print(f"   Duration: {len(samples)/fs:.2f} seconds")
        print(f"   Number of samples: {len(samples):,}")
        print("\nStarting advanced combined noise reduction...")
        clean_signal = process_audio(samples, fs, method='combined')
        
        filename_base = os.path.splitext(os.path.basename(filename))[0]
        output_file = os.path.join(output_dir, f"clean_enhanced_combined_{filename_base}.wav")
        plot_file = os.path.join(output_dir, f"noise_reduction_analysis_{filename_base}.png")
        
        print(f"\nSaving: {output_file}")
        save_wav(output_file, clean_signal, fs)
        print("\nGenerating analysis plots...")
        plot_analysis(samples, clean_signal, fs, save=True, output_file=plot_file)
        print(f"\nProcess completed!")
        print(f"Cleaned file: {output_file}")
        print(f"Analysis plot: {plot_file}")
    except FileNotFoundError:
        print(f"File not found: {filename}")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
