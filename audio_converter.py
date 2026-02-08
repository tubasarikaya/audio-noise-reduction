from pydub import AudioSegment
import os
import ffmpeg

class AudioConverter:
    """
    Converts audio files to WAV format.
    Handles various audio formats (mp3, aac, m4a, etc.) and converts them to WAV.
    """
    def __init__(self):
        """Initialize converter with supported formats."""
        self.supported_formats = {
            '.mp3': 'mp3',
            '.aac': 'aac',
            '.m4a': 'm4a',
            '.wma': 'wma',
            '.ogg': 'ogg',
            '.flac': 'flac'
        }
        self.sample_rate = 44100  
    
    def is_format_supported(self, file_extension):
        """Check if the given file extension is supported."""
        return file_extension.lower() in self.supported_formats
    
    def list_supported_formats(self):
        """Return list of supported audio formats."""
        return list(self.supported_formats.keys())
    
    def convert(self, filename):
        """
        Convert audio file to WAV format.
        First attempts conversion with pydub, falls back to ffmpeg if needed.
        """
        try:
            if not os.path.exists(filename):
                print(f"Error: File {filename} not found!")
                return None

            file_extension = os.path.splitext(filename)[1].lower()
            
            if file_extension == '.wav':
                return filename
            
            if not self.is_format_supported(file_extension):
                print(f"Error: Format {file_extension} not supported!")
                print("Supported formats:", ", ".join(self.list_supported_formats()))
                return None
                
            print(f"Converting {file_extension} file to WAV format...")
            
            wav_file = os.path.splitext(filename)[0] + ".wav"
            
            try:
                # Try pydub first
                sound = AudioSegment.from_file(filename, format=self.supported_formats[file_extension])
                sound = sound.set_frame_rate(self.sample_rate)
                sound.export(wav_file, format="wav")
            except Exception as e:
                print(f"Conversion with pydub failed, trying ffmpeg... ({str(e)})")
                try:
                    # Fallback to ffmpeg
                    stream = ffmpeg.input(filename)
                    stream = ffmpeg.output(stream, wav_file, acodec='pcm_s16le', ac=1, ar=self.sample_rate)
                    ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
                except Exception as e:
                    print(f"Conversion with ffmpeg also failed: {str(e)}")
                    return None
            
            if os.path.exists(wav_file):
                print(f"Conversion successful: {wav_file}")
                return wav_file
            else:
                print("Conversion failed: WAV file could not be created")
                return None
            
        except Exception as e:
            print(f"Conversion error: {str(e)}")
            return None
