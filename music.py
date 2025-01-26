import math

import cv2
import mediapipe as mp
import numpy as np
from pedalboard import Distortion, Pedalboard
from pydub import AudioSegment
from pydub.playback import play

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# --- Audio Setup ---
# Replace "your_song.mp3" with your audio file
audio = AudioSegment.from_file("./audio.mp3")

# Convert to float32 for pedalboard compatibility
audio_float32 = audio.set_sample_width(4).set_frame_rate(
    audio.frame_rate).set_channels(audio.channels)
# Get raw audio data as bytes
raw_audio = audio_float32.raw_data
# Convert raw audio data to a NumPy array with float32 data type
raw_audio_np = np.frombuffer(raw_audio, dtype=np.float32)

sample_rate = audio.frame_rate
channels = audio.channels

# Create pedalboard with Distortion effect
board = Pedalboard([
    Distortion(drive_db=20)  # Adjust drive_db for desired distortion level
])

# Process audio in chunks (for efficiency)
chunk_size = 1024
processed_audio = []

for i in range(0, len(raw_audio_np), chunk_size):
    chunk = raw_audio_np[i:i + chunk_size]

    # Pad the last chunk with zeros if it's smaller than chunk_size
    if len(chunk) < chunk_size:
        padding = np.zeros(chunk_size - len(chunk), dtype=np.float32)
        chunk = np.concatenate((chunk, padding))

    # Apply the effect
    effected_chunk = board(chunk, sample_rate)
    processed_audio.append(effected_chunk)

# Concatenate processed chunks
processed_audio_np = np.concatenate(processed_audio)

# Convert back to AudioSegment for playback
effected_audio_segment = AudioSegment(
    processed_audio_np.tobytes(),
    frame_rate=sample_rate,
    sample_width=audio_float32.sample_width,
    channels=channels
)

# Play the distorted audio
play(effected_audio_segment)
