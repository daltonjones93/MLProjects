import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import math

import librosa
import librosa.display
import numpy as np
import pygame
from pydub import AudioSegment
from pydub.playback import play
import soundfile as sf




def plot_audio_spectrum(mp3_file_path):
    # Load the audio file
    y, sr = librosa.load(mp3_file_path)

    # Compute the spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    
    

    # Plot the spectrogram
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram of the Audio Signal')
    plt.show()
    return D

# Example usage:
mp3_file_path = 'ethnic-violin-64897.mp3'


def find_strongest_pitch(spectrogram, sr):
    # Compute the Harmonic Product Spectrum (HPS)
    hps = np.prod(np.abs(spectrogram[:, ::2]), axis=1)

    # Find the index of the maximum value in the HPS
    max_index = np.argmax(hps)

    # Convert the index to frequency (Hz)
    strongest_pitch_hz = librosa.core.mel_frequencies(n_mels=len(hps), fmax=sr/2.0)[max_index]

    return strongest_pitch_hz

def plot_spectrogram_and_pitch(mp3_file_path):
    # Load the audio file
    y, sr = librosa.load(mp3_file_path)

    # Compute the spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    # Plot the spectrogram
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram of the Audio Signal')

    # Find the strongest pitch
    strongest_pitch = find_strongest_pitch(D, sr, hop_length=512)
    print(f"The strongest pitch is approximately {strongest_pitch:.2f} Hz")

    plt.show()



def hertz_to_note(frequency):
    # Calculate the number of half steps from A4
    n = 12 * math.log2(frequency / 440.0)

    # Round to the nearest integer
    n = round(n)

    # Define the note names
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Calculate the octave and note
    octave = n // 12
    note_index = n % 12
    note = note_names[note_index]

    return f"{note}{octave}"

# Example usage:

y, sr = librosa.load(mp3_file_path)
    # Compute the spectrogram
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
frequency = find_strongest_pitch(D, sr)
note = hertz_to_note(frequency)
print(f"The musical note for {frequency} Hz is {note}")



def generate_note(note_number, duration=1.0, output_file="output_note.wav"):
    # Calculate the frequency of the note
    frequency = librosa.midi_to_hz(note_number)

    # Generate a sine wave for the note
    t = np.arange(0, duration, 1/44100)  # Sampling rate of 44100 Hz
    note_wave = 0.5 * np.sin(2 * np.pi * frequency * t)

    # Save the note to a WAV file
    # librosa.output.write_wav(output_file, note_wave, 44100)
    sf.write(output_file, note_wave, 48000, 'PCM_24')


def play_note(note_number):
    # Generate and save the note
    output_file = f"note_{note_number}.wav"
    generate_note(note_number, output_file=output_file)

    # Load the WAV file and play it
    sound = AudioSegment.from_wav(output_file)
    play(sound)

    return output_file

# Example usage:
note_number = 37  # C#2
output_file = play_note(note_number)
print(f"The note has been played and saved to {output_file}")




