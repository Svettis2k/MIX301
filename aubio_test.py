import aubio
import numpy as np
import pyaudio
from math import log


A4 = 440
C0 = A4 * pow(2, -4.75)
notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def frequency_to_note(frequency):
    if frequency == 0 or frequency is None:
        return "No note"
    h = round(12 * log(frequency / C0, 2))
    octave = h // 12
    index = h % 12
    return notes[int(index)] + str(int(octave))


def setup_detection():
    # PyAudio object.
    p = pyaudio.PyAudio()

    # Open stream.
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=44100, input=True, frames_per_buffer=1024)

    # Aubio's pitch detection.
    p_detection = aubio.pitch("default", 2048, 2048//2, 44100)

    # Set unit.
    p_detection.set_unit("Hz")
    p_detection.set_silence(-40)

    return stream, p_detection


def start_pitch_detection(stream, p_detection):
    data = stream.read(1024)
    samples = np.fromstring(data, dtype=aubio.float_type)
    pitch = p_detection(samples)[0]

    # Compute the energy (volume) of the current frame.
    volume = np.sum(samples ** 2) / len(samples)

    # Format the volume output so that at most it has six decimal numbers.
    volume = "{:.6f}".format(volume)

    note = str(frequency_to_note(pitch))
    pitch = str(pitch)
    volume = str(volume)

    print("{\"note\": \"" + note + "\", \"pitch\": " + pitch + ", \"volume\": " + volume + "}")


if __name__ == "__main__":
    input_stream, pitch_detection = setup_detection()
    while True:
        start_pitch_detection(input_stream, pitch_detection)
