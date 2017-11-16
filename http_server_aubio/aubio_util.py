import aubio
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
