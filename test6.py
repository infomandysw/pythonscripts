# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:20:15 2023

@author: Mandy
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 12:20:14 2023

@author: Mandy
"""

import pydub;
import numpy;
import matplotlib.pyplot as pp;
import os;

# Convert MIDI note to frequency in Hz.
def noteToFreq(note):
    return 440 * 2 ** ((note - 69) / 12);

def keybd(ns, white_key = 0.05, black_key = 0.95):
    mod = numpy.round(ns) % 12;
    pic = numpy.zeros_like(ns);
    
    pic[:] = white_key;
    
    pic[mod == 1] = black_key;
    pic[mod == 3] = black_key;
    pic[mod == 6] = black_key;
    pic[mod == 8] = black_key;
    pic[mod == 10] = black_key;
    
    return pic;

# Input sound file for doing math on.
seg = pydub.AudioSegment.from_file("violin-note.mp3");
samples = numpy.array(seg.get_array_of_samples());
samples = samples.reshape((-1, seg.channels));
sampleRate = seg.frame_rate;

noteList = numpy.arange(24, 108, 0.1);
freqList = noteToFreq(noteList);
freqList = freqList.reshape((-1, 1));

dt = 1.0 / sampleRate;
ts = numpy.arange(0, 8192 * dt, dt);
ts = ts.reshape((1, -1));
phaseList = 2 * numpy.pi * freqList * ts;
shiftFactor = numpy.exp(-1j * phaseList);

windowSamples = samples[0:ts.shape[1], 0];
windowSamples = windowSamples.reshape((1, -1));

product = windowSamples * shiftFactor;
freqCoef = product.sum(axis = 1);
freqPow = numpy.absolute(freqCoef) ** 2;

keyboard = keybd(noteList);

pp.figure();
pp.title("Spectrum");
pp.xlabel("Note (MIDI number)");
pp.ylabel("Power");
pp.plot(noteList, keyboard, label = "Piano Keys");
pp.plot(noteList, freqPow / freqPow.max(), \
        label = "Signal Power");
pp.legend(loc = "best");
pp.savefig("spectrum.png", dpi = 300);
pp.close();

