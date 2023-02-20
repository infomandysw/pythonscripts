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

class NoteFt:
    def __init__(self,
                 sampleRate = 44100,
                 smoothing = 4096,
                 noteStep = 0.1,
                 noteMin = 24,
                 noteStop = 120):

        self.sampleRate = sampleRate;
        self.smoothing = smoothing;
        self.noteStep = noteStep;
        self.noteMin = noteMin;
        self.noteStop = noteStop;
        
        self.dt = 1 / sampleRate;
        
        self.noteList = numpy.arange( \
            noteMin, noteStop, noteStep, \
            dtype = numpy.float32);
        self.freqList = noteToFreq(self.noteList);
        self.phaseList = numpy.zeros( \
            (self.freqList.size,), \
            dtype = numpy.float32);
        
        self.resultSamples = numpy.zeros( \
            (self.freqList.size, smoothing), \
            dtype = numpy.complex128);
        self.resultCounts = numpy.zeros( \
            (self.freqList.size,), \
            dtype = numpy.int32);
        self.resultFills = numpy.zeros( \
            (self.freqList.size,), \
            dtype = numpy.int32);
            
    def write(self, samples):
        for i in range(0, self.freqList.size):
            f = self.freqList[i];
            phi = self.phaseList[i];
            count = self.resultCounts[i];
            
            for j in range(0, len(samples)):
                self.resultSamples[i, count] = \
                    samples[j] * numpy.exp(-1j * phi);
                    
                phi += 2 * numpy.pi * f * self.dt;
                count += 1;
                
                if phi >= 2 * numpy.pi:
                    phi -= 2 * numpy.pi;
                    
                if count >= self.smoothing:
                    self.resultFills[i] += 1;
                    count = 0;
                    
            self.phaseList[i] = phi;
            self.resultCounts[i] = count;
            
    def result(self):
        resultSum = ( \
            self.resultSamples / \
            self.freqList.reshape(-1, 1) \
        ).sum(axis = 1);
        
        resultSum[self.resultFills == 0] = 0;
        
        return numpy.absolute(resultSum * \
                              resultSum.conj());

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

ft = NoteFt(sampleRate = sampleRate, smoothing = 8192);
ft.write(samples[0:ft.smoothing]);
freqPow = ft.result();

keyboard = keybd(ft.noteList);

pp.figure();
pp.title("Spectrum");
pp.xlabel("Note (MIDI number)");
pp.ylabel("Power");
pp.plot(ft.noteList, keyboard, label = "Piano Keys");
pp.plot(ft.noteList, freqPow / freqPow.max(), \
        label = "Signal Power");
pp.legend(loc = "best");
pp.savefig("spectrum.png", dpi = 300);
pp.close();

