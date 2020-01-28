#TODO: check if we can make make_pattern faster... (splitting
#sound_lib to dedicated libs per channels has no effect)

#TODO: command 2 does not yet respond quite accurate enough on tempo/speed

"""Modtracker for Python:

What is does:
-------------

ModTracker is a player of mod tracks (old Amiga music format) loaded
from disk or inline mod-data. You can load instruments/samples from
disk or synthesize them in code.

Modtracker for Python is based on ProTracker.

- All basic effect except E-effects are implemented. (Effect B never
  found and not tested!)
- Read and play 4-channel mod files in the Ultimate Soundtracker or
  Protracker format.
- Load and save in native format (see next bullets for differences)
- Specify track-info inline as a array (python list)
- Load samples (sounds) from wav files (unsigned 8 bit int, signed 32
  bit float, signed 16 bit int, signed 32 bit int)
- Specify samples as a waveform-array (python list)
- Modtracker does not have a position table to specify a sequence of
  patterns to play. You can simulate this in your own python code.

How it works:
-------------
Python is not fast enough to synthesize sounds on-the-fly (not using
external libraries in e.g. C).  So Modtracker identifies and
synthesizes all unique sounds beforehand and stores this in a
library/dictionary.  For each channel a playlist of references to
these sounds is build. This minimizes the needed memory.  Python uses
pygame to play the sounds and numpy to synthesize the sounds with the
effects.

How to use it:
--------------
The basic framework is:
  1) Import modtrack              - from modtrack import tracker
  2) Init modtrack                - tracker.init
  3) Load or specify your song    - tracker.load / tracker.load_amigamodule / song array=[[--- --- 00 000 000],[--- --- 00 000 000]]
  4) load samples (optionally)    - wav2sample / tracker.custom_waveform
  5) Synthesize the song          - tracker.make_pattern
  6) Play the song                - tracker.play_pattern

A song array (python list) consists of:

  - rows to be played one after another,
  - and each row consists of 4 channels,
  - each channel has a sequence either in amiga format or in
    (expanded) native format
  - each channel has a sequence consisting of note1, (note2), sample
    number, effect1 (effect2)

  amiga_song= [
                ["C-3 04 000", "--- 00 000", "--- 00 000", "--- 00 000"],
                ["--- 00 000", "--- 00 000", "--- 00 000", "--- 00 000"],
              ]
  native_song=[
                ["C-3 --- 04 000 000", "--- --- 00 000 000", "--- --- 00 000 000", "--- --- 00 000 000"],
                ["--- --- 00 000 000", "--- --- 00 000 000", "--- --- 00 000 000", "--- --- 00 000 000"]
                ["D-3 --- 04 000 000", "--- --- 00 000 000", "--- --- 00 000 000", "--- --- 00 000 000"]
              ]

  - A note is 3 positions long, starting with 'A'-'G', followed by -
    or # and last the octave number

  - Sample numbers are 2 positions long, in hex and start at 01

  - Effects are 3 positions long, start with effect number 'A'-'F',
    followed by the effect value (2 positions) in hex

  - A note(group) starts if note1 is not --- and ends on the next row
    where note2 is not ---

  - Instrument is only needed in first row of note-group.

  - If the first row of the group has note1 and note2 specified, they
    are both played and all effects are applied to both of them.

  - If a consequetive row has note2 (but not note1 off course), this
    note2 is not played but interpreted as an argument of the effect
    on that row

  - Effects don't have to span the entire group and effect1 and
    effect2 can start/stop at different rows within the group,

Samples (instruments) can be internal, wav files or custom waveforms

  o Default samples/instruments are:
      1: tri, 2: saw, 3: pulse, 4:sin, 5:whitenoise,

  o If you load a sample from disk, this sample should have a frequency of C-3 (261Hz)

  o A custom sample from a waveform shape is made with a two
  dimensional array:

      triangle_waveform= ["  /\  ",
                          " /  \ ",
                          "/    \"]
      You can use every character you like to specify it. Don't put multiple characters in the same column.

All user-commands:
    init (volume, resolution, flags) - init pygame engine and window at desired resolution/flags and sets master volume
    songtitle                        - title of song

    clear (pytfilename)              - start new song, clear samples, pattern and sets filename for song
    load (pytfilename) 		         - load song in native Modtrack format
    save (pytfilename) 		         - save song in native Modtrack format
    load_amigamodule	(modfile)    - load song in Ultimate Soundtracker and ProTracker format
    custom_waveform (usr_waveform_array, volume, samplenr, name)
                                     - converts a wavefrom array and returns a sample (optionally set at samplenr)

    master_volume                    - must be set before make-pattern, can also be set from init()

    get_play_pos()                   - returns playing position in msecs
    get_play_row()                   - converts msecs in row nr in pattern

    abort_play                       - stops play at next row
    play_pattern(soundrefs,from_time)- play_pattern (soundrefs is optionally) from given time (optionally)


Effect commands - In song array the following effects can be used:
    0 - Normal play or Arpeggio             0xy : x-first halfnote add, y-second
    1 - Slide Up                            1xx : upspeed
    2 - Slide Down                          2xx : downspeed
    3 - Tone Portamento                     3xx : up/down speed
    4 - Vibrato                             4xy : x-speed,   y-depth
    5 - Tone Portamento + Volume Slide      5xy : x-upspeed, y-downspeed
    6 - Vibrato + Volume Slide              6xy : x-upspeed, y-downspeed
    7 - Tremolo                             7xy : x-speed,   y-depth
    9 - Set SampleOffset                    9xx : offset (23 -> 2300)
    A - VolumeSlide                         Axy : x-upspeed, y-downspeed
    B - Position Jump                       Bxx : go to start of pattern at position xx in position list
    C - Set Volume                          Cxx : volume, 00-40
    D - Pattern Break                       Dxx : go to row xx of next pattern in position list
    B+D - on same row                       Bxx Dyy: go to row yy of pattern at pos xx in pos list
    F - Set Speed___________________________Fxx : speed (00-1F) / tempo (20-FF)


    NOT IMPLEMENTED:
    E9- Retrig Note                         E9x : retrig from note + x vblanks
    E00/1=filter on/off - E1x/2x=FineSlide Up/Down - E30/1=tonep ctrl off/on
    E40/1/2=Vib Waveform sine/rampdown/square, E70/1/2=Tremolo Waveform

    E5x=set loop point,E6x=jump to loop+play x times

    EAx/EBx=Fine volslide up/down
    ECx/EDx=notecut after x vblanks/notedelay by x vblanks
    EEx/EFx=PatternDelay by x notes/Invert loop, x=speed

    Remarks:
    - B and D are not necessary in native format. In ModTrack you can (should) just make one complete pattern.
    - In native track format (two effects per sequence) the effect commands 5,6 are not necessary since they are combinations of other effects:
        5 -> 3, A
        6 -> 4, A
    - Amiga files use patterns of 64 rows and the F00 command is used to stop the song before a 64-row block
      In ModTrack a song can be of any number of rows, so to stop the song just don't add any rows after the last note.

    Internally patterns in amiga format will be rewritten to native format:
        Example of second note as chord (C-3 E-3) and effect parameter (D-3)
            C-3 E-3 00 102 000
            --- --- 00 102 A02
            --- D-3 00 302 A02
            --- --- 00 302 A02

        Compact sequences like in Protracker are expanded as follows:
            ProTracker  ->  ModTrack
            C-3 00 000      C-3 --- 00 000 000
            E-3 00 302      --- E-3 00 302 000
            ...
            C-3 00 000      C-3 --- 00 000 000
            E-3 00 302      --- E-3 00 302 000
            --- 00 513      --- --- 00 302 A13

        The effectcombo's 5 and 6 are rewritten to seperate effects:
            ProTracker  ->  ModTracker
            C-3 01 000      C-3 --- --- 01 000 000
            G-3 01 343      --- G-3 --- 01 343 000
            --- 01 502      --- --- --- 01 343 A02

            C-3 01 000      C-3 --- --- 01 000 000
            G-3 01 443      --- G-3 --- 01 443 000
            --- 01 502      --- --- --- 01 443 A02

"""

import pygame
from pygame.locals import *
import math
from scipy.io import wavfile
import numpy as np

from threading import Thread
import os

from copy import deepcopy
from sys import exit
from time import sleep, time
from pygame.mixer import Channel, get_busy, get_num_channels, set_num_channels
from pygame.sndarray import make_sound

class DebugPrint:
    def __init__(self, enabled):
        self.indent = 0
        self.enabled = enabled

    def print_indented(self, text):
        if self.enabled:
            print(' ' * self.indent + text)

    def header(self, name, fmt, args):
        self.print_indented('* %s %s' % (name, fmt % args))
        self.indent += 2

    def print(self, fmt, args):
        self.print_indented(fmt % args)

    def leave(self):
        self.indent -= 2

DP = DebugPrint(False)

#################################################################
# SET ALL TRACKER VARIABLES AND SOME MUTATION METHODS
#################################################################

sample_rate = 44100
SAMPLE_RATE = 44100
bits = 16

# from 0x00 to 0xFF; sometime the volume of the track is too loud, so
# this var helps you scale it down

MASTER_VOLUME = 0x40

# This is the volume parameter which is set by the track
TRACK_VOLUME = 0x20

# Enabled channels
ENABLED_CHANNELS = [1]
#ENABLED_CHANNELS = list(range(4))

def init(resolution, depth=0):
    """
    Initializes the pygame environment for audio.
    :param volume: Master-volume (0-255)
    :param resolution: Screen resolution for the pygame window
    :param flags: window flags
    :param depth: window depth
    """
    global size, bits
    size = resolution
    pygame.mixer.pre_init(SAMPLE_RATE, -bits, 1)
    pygame.init()
    display_surf = pygame.display.set_mode(resolution)
    return display_surf

tempo=0x7D
speed=0x06

notelist = [
    "C-", "C#", "D-", "D#", "E-", "F-",
    "F#", "G-", "G#", "A-", "A#", "B-"
]
FREQUENCIES = [
    4186.01, 4434.92, 4698.63, 4978.03, 5274.04, 5587.65,
    5919.91, 6271.93, 6644.88, 7040.00, 7458.62, 7902.13
]

FREQS = {'---': 0}

# Construct notes of all octaves
for doct in range(8):
    for note, freq8 in zip(notelist, FREQUENCIES):
        oct = (8-doct)
        notename = note+str(oct)
        notefreq = freq8/2**doct
        FREQS[notename] = notefreq

BASE_FREQ = FREQS['C-3']
octave_transpose=0

def nth_halfnote(note,n):
    """
    Takes note in format 'C#3', returns n-th note in same format
    The n in n-th-note can be positive or negative.
    """
    for notenr, notename in enumerate(notelist):
        if notename == note[:2]:
            next_notenr = notenr+n
            next_oct = int(note[2])
            if next_notenr < 0:
              next_oct=next_oct-1
              next_notenr=next_notenr+12
            if next_notenr >= 12:
              next_oct = next_oct+1
              next_notenr = next_notenr-12
            next_notename = notelist[next_notenr] + str(next_oct)
            return next_notename
    return "ERR"

def nth_fullnote(note,n):
    return nth_halfnote(note, n * 2)

def prev_fullnote(note):
    return nth_fullnote(note,-1)

def next_fullnote(note):
    return nth_fullnote(note,1)


#################################################################
#SET DEFAULT SAMPLES
#################################################################
# fill with some example waveforms of 3 sec each @ 44100,
# but could/should be filled from files
def saw_sample(sample,nr_samples,amp):
    return int((-1+2 * sample/(nr_samples-1)) * amp)
def tri_sample(sample,nr_samples,amp):
    if sample<nr_samples/2:
       return int((-1 + 4 * sample / (nr_samples-1)) * amp)
    else:
       sampleR=nr_samples/2-(sample-nr_samples/2)-1
       return int((-1 + 4 * sampleR / (nr_samples - 1)) * amp)
def pulse_sample(sample,nr_samples,amp):
    if sample < nr_samples / 2:
        return amp
    else:
        return -amp
def sin_sample(sample,nr_samples,amp):
    return int(amp*math.sin(2*math.pi*sample/nr_samples))
def whitenoise_sample(sample,nr_samples,amp):
    return np.random.randint(-amp,amp,1,np.int16)


def init_samples():
    """Adds some basic waveforms to the (empty samples) list, like
    saw(tooth), tri(angle), pulse, sin(us) and empy wave."""
    nr_samples = int(SAMPLE_RATE / BASE_FREQ)
    waves_in_3secs = int(3 * BASE_FREQ)
    amp = 255 * 32
    saw = np.zeros(nr_samples,np.int16)
    tri = np.zeros(nr_samples,np.int16)
    pulse = np.zeros(nr_samples,np.int16)
    sin = np.zeros(nr_samples,np.int16)
    empty=np.zeros(nr_samples,np.int16)
    for sample in range(0, nr_samples):
        saw[sample] = saw_sample(sample,nr_samples,amp)
        tri[sample] = tri_sample(sample, nr_samples, amp)
        pulse[sample] = pulse_sample(sample, nr_samples, amp)
        sin[sample] = sin_sample(sample, nr_samples, amp)

    saw=np.tile(saw,waves_in_3secs)
    tri=np.tile(tri,waves_in_3secs)
    pulse=np.tile(pulse,waves_in_3secs)
    sin=np.tile(sin,waves_in_3secs)
    #noise should be generated with as much unique random samples as possible.
    whitenoise=np.random.randint(-amp,amp,nr_samples*waves_in_3secs,np.int16)

    empty=np.tile(empty,waves_in_3secs)

    global samples
    samples = [
        {"name":"tri",
         "filename": "internal",
         "volume":"FF",
         "repeat_from":0,
         "repeat_len":len(tri),
         "len":len(tri),
         "data":tri},
        {"name":"saw",
         "filename": "internal",
         "volume":"FF",
         "repeat_from":0,
         "repeat_len":len(saw),
         "len":len(saw),
         "data":saw},
        {"name":"pulse",
         "filename": "internal",
         "volume":"FF",
         "repeat_from":0,
         "repeat_len":len(pulse),
         "len":len(pulse),
         "data":pulse},
        {"name":"sin",
         "filename": "internal",
         "volume":"FF",
         "repeat_from":0,
         "repeat_len":len(sin),
         "len":len(sin),
         "data":sin},
        {"name":"whitenoise",
         "filename": "internal",
         "volume":"FF",
         "repeat_from":0,
         "repeat_len":len(whitenoise),
         "len":len(whitenoise),
         "data":whitenoise}
    ]

init_samples()

#################################################################
#SET DEFAULT PATTERN
#################################################################
# a pattern is 64 rows (positions) and 4 cols (tracks/channels)
# format Note(#) Sharp/Flat(#) Octave(#) Instrument(##) Effect(###)
filename=""
songtitle="demo"

#simplepattern = "C2- D2-- E2- F2--"

def custom_waveform(usr_waveform_array,volume=0x40,samplenr=-1,name=None):
    nr_samples = int(SAMPLE_RATE / BASE_FREQ)

    usr_array=usr_waveform_array
    usr_width=len(usr_array[0])
    usr_height=len(usr_array)
    waveform=np.zeros([usr_width],np.int16)
    for x in range(usr_width):
        for y in range(len(usr_array)):
            if not usr_array[y][x]==' ': waveform[x]=(y/(usr_height-1)-0.5)*256*255*volume/0x40

    waveform=np.array(waveform,np.int16)

    x_from=np.linspace(0,nr_samples,usr_width)
    x_to=np.linspace(0,nr_samples,nr_samples)

    full_waveform=np.interp(x_to,x_from,waveform)

    data = np.clip(full_waveform,-32768, 32767)
    data = data.astype(np.int16)

    np.set_printoptions(precision=3,
                           threshold=40,
                           edgeitems=40, suppress=True)
    print (np.min(data),np.max(data),\
          (np.min(data)+np.max(data))/2)
    print (data.size,data)
    np.set_printoptions(precision=3,
                           threshold=8,
                           edgeitems=3, suppress=True)
    #quit()
    if name==None:
        name="usr_{:02d}".format(samplenr-1)
    filename=name+".wav"
    sample={"name": name,
            "filename": filename,
            "volume": "FF",
            'repeat_from': 0,
            "repeat_len": len(data),
            "len": len(data),
            "data": data }

    if not samplenr == -1:
        samples[samplenr - 1] = sample
    return sample

pattern = [
    ["C-2 01 000", "--- 00 000", "--- 00 000", "--- 00 000"],
    ["--- 00 000", "--- 00 000", "--- 00 000", "--- 00 000"],
    ["D-2 01 000", "--- 02 000", "--- 00 000", "--- 00 000"],
    ["--- 00 000", "--- 00 000", "--- 00 000", "--- 00 000"],
    ["--- 00 C00", "E-2 02 000", "--- 00 000", "--- 00 000"],
    ["--- 00 000", "--- 00 000", "--- 00 000", "--- 00 000"],
    ["--- 00 000", "F-2 02 000", "--- 00 000", "--- 00 000"],
    ["--- 00 000", "--- 00 000", "--- 00 000", "--- 00 000"],
    ["--- 00 000", "--- 00 C00", "G-2 03 000", "--- 00 000"],
    ["--- 00 000", "--- 00 000", "--- 00 000", "--- 00 000"],
    ["--- 00 000", "--- 00 000", "A-2 03 000", "--- 00 000"],
    ["--- 00 000", "--- 00 000", "--- 00 000", "--- 00 000"],
    ["--- 00 000", "--- 00 000", "--- 00 C00", "B-2 04 000"],
    ["--- 00 000", "--- 00 000", "--- 00 000", "--- 00 000"],
    ["--- 00 000", "--- 00 000", "--- 00 000", "C-3 04 000"],
    ["--- 00 000", "--- 00 000", "--- 00 000", "--- 00 000"],
    ["--- 00 000", "--- 00 000", "--- 00 000", "--- 00 C00"],
    ]

#################################################################
#INIT GLOBAL soundrefs
#################################################################
soundrefs       = 'Nard'
rowtimings      =  None
rowstarttimings =  None
#################################################################
#LOAD EXTERNAL SAMPLES/TRACKS
#################################################################

def resample(data, fps):
    """Resamples audio data (data, 16 bit ints) from sample-rate given by
    fps to sample-rate in global variable sample_rate"""
    sample_ratio = SAMPLE_RATE / fps
    base_samples = data.size
    nr_samples = int(base_samples * sample_ratio)

    # construct x-points for wform and resulting/interpolated wave
    x_data = np.linspace(0, 1, data.size)
    x_resampled = np.linspace(0, 1, nr_samples)

    resampled_data = np.interp(x_resampled, x_data, data).astype(np.int16)
    return resampled_data

def import_amigasample(bytes):
    """Converts a byte array from a mod-file (bytes; sample
    rate 16574.27) to audio data at sample-rate of global variable
    sample_rate
    """
    fps = 16574.27
    sample_ratio = SAMPLE_RATE / fps

    if len(bytes) == 0:
        # Need some data so function which process audio data won't
        # break.
        data_16bit_sample_rate = np.zeros(8, np.int16)
    else:
        # first we convert to int8 to convert raw data to signed 8 bit
        # (not only a shift is neccesary but also a mirroring of upper
        # part).
        data_8bit_8287Hz = np.array(bytes, dtype = np.int8)
        #data_8bit_8287Hz = np.array(bytes).astype(np.int8)
        #store in 32bit to provide enough room for scaling
        data_16bit_8287Hz = data_8bit_8287Hz.astype(np.int32) * 256
        # clip to 16bit boundaries and fit in 16 bit array
        data_16bit_8287Hz = np.clip(data_16bit_8287Hz, -32768, 32767)
        data_16bit_8287Hz = data_16bit_8287Hz.astype(np.int16)
        # Resample from amiga framerate (fps) to correct internal
        # framerate.
        data_16bit_sample_rate = resample(data_16bit_8287Hz, fps)

    return data_16bit_sample_rate, sample_ratio

def load_amigamodule(modfile):
    """
    Loads an amiga mod tracker file and returns the track text data
    """
    global samples, pattern, filename, songtitle
    import modtrack.loadmod as loadmod
    d = False

    #load module
    ret = loadmod.read_module(modfile)
    if not ret:
        return False

    #copy info to local vars
    filename = modfile
    songtitle = loadmod.songtitle

    samples = loadmod.samples
    for i, sample in enumerate(samples):
        name = sample['name']
        volume = sample['volume']
        amiga_data = sample['data']
        repeat_from = sample['repeat_from']
        repeat_len = sample["repeat_len"]
        lendata = sample["len"]
        if d:
            print("#, name, volume, repeat (f, t), len, len imported:",
                  i, name, volume, "(", repeat_from, repeat_len, ")",
                  lendata, len(amiga_data),amiga_data)
        sample_data, sample_ratio = import_amigasample(amiga_data)
        if repeat_from > 0:
            # Default amiga = 0 on no repeat
            repeat_from = int(repeat_from * sample_ratio)
        if repeat_len > 2:
            # Default amiga = 0 on no repeat
            repeat_len = int(repeat_len  * sample_ratio)
        lendata = sample_data.size
        if d:
            print("#, name, volume, repeat (f,t), len, len imported:",
                  i, name, volume, "(", repeat_from, repeat_len, ")",
                  lendata, sample_data.size,sample_data)

        sample["name"]        = name
        sample["filename"]    = "modfile"
        sample["volume"]      = volume
        sample['data']        = sample_data
        sample['repeat_from'] = repeat_from
        sample["repeat_len"]  = repeat_len
        sample["len"]         = lendata

    #find last pattern in table which is not 0
    if d:
        print ("table-size:", len(loadmod.pattern_table))
    pattern = []
    pat_row_nrs=[]
    new_pat_start=0
    for pos_nr in range(len(loadmod.pattern_table)):
        pat_nr = loadmod.pattern_table[pos_nr]
        # we need to make a copy because we edit it if B or D present
        pat_txt = deepcopy(loadmod.patterns[pat_nr])
        nrs = list(range(64))
        row_nrs = [str(pos_nr)+" -> "+str(pat_nr)+":"+str(s)+"   "
                   for s in nrs]

        first_row = new_pat_start
        last_row = 64
        new_pos_nr = pos_nr + 1
        new_pat_start = 0

        fnd = False
        for rownr, row in enumerate(pat_txt):
            if fnd:
                break
            if d:
                print ("      row:",rownr, row)
            for ichan, seq in enumerate(row):
                eff_cmd=seq[7:8]
                eff_val=int(seq[8:10],16)
                # B - Position Jump
                # Bxx : go to start of pattern at position xx in position list
                if eff_cmd == "B":
                    print("    B FOUND",seq)
                    pat_txt[rownr][ichan] = seq[0:7] + "000"
                    new_pos_nr = eff_val
                    last_row=rownr+1
                    new_pat_start=0
                    fnd=True
                # D - Pattern Break
                # Dxx : go to row xx of next pattern in position list
                elif eff_cmd == "D":
                    print("    D FOUND", seq)
                    pat_txt[rownr][ichan] = seq[0:7] + "000"

                    last_row = rownr + 1
                    new_pos_nr = pos_nr +1
                    if d:
                        print ("   rownr", rownr)
                        print("    last_row", last_row)
                        print("    new_pos_nr", new_pos_nr)
                        print("    new_pat_start", new_pat_start)
                    fnd = True
        if d:
            print ("first_row:last_row",first_row,last_row)
        pattern = pattern + pat_txt[first_row:last_row]
        pat_row_nrs = pat_row_nrs+row_nrs[first_row:last_row]
        first_row = new_pat_start
        pos_nr = new_pos_nr-1
    return pattern

def clear(pytfilename):
    """Start without any pattern info and resets the (loaded) samples to
    the default initial simple waveform samples.

    :param pytfilename: The filename the composition will have. (No
    saving occurs until save is called.)

    """
    global pattern, filename
    init_samples()
    pattern.clear()
    for i in range(0, 64):
        pattern.append(["--- --- 00 000 000"] * 4)

    filename = pytfilename.replace("\\", "/")

def load(pytfilename):
    """
    Loads a composition which was saved with the save() function and therefore
    in the internal format consisting of a python file and a subdirectory with all used samples.
    :param pytfilename: Filename to load (should have .pyt extension.)
    """
    # replace \ with / to prevent unintended escapes in string
    pytfilename = pytfilename.replace("\\", "/")

    #doload
    with open(pytfilename) as f:
        #content = f.readlines()
        content = f.read()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    #content = [x.strip() for x in content]

    exec (content)
    filename=pytfilename
    return

def save(pytfilename):
    """Saves a composition in the internal format consisting of a python
    file and a subdirectory with all used samples.

    :param pytfilename: Filename to save (should have .pyt extension.)
    """
    # replace \ with / to prevent unintended escapes in string
    pytfilename = pytfilename.replace("\\", "/")
    global filename
    filename = pytfilename #full filename

    #save internal sample data if needed
    bare_filename = os.path.basename(pytfilename)
    bare_filetitle= os.path.splitext(bare_filename)[0]
    bare_path = os.path.dirname(pytfilename)
    sample_path = bare_path + "/" + bare_filetitle + ".samples"
    # Make subdir if neccessary
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    for nr, sample in enumerate(samples):
        sample_filename = sample["filename"]
        sample_data = sample['data']
        sample_name = sample["name"]
        # Don't save empty samples.
        if sample_filename in ("internal", "modfile") and sample_name:
            sample_filename = sample_path + "/" + sample_name + ".wav"
            #save data
            wavfile.write(sample_filename, sample_rate, sample_data)
            #reset filename
            sample["filename"] = sample_filename

    #open pyt file and save meta data and track data
    pytfile = open(pytfilename, 'w')

    pytfile.write("import tracker\n\n")
    pytfile.write("tracker.filename  = '{}'\n".format(pytfilename))
    pytfile.write("tracker.songtitle = '%s'\n" % songtitle)
    pytfile.write("\n")

    #save meta info of samples
    lastsample = len(samples)-1
    pytfile.write     ("tracker.samples = [\n")
    for nr,sample in enumerate(samples):
        pytfile.write ("                     {\n")
        pytfile.write ("                        'name'        :'{}',\n"\
                       "                        'filename'    :'{}',\n"\
                       "                        'volume'      :'{}',\n"\
                       "                        'repeat_from' :{},\n"\
                       "                        'repeat_len'  :{},\n"\
                       "                        'len'         :{}\n".format(
                                                sample['name'],
                                                sample['filename'],
                                                sample['volume'],
                                                sample['repeat_from'],
                                                sample['repeat_len'],
                                                sample['len']
                                                )
                       )
        pytfile.write ("                      }")
        if nr < lastsample:
            pytfile.write(",\n")
    pytfile.write ("\n                    ]\n")
    pytfile.write("\n")

    #pattern data
    lastrow=len(pattern)-1
    pytfile.write("tracker.pattern = [\n")
    for nr,row in enumerate(pattern):
        pytfile.write("                 %s" % row)
        if nr<lastrow:
            pytfile.write(",\n")
        pass
    pytfile.write("\n               ]\n\n")

    #load sample wav files
    pytfile.write("from scipy.io import wavfile\n")
    pytfile.write("import np\n")
    pytfile.write("for sample in tracker.samples:\n")
    pytfile.write("    filename=sample['filename']\n")
    pytfile.write("    if not filename in ('','internal','modfile'):\n")
    pytfile.write("        fps, data = wavfile.read(filename)\n")
    pytfile.write("        if np.result_type(data )==np.float32:\n")
    pytfile.write("            data = data * (256* 128-1)\n")
    pytfile.write("            data = data.astype(np.int16)\n")
    pytfile.write("        resampled_data = tracker.resample(data,fps)\n")
    pytfile.write("        sample['data']=resampled_data\n")
    pytfile.write("\n")
    pytfile.write("print ('{} loaded!')\n".format(pytfilename))

    #close
    pytfile.close()
    print("Saved:'" + pytfilename + "'")
    return

#################################################################
#TRACKER METHODS
#################################################################

def speed_and_tempo_to_msec(speed_hex, tempo_hex):
    """Returns the duration of a tracker row (in msec) given the setting
    for speed and tempo.
    """
    # all F commands <0x20 are speed, >=0x20 is tempo
    # http://modarchive.org/forums/index.php?topic=2709.0
    # speed: number of 50Hz/20ms ticks before next row (default 6)
    # beat: 4 rows per beat
    # tempo: bmp / number of beats per minute (default 125)
    # 60 rows:
    #  21,6 sec: F12 F7D
    #  10,9 sec: F09 F7D
    #   7,2 sec: F06 F7D -> 0.12
    #  14,75 sec: F06 F3D
    #   4,8 sec: F06 FBD
    tempo = tempo_hex
    speed = speed_hex
    sec_per_beat=60/int(tempo_hex)
    sec_per_row=sec_per_beat/4
    sec_per_row=sec_per_row*int(speed_hex)/6
    msec_per_row=int(sec_per_row*1000)

    return msec_per_row

def stretch_sample_to_freq(freq, arr):
    """Converts the audio data (wfrom) to a specific frequency (freq) by
    stretching the audio data.
    """
    if arr.size < 2:
        return wform

    x_old = np.linspace(0, 1, arr.size)
    x_new = np.linspace(0, 1, int(arr.size * BASE_FREQ / freq))
    arr_new = np.interp(x_new, x_old, arr).astype(np.int16)

    fmt = 'Frequency interpolation %d -> %d (%.2f Hz)'
    DP.print(fmt, (arr.size, arr_new.size, freq))
    return arr_new


def deplop_wave(nwave,fadein_msec,fadeout_msec):
    """
    Reduces the plops at start and end of wave by slowly increasing (fadein_msec) and decreasing (fadeout_msec) the volume
    """
    #finally we fade volume at last 5msec of signal to avoid pop at end
    nr_samples = int(fadein_msec/1000*sample_rate)
    for i in range(0,nr_samples):
        vol=i/nr_samples
        sample=nwave[i]
        nsample=int(sample*vol)
        nwave[i]=nsample
    nr_samples=int(fadeout_msec/1000*sample_rate)
    for i in range(0,nr_samples):
        vol=i/nr_samples
        sample_nr=nwave.size-i-1
        sample=nwave[sample_nr]
        nsample=int(sample*vol)
        nwave[sample_nr]=nsample
    return nwave

def clamp(n, minn, maxn):
    """
    Makes sure the value n will not minn and maxn
    """
    return max(min(maxn, n), minn)

def pad_wave_to_duration(wave, duration):
    """Lengthens/truncs the audio data (wave) to the specified duration
    if necessary by adding zeros."""
    tot_samples = int(duration / 1000 * SAMPLE_RATE)
    #Next pad wave with zeros to match duration
    if wave.size == 0:
        nwave = np.zeros(tot_samples, np.int16)
    elif wave.size < tot_samples:
        pad_len = tot_samples - wave.size
        nwave = np.pad(wave, (0, pad_len),
                       'constant', constant_values=(0, 0))
    else:
        nwave = wave[:tot_samples]

    fmt = 'Padding %d -> %d (%.2f seconds)'
    args = wave.size, nwave.size, nwave.size / SAMPLE_RATE
    DP.print(fmt, args)
    return nwave

def sample_to_wave(sample, note, dur_msecs):
    """Converts a sample (at ref. freq C3) to a wave of the specified
    frequency and duration.

    If repeat-from and repeat-length are specified, this part is
    duplicated enough to achieve to required duration.

    :param sample: Sample dictionary with audio data and other info e.q. repeat-from en -length
    :param note: Target note to achieve
    :param dur_msecs: Target duration
    :return: Audio data

    """
    data = sample["data"]
    repeat_from = sample["repeat_from"]
    repeat_len = sample["repeat_len"]
    datalen = sample["len"]

    freq = FREQS.get(note)
    data2 = stretch_sample_to_freq(freq, data)

    ratio = data2.size / data.size
    repeat_from = int(repeat_from*ratio)  #default amiga if no repeat = 0
    repeat_len  = int(repeat_len*ratio)    #default amiga if no repeat = 2
    datalen=data2.size

    if repeat_from:
        repeat_to = repeat_from + repeat_len
        n_tail = datalen - repeat_to
        n_samples = int(dur_msecs / 1000 * SAMPLE_RATE)
        n_samples_target_body = n_samples - repeat_from - n_tail

        if n_samples_target_body > repeat_len:
            n_repeats = int(n_samples_target_body / repeat_len)
            head = data2[:repeat_from]
            body = np.tile(data2[repeat_from:repeat_to], n_repeats)
            tail = data2[repeat_to:]
            data2 = np.concatenate((head, body, tail))

    return pad_wave_to_duration(data2, dur_msecs)

# For reference see:
#  http://milkytracker.titandemo.org/docs/MilkyTracker.html#fx0xy
#  http://coppershade.org/articles/More!/Topics/Protracker_Effect_Commands/
#  https://www.youtube.com/watch?v=OmeuhvYoij0

def play_wave(arr):
    arr = arr.astype(np.int16)
    snd = make_sound(arr)
    fmt = 'Playing sound with duration %.2f s (%d samples).'
    DP.print(fmt, (snd.get_length(), arr.size))
    Channel(0).queue(snd)
    while get_busy():
        sleep(0.05)

def handle_set_volume(vol, first_sample, cmd_x, cmd_y):
    val = (cmd_x << 4) + cmd_y
    vol_i = val / 64 * MASTER_VOLUME / 0xff;
    vol_p = vol[:first_sample]
    vol_n = np.full(vol.size - first_sample, vol_i)
    return np.append(vol_p, vol_n)

# This is probably not correct
def handle_arpeggio(vol, nwave, note1, first_sample, n_samples, cmd_x, cmd_y):

    x_wave = np.linspace(0, nwave.size, nwave.size)
    note2 = nth_halfnote(note1, cmd_x)
    note3 = nth_halfnote(note1, cmd_y)
    notes = (note1, note2, note3)
    freqs = [FREQS.get(note) for note in notes]
    spaces = [1, freqs[1] / freqs[0], freqs[2] / freqs[0]]

    spaces_str = '%.2f %.2f %.2f' % (spaces[0], spaces[1], spaces[2])
    DP.header('ARPEGGIO', 'spaces [%s], %d samples',
              (spaces_str, n_samples))

    n_cycles = 2
    samples_per_note = n_samples / 3 / n_cycles
    samples_per_space = [s * samples_per_note for s in spaces]

    # Generate six notes
    at = first_sample
    x_samples = []
    for i in range(3 * n_cycles):
        to = int(at + samples_per_space[i % 3])
        x_samples.append(np.linspace(at, to, samples_per_note))
        at = to

    arpeggio_size = 2 * sum(samples_per_space)

    x_pre = np.linspace(0, first_sample, first_sample)
    x_trail = np.linspace(first_sample + arpeggio_size,
                          nwave.size,
                          nwave.size - first_sample - arpeggio_size)
    x_shift = np.concatenate([x_pre] + x_samples + [x_trail])
    x_shift = deplop_wave(x_shift, 0, 5)

    x_pad = np.full(int(arpeggio_size - n_samples), 0)
    x_shift = np.append(x_shift, x_pad)

    DP.print('pre/arp/post/pad %d/%d/%d/%d', (x_pre.size,
                                              arpeggio_size,
                                              x_trail.size,
                                              x_pad.size))

    nwave = nwave[:x_shift.size]
    x_wave = np.linspace(0, nwave.size, nwave.size)
    x_wave = x_wave[:x_shift.size]
    vol = vol[:x_shift.size]
    nwave = np.interp(x_shift, x_wave, nwave)

    DP.leave()
    return nwave, vol



def create_arpeggio_cycles(first_sample, n_samples, n_cycles, freqs):
    spaces = (1, freqs[1] / freqs[0], freqs[2] / freqs[0])

    # x_samples = []
    # samples_per_note = int(n_samples / 3 / n_cycles)
    # x_samples = [
    #     [np.linspace(first_sample     ] for idx in range(n_cycles)]
    # ]


def handle_volume_slide(vol, speed,
                        first_sample, n_samples,
                        cmd_x, cmd_y):

    n_trailing = vol.size - first_sample - n_samples

    dur_s = n_samples / SAMPLE_RATE

    ref_tempo = 0x7d
    # Tick speeds and fade durations?
    ref_speed = np.array(
        [3, 4, 5, 6, 8, 10, 12, 16, 24, 30])
    ref_fade_dur = np.array(
        [1.9, 1.7, 1.58, 1.52, 1.46, 1.42, 1.38, 1.36, 1.32, 1.32])
    fade_out = np.interp(speed, ref_speed, ref_fade_dur)
    fade_out *= ref_tempo / tempo
    fade_out_mul = 1.52 / fade_out

    vol_p = vol[:first_sample]
    vol_wt = vol[first_sample:]

    if cmd_x != 0:
        vol_i = (10 * dur_s) * cmd_x / 15 * fade_out_mul
        vol_w_delta = np.linspace(0, vol_i, n_samples)
        vol_t_delta = np.full(n_trailing, vol_i)
        vol_wt_delta = np.append(vol_w_delta, vol_t_delta)
        vol_wt += vol_wt_delta
    else:
        vol_i = (1.23 * dur_s) * cmd_y / 15 * fade_out_mul
        vol_w_delta = np.linspace(0, vol_i, n_samples)
        vol_t_delta = np.full(n_trailing, vol_i)
        vol_wt_delta = np.append(vol_w_delta, vol_t_delta)
        vol_wt -= vol_wt_delta
    return np.append(vol_p, vol_wt)

N_WAVES = 0

def modify_wave(sample, notes,
                effect_speedtempos, effect_cmds, effect_notes):
    """Takes a sample with audio data converts this to a wave with the
    correct frequences (notes), and applies all the effects
    (effect_cmds) and effect parameters (effect_notes) taking into
    account the tempo/prees per effect (effect_speedtempos)

    :param sample: sample which is a dictionary containing metainfo
    (naam, nrsamples, repeat) and audio data itself (int16 array)

    :param notes: notes (and thus freq) at which the audio data should
    be played back

    effect_speedtempos: speed/tempo of each effect in msecs

    :param effect_cmds: effect command in hex and format Cxy, where C
    is command ID and xy the value parameters

    :param effect_notes: additional target notes to which effects
    should achieve (e.g. bend to note) :return: array with audio data
    (int16)
    """
    # TODO: Some effects with same cmd but 00 for value will continu
    # cmd, 0x will change onlt par in cmd see
    # https://www.youtube.com/watch?v=ErrDlZf5ASM problably easiest to
    # rewrite effect_cmds so a 0 in second or later cmd is replaced by
    # last given value on that position but not all commands do this!
    # e.g. not 1 and 2
    global N_WAVES


    db = False

    # Create row durations

    effect_durs = [speed_and_tempo_to_msec(st[0], st[1])
                   for st in effect_speedtempos]

    # First determine how long note should last
    tot_dur = sum(effect_durs)



    # Some effect shorten wave (freq shifting), so we build some slack
    # in.
    freq_shifting=False
    for cmds in effect_cmds:
        if any(cmd[0] in '0123456' for cmd in cmds):
            freq_shifting = True
            break
    if freq_shifting:
        tot_dur *= 2
    freq = FREQS.get(notes[0])

    args = notes[0], tot_dur, freq_shifting, sample['len'], len(effect_durs)
    DP.header('MODIFY WAVE', '%s, %d ms (%s), %d, %d effects', args)

    wavenote = notes[0]
    if freq != 0:
        data = sample['data']
        wave = sample_to_wave(sample, wavenote, tot_dur)

        if notes[1] != "---":
            print('has second note...')
            wave1 = wave * 0.5
            wave2 = sample_to_wave(sample, notes[1], tot_dur) * 0.5
            min_samples = min(wave1.size, wave2.size)
            wave = wave1[:min_samples] + wave2[:min_samples]
    else:
        wave = np.array([], np.int16)

    # Without extra padding some effects which shorten the wave won't
    # work correctly like effect 0 if last effect for note which tries
    # to add a trail of negative size and will fail at end of
    # modify_wave we shorten wave to real size.
    nwave = pad_wave_to_duration(wave, tot_dur)
    if freq_shifting:
        tot_samples_real = int(nwave.size / 2)
    else:
        tot_samples_real = nwave.size

    tot_samples = nwave.size

    # Sometimes a mod contains a bug where an effect is applied
    # without a note
    if wavenote == "---":
        print('bug?')
        return nwave[:tot_samples_real]

    # Then we set wave volume to general volume different from C
    # command which only sends volume of 1 sample range of volume in
    # track is 0-64 (0x00-0x40); master_volume is 0-255 (0x00-0xFF).
    vol_i = TRACK_VOLUME / 64
    vol_i = vol_i * MASTER_VOLUME / 0xFF
    volf = np.full(nwave.size, vol_i)

    # Then we handle first-order effects, after which second-order
    # effects.
    for effect_nr in range(2):
        # Next see, if we have same effects to combine.
        neffect_speedtempos = []
        neffect_durs = []
        neffect_cmds = []
        neffect_notes = []
        irow = 0
        lastrow = len(effect_cmds) - 1
        while irow <= lastrow:
            cmd = effect_cmds[irow][effect_nr]
            speedtempo = effect_speedtempos[irow]
            dur = effect_durs[irow]
            notes = effect_notes[irow]
            jrow = irow + 1
            i_nextrow = -1
            while i_nextrow == -1:
                if jrow > lastrow:  # only 64 rows per track
                    neffect_cmds.append(cmd)
                    neffect_speedtempos.append(speedtempo)
                    neffect_durs.append(dur)
                    neffect_notes.append(notes)
                    i_nextrow = lastrow + 1
                    break
                # We combine row with same effect unless effect is 0xy
                # because this makes arpeggio logic much easier
                # (always 2 cycles per row).
                if effect_cmds[jrow][effect_nr] != cmd or (
                        cmd[0] == '0' and not cmd == '000'):
                    neffect_cmds.append(cmd)
                    neffect_speedtempos.append(speedtempo)
                    neffect_durs.append(dur)
                    neffect_notes.append(notes)
                    i_nextrow = jrow
                    break
                else:
                    dur = dur + effect_durs[jrow]
                jrow = jrow + 1
            irow = i_nextrow
        DP.print('Effect durations %s', neffect_durs)

        # And now we apply all the effects one after another
        first_sample = 0
        for speedtempo, dur, cmd, note in zip(neffect_speedtempos,
                                              neffect_durs,
                                              neffect_cmds,
                                              neffect_notes):
            n_samples = int(dur / 1000 * sample_rate)
            last_sample = first_sample + n_samples

            trailing_samples = tot_samples - last_sample
            speed = speedtempo[0]
            tempo = speedtempo[1]
            cmd_id = cmd[0]
            cmd_val = cmd[1:3]  # two bytes labeled x and y below
            cmd_xy = int(cmd_val, 16)
            cmd_x = int(cmd[1:2],16)   # for convenience
            cmd_y = int(cmd[2:3],16)   # for convenience

            # Normal play of Arpeggio; xy first halfnote add, second
            if cmd_id == '0' and not cmd == '000':
                assert tot_samples == nwave.size
                # Arpeggio not affected by speed unless speed <3 (than
                # no (01) or fast (02) arpeggio).
                nwave, volf = handle_arpeggio(volf, nwave,
                                              wavenote,
                                              first_sample, n_samples,
                                              cmd_x, cmd_y)
                print(nwave.size, volf.size)
                tot_samples = nwave.size
            # Slide Up/Dwn/ToNote       ; xx upspeed
            elif cmd_id in '123':
                #prevSamples not effected,
                # n_samples freq is stretched/squeezed
                # trailing_samples has ending freq
                #max bend for 1 octave (C-3 - B-3) at F06, F7D (120ms/row) is achieved at 7 rows, max bend of 2 octaves at 14 rows

                rel_first_sample = first_sample/tot_samples
                rel_last_sample = last_sample/tot_samples

                x_wave = np.linspace(0, 1, tot_samples)
                steps_per_sec = 60
                # Steps should increase if duration of effect
                # increases, otherwise the steps will be heard.
                steps = int(dur/1000*steps_per_sec)
                if db:
                    print ("steps:",steps)
                window_samples = int(n_samples/steps)
                #x_shift=np.array([],np.int16)
                x_shift = np.linspace(0, rel_first_sample, first_sample)

                min_samples = 1
                max_samples = window_samples * 100
                if cmd_id == '1':
                    note = 'B-3' #max note freq on any bend
                if cmd_id == '2':
                    note = 'C-1' #min note freq on any bend
                basenote = wavenote
                if db:
                    print ("NOTES:",note)
                targetnote = note
                basefreq = FREQS[basenote]
                targetfreq = basefreq + (FREQS[targetnote] - basefreq)
                if targetfreq > FREQS[targetnote]:
                    targetfreq = FREQS[targetnote]
                if db:
                    print ("basenote, targetnote: ", basenote, targetnote)
                if basefreq < targetfreq:
                    dsamples = -cmd_xy * 10.5 * speed/6 * tempo/0x7D /8
                    min_samples = int(window_samples * basefreq / targetfreq)
                else:
                    # not quite accurate
                    dsamples = +cmd_xy * 5 * speed/6 * tempo/0x7D
                    max_samples = int(window_samples * basefreq / targetfreq)
                ct=0
                if db:
                    print ("min-max samples: ",min_samples,max_samples)
                if db:
                    print ("rel_first_sample,rel_last_sample: ","%.3f" % rel_first_sample,"%.3f" % rel_last_sample)
                for i in range(steps):
                    if i > 0:
                        ct=1
                    fr = i / steps * (rel_last_sample - rel_first_sample) + rel_first_sample
                    to = (i + 1) / steps * (rel_last_sample - rel_first_sample) + rel_first_sample
                    x_window = np.linspace(fr, to, window_samples)
                    # We cut overlap in boundaries, this will cause
                    # x-shift to have #step samples less than it
                    # should be, but with normale usage (waves with ca
                    # 44 100 samples/sec and <50 steps/sec) this is
                    # unnoticable
                    x_shift = np.append(x_shift,x_window[ct:])
                    if db:
                        print ("x_window: ", "%.3f" % fr,"-", "%.3f" % to, " #", window_samples, x_window)
                    window_samples = int(window_samples + dsamples)
                    window_samples = clamp(window_samples, min_samples, max_samples)

                if db:
                    print ("x_shift pre+win: ", x_shift.size,x_shift)

                # x_shift already has head/pre and window, now only add trail
                d = x_shift[x_shift.size - 1] - x_shift[x_shift.size - 2]
                new_trailing_samples = (1 - rel_last_sample) / d + 1
                if db:
                    print ("d :","%.4f" % d,"->",new_trailing_samples,"new_trailing_samples")
                x_trail = np.linspace(rel_last_sample + d, 1, int(new_trailing_samples))
                x_shift = np.append(x_shift, x_trail)
                if db:
                    print ("x_shift trl: ", x_trail.size,x_trail)
                    print ("-------")
                    print ("x_shift: ",x_shift.size,x_shift)
                    print ("x_wave: ", x_wave.size, x_wave)
                    print ("nwave: ", nwave.size,nwave)
                nwave = np.interp(x_shift, x_wave, nwave).astype(np.int16)  # return ndarray
                # This changes length of wave, so we have to repad
                # wave to length of note first however we need to
                # deplop to prevent plop with new padded zeros
                nwave = deplop_wave(nwave,5,5)
                nwave = pad_wave_to_duration(nwave, tot_dur)
            elif cmd_id == '4':
                # Vibrato (alternate freq)      ; xy speed,depth
                # Freq is independent of speed but depends on tempo
                # (half tempo is half freq.
                if db:
                    print("nwave:", nwave.size, first_sample, n_samples, trailing_samples)
                    print("tot_samples:", tot_samples)
                rel_first_sample = first_sample / tot_samples
                rel_last_sample = last_sample / tot_samples

                x_window = np.linspace(rel_first_sample, rel_last_sample, n_samples)
                x_trail = np.linspace(rel_last_sample, 1, trailing_samples)
                x_wave = np.linspace(0,1,tot_samples)

                # increase sample spaceing to 2x dsamples reduces notefreq/2
                # increase sample spaceing to 2x dsamples reduces notefreq/2
                dsamples = x_wave[first_sample + 1] - x_wave[first_sample]
                #freq=cmd_x * x_window.size/sample_rate
                #oscillation frequency / speed of changes of note frequency
                freq = 0.6 * cmd_x * n_samples / sample_rate * (tempo / 0x7D) # freq increased if more samples
                if freq==0:
                    break
                #how much up and down we want the note frequency oscillate
                strength=cmd_y # max strength=16 (F)
                rel_strength=strength/15
                basenote = wavenote
                upnote   = next_fullnote(basenote)
                downnote = prev_fullnote(basenote)
                if db:
                    print ("wavenote,downnote,upnote:",wavenote,downnote,upnote)
                basefreq = FREQS[basenote]
                upfreq   = FREQS[upnote]
                downfreq = FREQS[downnote]
                #to make it more stand out, we overshoot the one note up/down oscillation
                upfreq   = upfreq   + (upfreq-basefreq)  *0.2
                downfreq = downfreq + (downfreq-basefreq)*0.2
                if db:
                    print("wavenote, downnote, upnote:",
                          basefreq, downfreq, upfreq)
                #quit()
                #goal: to every x-coord in sample_x we change the sample-distance by
                #       adding a sinus wave to x-coords
                #       addition has max of -100% of note down distance to 100% of note up distance
                #1) make sin wave of needed freq
                x_sinsamples = np.linspace(0,np.pi*2,x_window.size)
                if db:
                    print ("x_samples: ",x_sinsamples )
                x_sin=np.sin(x_sinsamples*freq)                    # x_windows from 0 to 1 we need 0 to 2*pi*freq
                #2) make amplitude array with length of 1 wave
                #   and amplitudes max and min determined by note distance up and down
                x_max = upfreq / basefreq - 1
                x_min = 1 - downfreq / basefreq
                a_halfwavesize=int(x_window.size/2/freq)
                a_max = np.full(a_halfwavesize,x_max)
                a_min = np.full(a_halfwavesize,x_min) #shortening distance moves freq to up note, so a_min is determined by x_max (upfreq/basefreq-1)
                x_amp=np.append(a_max,a_min)
                if db:
                    print("x_window      : ", x_window.size)
                    print("dsamples      : ", "%.9f" % dsamples)
                    print("cmd,freq      : ", cmd_x, freq)
                    print("cmd,rel_streng: ", cmd_y,rel_strength)
                    print("x_sin (amp=1) : ", x_sin.size,x_sin)
                    print("note, up, down: ", "{} ({:.1f}Hz), {} ({:.1f}Hz), {} ({:.1f}Hz)".format(basenote,basefreq,upnote,upfreq,downnote,downfreq))
                    print("x_min, a_min  : ", "%.3f" % x_min,a_min.size,a_min)
                    print("x_max, a_max  : ", "%.3f" % x_max,a_max.size,a_max)
                    print("x_amp         : ", x_amp.size, x_amp)
                #3) tile amplitude array to length of sinus wave / window
                nr_amp = math.ceil(x_sin.size/x_amp.size)
                halfpisamplenr = x_amp.size/4
                x_amp = np.tile(x_amp,nr_amp)
                if db:
                    print("nr_amp      : ", nr_amp)
                    print("x_amp   : ", x_amp.size,x_amp)
                x_amp = x_amp[:x_window.size]

                #4) scale sinus to amplitude and account for rel_strength
                x_d = x_sin*x_amp
                if db:
                    print ("min,max of x_d: ", np.amin(x_d),np.amax(x_d))
                x_d = x_d*rel_strength
                if db:
                    print("min,max of x_d: ", np.amin(x_d), np.amax(x_d))
                #   and make max shift (if x_amp=1) equal to dsamples
                x_d = x_d*dsamples
                if db:
                    print("min,max of x_d: ", np.amin(x_d), np.amax(x_d))
                #   max_shift is relative to previous sample, so x_s[i]=x_s[i-1]+x_d[i]
                x_s = np.cumsum(x_d)
                if db:
                    print("min,max of x_s: ", np.amin(x_d), np.amax(x_d))

                #5) now we scale amplitude to exact note oscillation:
                #   a) the max slope of the sinus determines freq shift, max-slope of 1.0 is normal and is 1 octave
                #   b) we need slope equal to
                #       - difference in frequency to one note up (upfreq/basefreq-1)
                #       - and accounting for strength of effect )rel)strength)
                maxshift=x_s[int(halfpisamplenr)]-x_s[int(halfpisamplenr)-1]
                max_x_d01=(upfreq/basefreq - 1)*rel_strength*dsamples
                compensate = 1

                x_s=x_s*compensate

                #6) add shifts in x-coordinates (x_d) to x coordinates (x_windows)
                x_shift_window = x_window + x_s
                if db:
                    print("x_d     : ", x_d.size, x_d)
                    print("x_window: ", x_window.size, x_window)
                    print("x_shift_window: ", x_shift_window.size, x_shift_window)

                # 7) add head/pre and trail to x_windows.
                x_shift_trail=x_trail+x_d[x_d.size-1]
                # Add last sample-displacement to prevent sudden amp
                # change between window and trail and thus a pop
                # however if x_d.size>0 then last x-coord in shift
                # array will be > 1...
                x_shift_pre = np.linspace(0, rel_first_sample, first_sample)
                x_shift = x_shift_pre
                x_shift = np.append(x_shift_pre,x_shift_window)
                x_shift = np.append(x_shift,x_shift_trail)
                if db:
                    print("Size pre,window, trail, total: ",
                          x_shift_pre.size, x_shift_window.size,
                          x_shift_trail.size, x_shift.size)

                #8) interpolate new
                nwave = np.interp(x_shift, x_wave, nwave).astype(np.int16)  # return ndarray
            elif cmd_id == '5':
                # Tone Portamento + Volume Slide; xy upspeed, downspeed
                #
                # This is rewritten in maketrack() to 3 as first
                # effect and A as second effect
                pass
            elif cmd_id == '6':
                # Vibrato + Volume Slide ; xy
                #
                # upspeed, downspeed this is rewritten in maketrack()
                # to 4 as first effect and A as second effect
                pass
            elif cmd_id == '7':
                print('TREMOLO')
                # Tremolo		          ; xy speed,depth - volume isn't reset when the command is discontinued.
                #
                # Freq is independent of speed but depends on tempo
                # (half tempo is half freq).
                voli = cmd_y / 15
                if db:
                    print("cmd_y,voli:", cmd_y, voli)
                #LFO 3-10Hz
                freq = 10 * cmd_x / 15 #10Hz max, don't now what LFO's freq in FastTracker is
                freq=freq * tempo/0x7D
                if freq==0:freq=1
                window = sample_rate // freq
                #print ("freq,window:",freq,window)
                if window==0: window=1
                nr_windows = math.ceil(n_samples / window + 1)
                volrad = np.linspace(0, 2*np.pi, window)
                volsin= np.sin(volrad)    # amplitud is from -1 to 1
                volsin= (volsin +1)/2 * voli # amplitude is from 0 to voli
                volsin= volsin+(1-voli)       # amplitude is from (1-voli) to 1
                #print("volsin:", volsin.size, volsin, volsin[volsin.size/4],volsin[3*volsin.size/4])
                volw = np.tile(volsin, nr_windows)
                #print("nr_windows,volw: ", nr_windows, volw, voln.size, "/",n_samples)
                volw=volw[:n_samples]
                voli=volw[n_samples-1] #see comment that volume is not reset after command
                if voli==0: voli=1/255 # prevents total information loss so A cmd still works
                volp = volw[:first_sample]
                volt = np.full(trailing_samples, voli)
                volf = np.concatenate((volp, volw, volt))
            elif cmd_id == '9':  # Set SampleOffset       ; xx offeset (23 -> 2300)
                twave = wave[256*cmd_xy:]
                if db:
                    print(cmd_val,cmd_xy)
                if len(twave) == 0:
                    nwave = np.zeros(tot_samples, np.int16)
                elif twave.size < tot_samples:
                    padLen = tot_samples - twave.size
                    nwave = np.pad(twave, (0, padLen), 'constant', constant_values=(0, 0))
                else:
                    nwave = twave[:tot_samples]
            # VolumeSlide
            elif cmd_id == 'A':
                volf = handle_volume_slide(volf, speed,
                                           first_sample,
                                           n_samples,
                                           cmd_x, cmd_y)
            elif cmd_id == 'B':  # Position Jump       ; xx songposition
                pass
            elif cmd_id == 'C':  # Set Volume	       ; xx volume 00-40
                volf = handle_set_volume(volf, first_sample, cmd_x, cmd_y)
            elif cmd_id == 'D':  # Pattern Break       ; xx break position in next pattern
                #handled in load_amigamodule()
                pass
            # Set speed 	       ; xx speed (00-1F) / tempo (20-FF)
            elif cmd_id == 'F':
                pass

            #calc offset of next effect
            first_sample = first_sample + n_samples

            # check if tot_samples equals length of nwave
            if not nwave.size == tot_samples:
                errmsg = "Total size of nwave changed (" + str(tot_samples) + "->" + str(nwave.size) + ")! Bug present in effect " + cmd_id+" ("+cmd+")."
                raise ValueError(errmsg)



    # we apply volume
    volf = np.clip(volf, 0, 1)
    nwave = nwave * volf

    #finally we fade volume at last 5msec of signal to avoid pop at end
    nwave = nwave[:tot_samples_real]
    nwave = deplop_wave(nwave,5,5)

    # play_wave(nwave)
    # N_WAVES += 1
    # if N_WAVES == 2:
    #     exit(1)


    DP.leave()



    return nwave.astype(np.int16)

def split_sequence(seq_text):
    """Splits a text sequence to notes, instrument number and effect
    commands

    :param seq_text: string e.g. 'C-3 E-3 01 343 000' in which the
    component order is not important
    :return: lists of notes, instr, effects
    """
    # C-3 E-3 G-3 01 343 000
    #notes are 3 characters with middle char '-' or '#'
    #second note is optional
    #instrument is 2 characters
    notes = []
    effects = []
    instr = ''
    parts =seq_text.split()
    err = ""
    hexchars = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                'A', 'B', 'C', 'D', 'E', 'F')
    for part in parts:
        if len(part) == 3 and part[1] in '-#' and \
             part[0] in 'ABCDEFG' and \
             part[2].isdigit():
                if part[0:2]=="E#" or part[0:2]=="B#":
                    err= part + " is not a valid note..."
                notes.append(part)
        elif part=="---":
            notes.append(part)
            pass #valid, just empty note
        elif len(part)==2 and \
                part[0] in hexchars and \
                part[1] in hexchars:
            if instr!='':
                err = "Only one instrument assignement allowed..."
            instr=part
        elif len(part)==3 and \
             part[0] in hexchars and \
             part[1] in hexchars and \
             part[2] in hexchars:
             effects.append(part)
        else:
            err=err+ "Syntax error in '"+part+"' of track sequence '"+seq_text+"'"

    if err:
        raise ValueError(err)
    while len(notes)<2:
        notes.append("---")
    while len(effects) < 2:
        effects.append("000")
    return notes, instr, effects

def splitseq2str(splitseq):
    """
    Converts a tuple consisting of notes, instrument and effect lists to a string.
     (Inverse of split_sequence() )
    """
    out=""
    notes, instr, effects=splitseq
    for nr in range (0,2):
        out=out+notes[nr]+" "
    out=out+instr+" "
    for nr in range(0, 2):
        out = out + effects[nr] + " "
    return out[:-1]

def rewrite_pattern(pattern_text):
    """
    Rewrite_pattern is only meant for rewriting legacy mod-format text
    like form Ultimate SoundTracker, ProTracker and the likes.
    """
    #Therefore it operates on raw text strings and not on a split pattern.

    lastrow = len(pattern_text) - 1
    for ichannel in range(0,4):
        irow = 0
        prevnote="---"
        preveffectadd="---"
        prev3xy = ''
        prev4xy = ''
        while irow <= lastrow:
            seq = pattern_text[irow][ichannel]
            # 1) check for portamento to note (cmd 3; freq shift from PREV/LAST note to given note)
            # so the real note is the prev note with a portamento up/down (cmd 1/2) with
            # the given note as max shift -> we will rewrite the sequence so the note = '---'
            # and therefore continued from the prev note to the target note in the second effect field!
            note, instr, effect = seq[0:3], seq[4:6], seq[7:10]
            cmd_id=effect[0:1]

            # for 3-command
            # C-3 01 000
            # E-3 01 343
            #  -> C-3  01 000
            #     ---  01 343 E-3
            if cmd_id == "3":
                newseq=""
                if note != "---":#first row
                    freq_target = FREQS[note]
                    newseq="---"+" "+instr+" "+effect+" "+note #note is removed for continues bend
                    preveffectadd=note
                else: #consequetive rows
                    newseq = "---" + " "+instr+" " + effect + " " + preveffectadd
                pattern_text[irow][ichannel] = newseq
            if note != "---":#next note group
                    prevnote=note

            #for 5-command
            # NOT IN EFF
            # C-3 01 000
            # G-3 01 343
            # --- 01 502
            #
            #     C-3 --- --- 01 000 000
            #     --- G-3 --- 01 343 000
            #     --- --- --- 01 343 A02
            if cmd_id == '3':
                prev3xy = effect[1:3]
            if cmd_id == '5':  # Tone Portamento + Volume Slide; xy upspeed, downspeed
                # this needs to be rewritten to 3 as first effect and A as second effect
                # xy for 3 is taken from last 3-effect played in same note group
                # xy for A is taken from sequence itself
                newseq = seq[0:7]+ "3{0:0<2}".format(prev3xy)+" A"+seq[8:10]
                pattern_text[irow][ichannel] = newseq

            # for 6-command
            if cmd_id == '4':
                prev4xy = effect[1:3]
            if cmd_id == '6':  # Vibrato + Volume Slide ; xy upspeed, downspeed
                # this needs to be rewritten to 4 as first effect and A as second effect
                # xy for 4 is taken from last 4-effect played in same note group
                # xy for A is taken from sequence itself
                newseq = seq[0:7]+ "4{0:0<2}".format(prev4xy)+" A"+seq[8:10]
                pattern_text[irow][ichannel] = newseq

            irow = irow + 1

    # Dxx command:
    # we receive all patterns stitched together (so n x 64 rows long)
    # D command jumps to line xx in next 64-row black of pattern
    # The value of D is (strangely enough) in decimals!
    rownrs = []
    for i in range(0, len(pattern_text)):
        rownrs.append(i)
    irow = lastrow
    while irow >=0:
        fnd=-1
        for ichannel in range(0, 4):
            seq = pattern_text[irow][ichannel]
            note, instr, effect = seq[0:3], seq[4:6], seq[7:10]
            cmd_id = effect[0:1]
            if cmd_id == 'D':
                fnd = int(effect[1:3],10)
        if fnd>=0:
            blocknr=irow//64
            rownr=fnd
            print ("-----------------")
            print("len(pattern_text):", len(pattern_text))
            print("seq              :", pattern_text[irow])
            print("irow             :", irow)
            #we jump to next block
            blocknr=blocknr+1
            print("blocknr          :",blocknr)
            print("rownr            :",rownr)
            #and to rownr equal to value of D
            newrownr=blocknr*64+rownr
            print("newrownr         :",newrownr)
            del pattern_text[irow:newrownr]
            del rownrs[irow:newrownr]
            print ("rownrs:",len(rownrs),rownrs)

        irow = irow - 1
    return

def transpose(notes,nr_octaves):
    newnotes=[]
    for note in notes:
        t=note[2:3]
        if t in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            curoct=int(t)
            newoct=curoct+nr_octaves
            newnotes.append(note[0:2]+str(newoct))
        else:
            newnotes.append(note)
    return newnotes

sound_lib = {}
def make_pattern_inner(pattern_text,
                       ichan,
                       pattern_rowspeedtempos,
                       pattern_refs):
    global samples, sound_lib, octave_transpose

    DP.header('MAKE PATTERN INNER', '', ())

    # We return var so address should remain same and =[] will assign
    # new mem slot.
    pattern_refs.clear()

    lastrow = len(pattern_text)-1
    lastinstr = "00"
    irow = 0
    c = False
    while irow <= lastrow:
        seq = pattern_text[irow][ichan]
        DP.print('#%03d %s', (irow, seq))

        notes, instr, effects = split_sequence(seq)
        notes = transpose(notes, octave_transpose)

        # notes[1] not supported for chords, only for special effects
        # like #3 and #5
        # note = notes[0] if notes else "---"

        # Sometimes the instrument number is omitted, this means the
        # last instrument is still active.

        if instr == "00":
            instr = lastinstr
        lastinstr = instr

        # Instruments start at #1 in a trackers, sample array in
        # python at 0.
        sample_idx = int(instr, 16) - 1

        sample = samples[sample_idx]
        samplename = sample["name"]

        # Gather positions(rows) and effects associated with note
        # (first col contains '---') and also construct id for row
        # collection
        effect_cmds = [effects]
        effect_speedtempos = [pattern_rowspeedtempos[irow]]

        seq_id = "({:02x},{:02x})".format(effect_speedtempos[0][0],
                                          effect_speedtempos[0][1]) \
                                          + ' ' + samplename +' '+seq
        jrow = irow + 1
        irow_nextNote = -1
        # notes on first row are primary note for wave
        effect_notes = ["---"]
        while irow_nextNote == -1:
            if jrow > lastrow:
                irow_nextNote = lastrow + 1
                break
            next_seq = pattern_text[jrow][ichan]
            next_notes, _, next_effects = split_sequence(next_seq)
            # notes[1] not supported for chords, only for special
            # effects like #3 and #5
            next_note = next_notes[0] if next_notes else "---"

            if next_note != '---':
                irow_nextNote = jrow
                break
            else:
                effect_cmds.append(next_effects)
                effect_speedtempos.append(pattern_rowspeedtempos[jrow])
                # Only append second note, first_note should start new
                # group.
                effect_notes.append(next_notes[1])
                seq_id = seq_id + " ; " + "({:02x},{:02x})".format(
                    effect_speedtempos[0][0],
                    effect_speedtempos[0][1]) + ' ' + next_seq
            jrow = jrow + 1
        DP.print('Effects %s', effect_cmds)

        if not seq_id in sound_lib:
            # We rely on modify_wave to padd/trunc wave to match full
            # duration of all rowtimings together.
            nwave = modify_wave(sample, notes,
                                effect_speedtempos,
                                effect_cmds,
                                effect_notes)
            snd = make_sound(nwave)
            sound_lib[seq_id] = snd
        pattern_refs.append(sound_lib[seq_id])
        # restart loop for next Note
        irow = irow_nextNote
    DP.leave()


# legacy = True, pattern_text = None
def make_pattern(legacy, pattern_text):
    """Takes a list (pattern_text) of strings in which each string consists of seperate parts per channel with a total of 4 parts/sequences.
    Per channel/track the sequences are grouped in initial note sequence and successive effect sequences and each group is converted to a sound object.
    These sound objects are added to a list for each track/channel.

    :param legacy: if legacy=True the pattern is considered external
    Amiga mod-file data which needs to be rewritten (rewrite_pattern)

    :param pattern_text: Array of strings in which each string
    consists of 4 parts/sequences like 'C-3 01 D20' (for each
    track/channel 1 sequence)

    :return: a list of 4 sublists, where each sublists contains the
    sound objects of a channel/track the rewritten pattern_text (only
    rewritten of legacy=True) the start timings of each row
    """
    global pattern, samples
    if pattern_text == None:
        pattern_text = pattern
    d = False
    # per sequence we check if it is still not synthesized and present
    # in waves_lib dictionary if not do so

    #if legacy (external mod file like ProTracker, Ultimate
    #Soundtracker) we need to do some rewriting, because the tracker
    #format is sometimes inconsequent
    d = False
    if legacy:
        rewrite_pattern(pattern_text)
    pattern_text = [[splitseq2str(split_sequence(seq)) for seq in row]
                    for row in pattern_text]

    #first build row timings
    pattern_rowspeedtempos = []
    ispeed  = 0x06
    itempo  = 0x7D
    d = False
    for row_text in pattern_text:
        # Check if any row contains (multichan) speed commands (should
        # all be handled before playing samples.
        speedCmds = []
        speedtempo = [ispeed, itempo]
        for sequence_text in row_text:
            notes, instr, effects = split_sequence(sequence_text)
            for effect in effects:
                effectCmd = effect[:1]
                effectVal = int(effect[1:3], 16)
                if effectCmd == 'F':
                    if effectVal <= 0x1F:
                        ispeed = effectVal
                    if effectVal >  0x1F:
                        itempo = effectVal
                    speedtempo = [ispeed,itempo]
        pattern_rowspeedtempos.append(speedtempo)

    lastrow = len(pattern_text) - 1
    lastinstr="00"

    t0 = time()
    background_threads=[]
    pattern_refs = [[], [], [], []]

    for i in ENABLED_CHANNELS:
        try:
            make_pattern_inner(pattern_text, i,
                               pattern_rowspeedtempos,
                               pattern_refs[i])
        except ZeroDivisionError:
            print('Divide by Zero!')
        except ValueError:
            print('Nwave error')
    print ("Time:", time()-t0)

    # convert rowtimings to cumulative timing
    lasttime = 0
    pattern_rowstarttimings = [0]
    for irow, speedtempo in enumerate(pattern_rowspeedtempos):
        rowtiming = speed_and_tempo_to_msec(speedtempo[0], speedtempo[1])
        lasttime = lasttime + rowtiming
        pattern_rowstarttimings.append(lasttime)

    global soundrefs,rowstarttimings
    pattern = pattern_text
    soundrefs = pattern_refs
    rowstarttimings = pattern_rowstarttimings
    return soundrefs, pattern_text, rowstarttimings

def play_pattern(pattern_soundrefs = None, from_time = 0):
    global req_abortplay, req_pause, req_resume

    assert not playing
    if playing:
        req_abortplay = True
        while req_abortplay:
            pass
    req_pause=False
    background_thread = Thread(target = play_pattern_inner,
                               args = (pattern_soundrefs, from_time))
    background_thread.start()
    return background_thread

req_abortplay=False
def abort_play():
    global req_abortplay
    req_abortplay=True
    return

req_pause = False
req_resume = False

play_pos=0

def get_play_pos():
    return play_pos

_lastplayrow_=0
_rowstarttimings_=[]

def get_play_row(row_starttimings = None):
    global play_pos, _lastplayrow_, rowstarttimings
    if row_starttimings==None:
        row_starttimings=rowstarttimings
    if len(row_starttimings)==0:
        return -1

    pos_msec=play_pos*1000
    if row_starttimings[_lastplayrow_] > pos_msec:
        _lastplayrow_=0
    for irow in range (_lastplayrow_,len(row_starttimings)):
        if row_starttimings[irow]>pos_msec:
          _lastplayrow_=irow-1
          return _lastplayrow_

    #if we haven't found anything, the track has ended
    return len(row_starttimings)

playing=False

def play_pattern_inner(pattern_soundrefs = None, from_time = 0):
    """Takes the list with the sound object generated in make_pattern()
    and plays the soundobjects from the time (from_time) specified.
    """
    global soundrefs, req_abortplay, req_pause, req_resume, play_pos, playing

    if pattern_soundrefs == None:
        pattern_soundrefs = soundrefs

    print("----play_pattern-----------------------------------------")
    print(f"max audiochannels: {get_num_channels()}")
    print(f"start play at    : {from_time} sec")
    set_num_channels(4)
    #stop any playing sound om the channels
    for ichan in range(4):
        Channel(ichan).stop()  # clears queue

    #determine where to start playing and slice first samples
    idx = [1, 1, 1, 1]
    first_snds = [None, None, None, None]
    if from_time > 0:
        for ichan in range(4):
            channel = pattern_soundrefs[ichan]
            dur=0
            for snd_idx, snd in enumerate(channel):
                sndlen = snd.get_length()
                if dur <= from_time and dur+sndlen>=from_time:
                    idx[ichan] = snd_idx+1
                    sec_offset=from_time-dur
                    samples=snd.get_raw()
                    samples=pygame.sndarray.array(snd)
                    nr_samples=len(samples)
                    nr_samples_offset=int(sec_offset/sndlen * nr_samples)
                    samples_slice=samples[nr_samples_offset:]
                    snd_slice = make_sound(samples_slice)
                    first_snds[ichan]=snd_slice
                    break
                dur = dur + sndlen
    else:
        for ichan in ENABLED_CHANNELS:
            first_snds[ichan] = pattern_soundrefs[ichan][0]

    soundref_lengths = [len(sr) for sr in pattern_soundrefs]
    print(f'Starting channels {soundref_lengths}.')

    playing = True
    nr_ended = 0
    queue_update_interval_sec = 0.05
    play_starttime = time()
    NOT_PAUSED = 0
    pause_time = NOT_PAUSED #needed to restart on resume
    endtime = [0,0,0,0]

    # Play first samples.
    for ichan in ENABLED_CHANNELS:
        Channel(ichan).queue(first_snds[ichan])

    # Continue filling queues with new sounds and detect user events.
    while True:
        if pause_time == NOT_PAUSED:
            new_play_pos = time() - play_starttime + from_time

            # Check if outside action (window-move/resize) has paused
            # play routine
            if new_play_pos - play_pos < queue_update_interval_sec * 4:
                play_pos = new_play_pos
            else:
                print('recursing')
                play_pattern_inner(pattern_soundrefs, play_pos)
                return

        if req_abortplay:
            return

        # Keep queue filled, if empty add next sound to channel queue.
        if pause_time == NOT_PAUSED:
            for i in ENABLED_CHANNELS:
                if not Channel(i).get_queue():
                    if idx[i] < len(pattern_soundrefs[i]):
                        channel = pattern_soundrefs[i]
                        snd = channel[idx[i]]
                        Channel(i).queue(snd)
                        idx[i] += 1
                        print(f'Refilling #{i} w/idx {idx[i]}')
                if not Channel(i).get_busy():
                    if idx[i] >= len(pattern_soundrefs[i]):
                        if endtime[i]==0:
                            dt = time()-play_starttime
                            endtime[i] = dt
                            nr_ended = nr_ended + 1
                if nr_ended == 4:
                    playing = False
            # Next channel
            # We don't want to hog CPU, so wait a bit.
            sleep(queue_update_interval_sec)
