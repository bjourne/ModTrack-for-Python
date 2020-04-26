from sys import exit
from modtrack import DP

def nibbles(bt):
    h = bt >> 4
    l = bt & 0x0F
    return h,l

def nibbles2byte(low,high):
    return high*16+low

def nibbles2(bt_array):
    nibble_array=[]
    for bt in bt_array:
        h,l=nibbles(bt)
        nibble_array.append(h)
        nibble_array.append(l)
    return nibble_array

def hexs(bt_array):
    hexout=""
    for bt in bt_array:
        hexout=hexout+(hex(bt)[2:].zfill(2))
    return hexout.upper()

format = ""
samples=[]
patterns=[]
pattern_table=None
nr_playedpatterns=0

def amigaword_toint(b1,b2):
    return b2*256+b1

MOD_FORMATS = {
    'STK.' : (
        "Ultimate Soundtracker (Original) 4 channel / 15 instruments",
        15, True
    ),
    'M.K.' : (
        "Protracker 4 channel / 31 instruments",
        31, True
    ),
    'M!K!' : (
        "Protracker 4 channel / 31 instruments / >64 patterns",
        31, False
    ),
    'FLT4' : (
        "Startracker 4 channel / 31 instruments",
        31, True
    ),
    'FLT8' : (
        "Startracker 8 channel / 31 instruments",
        31, False
    ),
    '2CHN' : (
        "Fasttracker 2 channel / 31 instruments",
        31, False
    ),
    '4CHN' : (
        "Fasttracker 4 channel / 31 instruments",
        31, True
    ),
    '6CHN' : (
        "Fasttracker 6 channel / 31 instruments",
        31, False
    ),
    '8CHN' : (
        "Fasttracker 8 channel / 31 instruments",
        31, False
    ),
    'CD81' : (
        "Atari oktalyzer 8 channel / 31 instruments",
        31, False
    ),
    'OKTA' : (
        "Atari oktalyzer 8 channel / 31 instruments",
        31, False
    ),
    'OCTA' : (
        "Atari oktalyzer 8 channel / 31 instruments",
        31, False
    ),
    '16CN' : (
        "Taketracker 16 channel / 31 instruments",
        31, False
    ),
    '32CN' : (
        "Taketracker 32 channel / 31 instruments",
        31, False
    )
}

def formdesc_from_bytes(bytes4):
    try:
        format = bytes4.decode("utf-8")
    except UnicodeDecodeError:
        format = "STK."
    if bytes4 == b'\x00\x00\x00\x00':
        format = "STK."
    if not format.isprintable():
        format = "STK."
    return format, MOD_FORMATS[format]

# https://wiki.multimedia.cx/index.php/Protracker_Module
# http://www.fileformat.info/format/mod/corion.htm
# http://elektronika.kvalitne.cz/ATMEL/MODplayer3/doc/MOD-FORM.TXT
# http://www.eblong.com/zarf/blorb/mod-spec.txt
# http://web.archive.org/web/20120806024858/http://16-bits.org/mod/
# ftp://ftp.modland.com/pub/documents/format_documentation
#     /FireLight%20MOD%20Player%20Tutorial.txt

def read_module(filename):
    global samples,patterns,pattern_table,nr_playedpatterns

    with open(filename, 'rb') as fh:
        barr = bytearray(fh.read())

    # Compressed with PowerPacker, we can't decode this
    if barr[0:4] == "PP20":
        return None

    nr_channels = 4
    id_bytes = barr[1080:1084]
    id, (desc, nr_samples, compatible) = formdesc_from_bytes(id_bytes)
    if not compatible:
        errmsg = f'Format {id} ({desc}) is not supported!'
        raise ValueError(errmsg)

    songtitle = barr[0:20].decode("utf-8")

    fmt = 'song "%s" type %s'
    DP.header('LOADING', fmt, (songtitle, desc))

    offset = 20
    for sample in range (0, nr_samples):
        sample = {}
        sample["name"] = barr[offset:offset + 22]
        sample['name'] = sample['name'].decode("utf-8").replace('\x00', '')

        # sample len in words (1word=2bytes). 1st word overwritten by tracker
        sample['len'] = 2*int.from_bytes(
            barr[offset+22:offset+24],
            byteorder="big",signed=False)
        sample["finetune"] = barr[offset + 24]#.decode("utf-8")
        sample["volume"] = barr[offset + 25]#.decode("utf-8")
        sample["repeat_from"] = 2 * int.from_bytes(barr[offset + 26:offset + 28],byteorder="big",signed=False)
        sample["repeat_len"] = 2 * int.from_bytes(barr[offset + 28:offset + 30],byteorder="big",signed=False)

        fmt = 'sample "%-20s" %5d bytes repeat %2d:%2d vol %2d'
        args = (sample['name'], sample['len'],
                sample['repeat_from'], sample['repeat_len'],
                sample['volume'])
        DP.print(fmt, args)
        samples.append(sample)
        offset=offset+30

    DP.print('offset %d', offset)

    #offset=470 15 samples Ultimate Soundtracker, id at 600
    #offset=950 31 samples Protracker and similar, id at 1080

    nr_playedpatterns=barr[offset] # hex value was loaded as byte and is automatically converted to int
    offset=offset+1
    dummy127=barr[offset]
    offset=offset+1
    pattern_table=barr[offset:offset+128]
    offset=offset+128

    DP.print('offset %d', offset)
    # Only other format then Ultimate Soundtracker have bytes to
    # specify format
    if not format == "STK.":
      dummyformat = barr[offset:offset+4].decode("utf-8")
      offset = offset+4

    DP.print('patterns %d, format %s' , (nr_playedpatterns, format))

    #read nr patterns stored
    #equal to the highest patternnumber in the song position table(at offset 952 - 1079).
    nr_patterns_stored=0
    for chnr in range(128):
        DP.print('pattern_table[%3d] = %d', (chnr, pattern_table[chnr]))
        # Check for first not possible because 0 is also a valid
        # pattern number
        if pattern_table[chnr]!=0:
            nr_patternsplayed=chnr+1
        if (pattern_table[chnr]+1)>nr_patterns_stored:
            nr_patterns_stored=(pattern_table[chnr]+1)

    pattern_table=pattern_table[:nr_playedpatterns]
    DP.print("nr patterns stored: %d", nr_patterns_stored)

    notelist = ["C-", "C#", "D-", "D#", "E-", "F-", "F#", "G-", "G#", "A-", "A#", "B-"]
    periods = [
        1712,1616,1525,1440,1357,1281,1209,1141,1077,1017, 961, 907,
        856, 808, 762, 720, 678, 640, 604, 570, 538, 508, 480, 453,
        428, 404, 381, 360, 339, 320, 302, 285, 269, 254, 240, 226,
        214, 202, 190, 180, 170, 160, 151, 143, 135, 127, 120, 113,
        107, 101,  95,  90,  85,  80,  76,  71,  67,  64,  60,  57,
    ]
    def period2note(period):
        notenr=-1
        for nr,val in enumerate(periods):
            if val==period:
                notenr=nr % 12
                oct=nr//12
        if notenr>=0:
            note=notelist[notenr]+str(oct)
        else:
            note="---"
        return note

    patterns = []
    for pattern in range (0,nr_patterns_stored):
        pattern=[]
        for row in range(64):
            row=[]
            txt=""
            txt2= ""
            for channel in range(nr_channels):
              bytes = barr[offset:offset+4]

              nibbles=hexs(bytes)
              samplenr=int(nibbles[0]+nibbles[4],16)
              samplehex=nibbles[0]+nibbles[4]



              noteperiod=int(nibbles[1:4],16)

              effect = nibbles[5:8]
              txt2=txt2+"{} -> {} {} {} | ".format(nibbles,noteperiod,samplehex,effect)
              note=period2note(noteperiod)
              seq_text="{:<3} {} {:<3}".format(note,samplehex,effect)
              row.append(seq_text)
              #txt = txt + seq_text+ " "
              txt=txt+"|"+nibbles+"|"+str(noteperiod)
              offset=offset+4
            pattern.append(row)
        patterns.append(pattern)

    fmt = '%2d, offset %6d, len %5d'
    for i, sample in enumerate(samples):
        sample_len=sample["len"]
        # first two bytes always two zeros, and used for repeating is
        # the sample is to be terminated.
        sample_data=barr[offset+2:offset+sample_len]
        sample['data'] = sample_data
        DP.print(fmt, (i, offset, sample_len))
        offset = offset+sample_len
    if offset != len(barr):
        print("ERROR....NOT ALL BYTES PROCESSED! ",
              len(barr)-offset, "remain.")
    DP.leave()
    return True
