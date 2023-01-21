import json
import os

import librosa
import numpy as np
import soundfile as sf 

from argparse import ArgumentParser
from tqdm.auto import tqdm


def num2str(ix, total):
    if total <= 10:
        return f'{ix:01d}'
    elif total <= 100:
        return f'{ix:02d}'
    elif total <= 1_000:
        return f'{ix:03d}'
    else:
        return ''


parser = ArgumentParser('Audio')
parser.add_argument('-i', '--input', dest='input', type=str, required=True)

args = parser.parse_args()
settings = json.load(open(args.input + 'input/settings.json'))
with open(args.input + 'audio/sel.txt') as fp:
    lines = [line.strip() for line in fp.readlines()]

os.makedirs(args.input + 'audio_mx', exist_ok=True)
full = None 
srf = None
for ix, f in enumerate(tqdm(lines)):
    
    inpath = args.input + 'audio/' + f
    outpath = args.input + f'audio_mx/{num2str(ix, len(lines))}.wav'
    wav, sr = sf.read(inpath)
    
    # pad 
    wav_p = librosa.util.pad_center(wav, size=wav.shape[0] + 2 * sr * settings['audio']['pad'])
    
    # normalize 
    ratio = settings['audio']['normalization_level']/np.max(np.abs(wav_p))
    wav_pn = ratio * wav_p
    
    if ix:
        full = np.append(full, wav_pn)
    else:
        full = np.array(wav_pn)
        srf = sr 
    
    sf.write(outpath, wav_pn, sr)

sf.write(args.input + f'audio_mx/out.wav', full, srf)
