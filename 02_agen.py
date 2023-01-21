import json
import torchaudio

from os import makedirs, path, chdir

from argparse import ArgumentParser 
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices
from tqdm.auto import tqdm


def num2str(ix, total):
    # assuming total > 0
    # assuming counting ix from 0 
    # assuming ix < total 
    # assuming ix, total both int
    if total <= 10:
        return f'{ix:01d}'
    elif total <= 100:
        return f'{ix:02d}'
    elif total <= 1_000:
        return f'{ix:03d}'
    else:
        return ''
    
parser = ArgumentParser('Audio Generation')
parser.add_argument('-i', '--input', dest='input', type=str, required=True)
args = parser.parse_args()

chdir('tortoise-tts')
tgt_dir = '../' + args.input

settings = json.load(open(tgt_dir + 'input/settings.json'))
with open(tgt_dir + 'text/script.txt') as fp:
    lines = [line.strip() for line in fp.readlines()]
    total = len(lines)

tts = TextToSpeech()
preset = settings['audio']['quality']  # 'fast'  # 'standard'
voice = settings['audio']['voice']  # 'deniro'
voice_samples, conditioning_latents = load_voice(voice)

times = settings['audio']['samples']
makedirs(tgt_dir + 'audio', exist_ok=True)
# for ix, line in enumerate(lines):
#     for iy in range(times):
#         outpath = tgt_dir + 'audio/' + num2str(ix, len(lines)) + '_' + num2str(iy, times) + '.wav'
#         exists = path.isfile(outpath)
#         print(ix, iy, exists)
    
#         if not exists:
#             gen = tts.tts_with_preset(line, voice_samples=voice_samples, conditioning_latents=conditioning_latents, preset=preset)
#             torchaudio.save(outpath, gen.squeeze(0).cpu(), 24000)

# Proposed re-write:
# 
# Use Top-K samples instead of Top-1 sample of K forward passes 
for ix, line in enumerate(tqdm(lines)):
    outpath = tgt_dir + 'audio/' + num2str(ix, total) + '_' + num2str(0, times) + '.wav'
    exists = path.isfile(outpath)
    if exists:
        print(outpath, 'exists... Skipping.')
        continue
    
    gen = tts.tts_with_preset(
        line, 
        voice_samples=voice_samples, 
        conditioning_latents=conditioning_latents, 
        preset=preset,
        k=times,
    )
    for iy in range(times):
        outpath = tgt_dir + 'audio/' + num2str(ix, len(lines)) + '_' + num2str(iy, times) + '.wav'
        torchaudio.save(outpath, gen[iy].squeeze(0).cpu(), 24000)
