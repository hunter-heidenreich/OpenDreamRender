import json
import os 
import subprocess

from glob import glob
from subprocess import Popen, PIPE

from argparse import ArgumentParser
from tqdm.auto import tqdm

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


def num2str(ix, total):
    if total <= 10:
        return f'{ix:01d}'
    elif total <= 100:
        return f'{ix:02d}'
    elif total <= 1_000:
        return f'{ix:03d}'
    else:
        return ''
    

parser = ArgumentParser('Upsample')
parser.add_argument('-i', '--input', type=str, required=True)
pargs = parser.parse_args()

settings = json.load(open(pargs.input + 'input/settings.json'))
with open(pargs.input + 'text/script.txt') as fp:
    lines = list(fp.readlines())
    total = len(lines)

for nx in tqdm(range(total), desc='Rendering Videos.'):
    tgt_dir = pargs.input + f'frames/4k_{num2str(nx, total)}/'
    out_file = pargs.input + f'frames/{num2str(nx, total)}.mp4'
        
    if os.path.isfile(out_file):
        print(out_file, 'exists... Skipping.')
        continue
    
    subprocess.run([
        'ffmpeg', 
        '-f', 'image2',
        '-r', str(int(settings['video']['fps'] / settings['video']['savings'])), 
        '-i', tgt_dir + 'out_%05d.jpg', 
        '-vcodec', 'libx264', 
        '-crf', '18',  
        '-pix_fmt', 'yuv420p',
        out_file
    ])
    
