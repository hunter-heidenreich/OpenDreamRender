import json
import os 

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

from upsampling import PipelineRealESRGAN
upsampling_pipeline = PipelineRealESRGAN.from_pretrained('nateraw/real-esrgan')
upsampling_pipeline.upsampler.model = upsampling_pipeline.upsampler.model.to("cuda")
print(upsampling_pipeline.upsampler)

for nx in tqdm(range(total)):
    tgt_dir = pargs.input + f'frames/{num2str(nx, total)}/' 
    out_dir = pargs.input + f'frames/4k_{num2str(nx, total)}/'
        
    if os.path.isfile(out_dir):
        print(out_dir, 'exists... Skipping.')
        continue
        
    os.makedirs(out_dir, exist_ok=True)
    for f in tqdm(sorted(glob(f'{tgt_dir}*.jpg'))):
        img = upsampling_pipeline(f)
        img.save(out_dir + f.split('/')[-1])
