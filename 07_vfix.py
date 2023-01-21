import os
import ffmpeg

from argparse import ArgumentParser
from glob import glob 
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


parser = ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True)
parser.add_argument('-f', '--fade', type=float, default=0.5)

args = parser.parse_args()

fadelen = args.fade

tgt_dir = args.input + 'frames/'
out_dir = args.input + 'clips/'
os.makedirs(out_dir, exist_ok=True)


vfns = sorted(list(glob(tgt_dir + '*.mp4')))
fouts = []
for fn in tqdm(vfns):
    file_in = fn 
    file_out = out_dir + fn.split('/')[-1]
    info = ffmpeg.probe(file_in)
    d = float(info['format']['duration'])
    cmd = f'ffmpeg -y -i {file_in} -vf "fade=in:st=0:d={fadelen},fade=out:st={d - fadelen}:d={fadelen}" {file_out}'
    print(cmd)
    os.system(cmd)
    fouts.append(f"file '{fn.split('/')[-1]}'\n")

with open(out_dir + 'files.txt', 'w+') as fp:
    fp.writelines(fouts)
    
cmd = f'cd {out_dir}; ffmpeg -f concat -i files.txt -c copy out.mp4'
print(cmd)
os.system(cmd)
