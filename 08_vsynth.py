import os 

from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True)
parser.add_argument('-v', '--vol', type=float, default=0.6)

args = parser.parse_args()


vfn = args.input + 'clips/out.mp4'
afn = args.input + 'audio_mx/out.wav'
mfn = args.input + 'input/music.mp3'

cmd = f'ffmpeg -i {vfn} -i {afn} {args.input}out.mp4'
print(cmd)
os.system(cmd)

cmd = f'ffmpeg -i {args.input}out.mp4 -i {mfn} -filter_complex "[1:a]volume={args.vol},apad[A];[0:a][A]amerge[out]" -c:v copy -map 0:v -map [out] -shortest {args.input}out-fx.mp4'
print(cmd)
os.system(cmd)
