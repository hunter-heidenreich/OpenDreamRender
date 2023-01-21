from os import makedirs

from argparse import ArgumentParser 
from nltk.tokenize import sent_tokenize


parser = ArgumentParser('Text Processing')
parser.add_argument('-i', '--input', dest='input', type=str, required=True)

args = parser.parse_args()

with open(args.input + 'input/raw.txt') as fp:
    lines = fp.readlines()
    
lines = [line.strip() for line in lines if line.strip()]
lines = ''.join([lx + '\n' for line in lines for lx in sent_tokenize(line)]).strip()

makedirs(args.input + 'text', exist_ok=True)
with open(args.input + 'text/script.txt', 'w+') as fp:
    fp.write(lines)
