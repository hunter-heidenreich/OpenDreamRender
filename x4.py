from argparse import ArgumentParser
from upsampling import PipelineRealESRGAN


upsampling_pipeline = PipelineRealESRGAN.from_pretrained('nateraw/real-esrgan')
upsampling_pipeline.upsampler.model = upsampling_pipeline.upsampler.model.to("cuda")
print(upsampling_pipeline.upsampler)

parser = ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True)
args = parser.parse_args()

upsampling_pipeline(args.input).save('./thumb.jpg')
