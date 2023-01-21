pip install scipy
git clone https://github.com/jnordberg/tortoise-tts.git
cd tortoise-tts
pip install transformers==4.19.0
pip install -r requirements.txt
python setup.py install
pip install .
cd ../
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 
pip install omegaconf==2.2.3 einops==0.4.1 pytorch-lightning==1.7.4 torchmetrics==0.9.3 torchtext==0.13.1 transformers==4.21.2 kornia==0.6.7
git clone https://github.com/deforum/stable-diffusion
pip install -e "git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers"
pip install -e "git+https://github.com/openai/CLIP.git@main#egg=clip"
pip install accelerate ftfy jsonmerge matplotlib resize-right timm torchdiffeq
git clone https://github.com/shariqfarooq123/AdaBins.git
git clone https://github.com/isl-org/MiDaS.git
git clone https://github.com/MSFTserver/pytorch3d-lite.git
git clone https://github.com/deforum/k-diffusion/
echo '' > k-diffusion/k_diffusion/__init__.py
pip install --upgrade numexpr scipy accelerate realesrgan tqdm nltk ffmpeg-python
