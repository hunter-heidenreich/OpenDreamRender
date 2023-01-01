pip3 install -U scipy
git clone https://github.com/jnordberg/tortoise-tts.git
cd tortoise-tts
pip3 install transformers==4.19.0
pip3 install -r requirements.txt
python3 setup.py install
cd ../