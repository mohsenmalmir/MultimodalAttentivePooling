sudo apt update
sudo apt install python3 python3-dev python3-venv
sudo apt-get install wget
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
pip --version


sudo apt install build-essential
sudo apt-get install libxml2
wget https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda_11.2.0_460.27.04_linux.run
sudo sh cuda_11.2.0_460.27.04_linux.run


pip install scikit-image
pip install tensorboard
sudo apt-get install git
pip install  torch torchvision
pip install  gensim
cd ~/
git clone https://github.com/mohsenmalmir/MultimodalAttentivePooling.git
cd MultimodalAttentivePooling && git checkout mohsen/tvqa
cd ../ && git clone https://github.com/jayleicn/TVRetrieval.git
cd ~/MultimodalAttentivePooling
PYTHONPATH=./ python3 ./scripts/data/prep_tvr.py \
    -i ~/TVRetrieval/data/tvr_train_release.jsonl -d ~/frames_hq/ \
    -s ~/TVRetrieval/data/tvqa_preprocessed_subtitles.jsonl -o ~/tvr_train.json
PYTHONPATH=./ python3 ./scripts/data/prep_tvr.py \
    -i ~/TVRetrieval/data/tvr_val_release.jsonl -d ~/frames_hq/ \
    -s ~/TVRetrieval/data/tvqa_preprocessed_subtitles.jsonl -o ~/tvr_val.json