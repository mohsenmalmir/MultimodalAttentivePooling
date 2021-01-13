pip install  torch torchvision
pip install  gensim
cd ~/
git clone https://github.com/mohsenmalmir/MultimodalAttentivePooling.git
cd MultimodalAttentivePooling && git checkout mohsen/tvqa
cd ../ && git clone https://github.com/jayleicn/TVRetrieval.git
PYTHONPATH=./MultimodalAttentivePooling python MultimodalAttentivePooling/scripts/data/prep_tvr.py \
    -i TVRetrieval/data/tvr_train_release.jsonl -d /content/tvqa/ \
    -s TVRetrieval/data/tvqa_preprocessed_subtitles.jsonl -o tvr_train.json
PYTHONPATH=./MultimodalAttentivePooling python MultimodalAttentivePooling/scripts/data/prep_tvr.py \
    -i TVRetrieval/data/tvr_val_release.jsonl -d /content/tvqa/ \
    -s TVRetrieval/data/tvqa_preprocessed_subtitles.jsonl -o tvr_val.json