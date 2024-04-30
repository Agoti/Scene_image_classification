mkdir checkpoints
mkdir config
mkdir data

cd data
wget https://cloud.tsinghua.edu.cn/seafhttp/files/ff200020-8402-4613-a312-d3194a6fb092/scene_classification.zip
unzip scene_classification.zip
rm scene_classification.zip

cd ..
python Config.py
