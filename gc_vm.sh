wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
chmod +x /home/rachel/Anaconda3-2019.10-Linux-x86_64.sh
./Anaconda3-2019.10-Linux-x86_64.sh

conda create -n tf2  python=3.5
conda activate tf2

pip install --upgrade pip
pip install pandas
pip install --upgrade pandas
pip install tensorflow
pip install scikit-learn
conda install -c conda-forge matplotlib

sudo apt-get install libblas-dev libatlas-base-dev
pip install datalab
pip install --upgrade google-cloud-storage

cd /dev/shm
sudo chmod -R 777 /dev/shm

# delete all connent in vim
:1,$d   

# startup.sh, run python code automatically 
#! /bin/bash
source activate tf2cpu
cd /home/rachel/bert_class/bert_classify/bert_code/berts/
python run_classifier.py
