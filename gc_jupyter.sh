wget http://repo.continuum.io/archive/Anaconda3-4.0.0-Linux-x86_64.sh
bash Anaconda3-4.0.0-Linux-x86_64.sh
source ~/.bashrc
pip install tensorflow
pip install keras
ls ~/.jupyter/jupyter_notebook_config.py
jupyter notebook --generate-config
sudo vim /home/rachel/.jupyter/jupyter_notebook_config.py


c = get_config()
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8888

from IPython.lib import passwd
password = passwd("8888")
c.NotebookApp.password = password


jupyter-notebook --no-browser --port=8888

conda create -n tf2  python=3.6

source activate tf2
pip install tensorflow

sudo apt-get install python3-dev
sudo apt-get install libevent-dev
sudo apt-get install python-dev 
sudo apt-get install libblas-dev libatlas-base-dev
sudo apt-get install libblas-dev 
sudo apt-get update
sudo apt-get --yes install libatlas-base-dev

pip install --upgrade pip
pip install -U setuptools pip
pip install deeppavlov
# ModuleNotFoundError: No module named 'bert_dp'
python -m deeppavlov install squad_bert

conda install -c anaconda jupyter
pip install decorator
pip install jinja2
pip install prometheus-client
pip install ipywidgets
pip install qtconsole
pip install prometheus-client
pip install defusedxml
pip install pygments
pip install testpath
pip install ipywidgets
pip install qtconsole
pip install jinja2
jupyter-notebook --no-browser --port=8888

pip install google.cloud.storage

touch downloadFile.py
sudo vim downloadFile.py 

from  __future__ import absolute_import, division, print_function, unicode_literals

from google.cloud import storage

downloadFiles_list = ['test.csv','train.csv','valid.csv']
for name in downloadFiles_list:
    storage_client = storage.Client()
    bucket_name ='temp-rachel'
    bucket = storage_client.bucket(bucket_name)
    stats = storage.Blob(bucket=bucket, name=name).exists(storage_client)
    if stats:
        print("the preprocessed file exist! Download the preprocessed file.")
        output_file_name = name
        storage.Blob(bucket=bucket, name=name).download_to_filename(output_file_name)