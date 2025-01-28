# TL-PINNs
Third year individual project on Transfer Learning (TL) integrated PINNs for fluid mechanics.
I'm currently connecting this repository to MobaXterm for HPC usage.

Workflow is as follows:

Make change in VS -> Push to GitHub -> pull on Moba using 'git pull origin main' from within cd

Make change in Moba -> Push to GitHub ('make the change' -> git status -> git add . -> git commit -m "update message" -> 
git push origin main) -> Pull on local machine in GitHub Desktop

# Requirements

TensorFlow == 2.7.0  
NumPy == 1.24.4  
Cuda == 11.2  
Cudnn == 8.1  
matplotlib == 3.2.2  
Scipy == 1.10.1  

# Naming Convention
{run_ID}_nIter

n_Iter is number of iterations

run_ID ->
A_B_C_D:
A is the Re of the finished model,
B is PINN for a PINN model and TL for a transfer learned model
C is the frozen layers if applicable
D is something else


# Process

For training and verifying PINN model:
Clean and process source data for training case A -> Train model A on said data -> Clean and process test data -> Make and plot flow predictions -> Compare predictions to true values and compute errors

For TL:
Change hyper paramters of model A -> process target data -> re-train model on target data -> Make and plot flow predictions -> Compare predictions to true values and compute errors 


absl-py                   0.15.0
anyio                     4.2.0
argon2-cffi               21.3.0
argon2-cffi-bindings      21.2.0
asttokens                 2.0.5
astunparse                1.6.3
async-lru                 2.0.4
attrs                     24.2.0
Babel                     2.11.0
backcall                  0.2.0
beautifulsoup4            4.12.3
bleach                    4.1.0
Brotli                    1.0.9
cachetools                5.5.0
certifi                   2024.8.30
cffi                      1.17.1
charset-normalizer        3.3.2
colorama                  0.4.6
comm                      0.2.1
contourpy                 1.1.1
cycler                    0.12.1
debugpy                   1.6.7
decorator                 5.1.1
defusedxml                0.7.1
exceptiongroup            1.2.0
executing                 0.8.3
fastjsonschema            2.16.2
flatbuffers               1.12
fonttools                 4.55.5
gast                      0.4.0
google-auth               2.35.0
google-auth-oauthlib      0.4.6
google-pasta              0.2.0
grpcio                    1.34.1
h11                       0.14.0
h5py                      3.1.0
httpcore                  1.0.2
httpx                     0.27.0
idna                      3.10
importlib-metadata        7.0.1
importlib_resources       6.4.0
ipykernel                 6.29.5
ipython                   8.12.2
jedi                      0.19.1
Jinja2                    3.1.4
json5                     0.9.6
jsonschema                4.23.0
jsonschema-specifications 2023.7.1
jupyter_client            8.6.0
jupyter_core              5.7.2
jupyter-events            0.10.0
jupyter-lsp               2.2.0
jupyter_server            2.14.1
jupyter_server_terminals  0.4.4
jupyterlab                4.2.5
jupyterlab-pygments       0.1.2
jupyterlab_server         2.27.3
keras-nightly             2.5.0.dev2021032900
Keras-Preprocessing       1.1.2
kiwisolver                1.4.7
loging                    0.0.1
Markdown                  3.7
MarkupSafe                2.1.3
matplotlib                3.2.2
matplotlib-inline         0.1.6
mistune                   2.0.4
nbclient                  0.8.0
nbconvert                 7.16.4
nbformat                  5.10.4
nest-asyncio              1.6.0
networkx                  3.1
notebook                  7.2.2
notebook_shim             0.2.3
numpy                     1.19.2
oauthlib                  3.2.2
opt-einsum                3.3.0
overrides                 7.4.0
packaging                 24.1
pandas                    2.0.3
pandocfilters             1.5.0
parso                     0.8.3
pickleshare               0.7.5
pillow                    10.4.0
pip                       24.2
pkgutil_resolve_name      1.3.10
platformdirs              3.10.0
prometheus-client         0.14.1
prompt-toolkit            3.0.43
protobuf                  3.20.3
psutil                    5.9.0
pure-eval                 0.2.2
pyasn1                    0.6.1
pyasn1_modules            0.4.1
pycparser                 2.21
Pygments                  2.15.1
pyparsing                 3.1.4
PySocks                   1.7.1
python-dateutil           2.9.0.post0
python-json-logger        2.0.7
pytz                      2024.1
pywin32                   305.1
pywinpty                  2.0.10
PyYAML                    6.0.2
pyzmq                     25.1.2
referencing               0.30.2
requests                  2.32.3
requests-oauthlib         2.0.0
rfc3339-validator         0.1.4
rfc3986-validator         0.1.1
rpds-py                   0.10.6
rsa                       4.9
scipy                     1.4.1
Send2Trash                1.8.2
setuptools                75.1.0
six                       1.15.0
sniffio                   1.3.0
soupsieve                 2.5
stack-data                0.2.0
tensorboard               2.11.2
tensorboard-data-server   0.6.1
tensorboard-plugin-wit    1.8.1
tensorflow                2.5.0
tensorflow-estimator      2.5.0
termcolor                 1.1.0
terminado                 0.17.1
tinycss2                  1.2.1
tomli                     2.0.1
tornado                   6.4.1
tqdm                      4.67.1
traitlets                 5.14.3
typing-extensions         3.7.4.3
tzdata                    2025.1
urllib3                   2.2.3
wcwidth                   0.2.5
webencodings              0.5.1
websocket-client          1.8.0
Werkzeug                  3.0.6
wheel                     0.44.0
win-inet-pton             1.1.0
wrapt                     1.12.1
zipp                      3.20.2