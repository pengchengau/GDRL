# Graphic Deep Reinforcement Learning for Dynamic Resource Allocation in Space-Air-Ground Integrated Networks - GDRL Simulation Code Package

The simulation code package is related to the following publication. 

**Y. Cai, P. Cheng, Z. Chen, W. Xiang, B. Vucetic, and Y. Li, "Graphic Deep Reinforcement Learning for Dynamic Resource Allocation in Space-Air-Ground Integrated Networks," IEEE Journal on Selected Areas in Communications, vol. 43, no. 1, pp. 334-349, January 2025.**

Kindly cite this paper if you use this simulation code package in your research.

The author for this code package is Dr. Yue Cai.

Please visit https://pengchengau.github.io/ for other related publications and code packages.

----------------------------------------------------------------------------------------------------

**Abstract**: Space-Air-Ground integrated network (SAGIN) is a crucial component of the 6G, enabling global and seamless communication coverage. This multi-layered communication system integrates space, air, and terrestrial segments, each with computational capability, and also serves as a ubiquitous computing platform. An efficient task offloading and resource allocation scheme is key in SAGIN to maximize resource utilization efficiency, meeting the stringent quality of service (QoS) requirements for different service types. In this paper, we introduce a dynamic SAGIN model featuring diverse antenna configurations, two timescale types, different channel models for each segment, and dual service types. We formulate a problem of sequential decision-making task offloading and resource allocation. Our proposed solution is an innovative online approach referred to as graphic deep reinforcement learning (GDRL). This approach utilizes a graph neural network (GNN)-based feature extraction network to identify the inherent dependencies within the graphical structure of the states. We design an action mapping network with an encoding scheme for end-to-end generation of task offloading and resource allocation decisions. Additionally, we incorporate meta-learning into GDRL to swiftly adapt to rapid changes in key parameters of the SAGIN environment, significantly reducing online deployment complexity. Simulation results validate that our proposed GDRL significantly outperforms state-of-the-art DRL approaches by achieving the highest reward and lowest overall latency.

----------------------------------------------------------------------------------------------------

**Code Explaination:** This model employs a graph convolutional  neural network (GCN) to extract features from topologicially structured dataset. It encompasses three  parts: 1. LAGN: It is built upon TRPO to generate latent actions. 2. AMN: It is an autoencoder structured network that generate both discrete offloading and continous allocation decisions. 3. GFEN: It is construcuted by GCN to perform feature extraction. 

### Requirements
Code is written in python and requires the installation of `torch`, `argparse`, `stable_baselines3`,  `numpy`, `scipy`, `gymnasium`, `sb3_contrib`, and `itertools` packages. Additionally, Anaconda is required and the version of Python utilized is 3.7. Anaconda can be installed via [Anaconda Webpage](https://anaconda.org/anaconda) by clicking Download Anaconda and other packages can be installed via:
```
pip install numpy
pip install torch
pip install argparse
pip install stable_baselines3
pip install scipy
pip install gymnasium
pip install itertools
pip install sb3_contrib
```
The model is trained using GPU by setting device="cuda", please make sure that CUDA is correctly installed and the computer used to train this model has a CUDA-capable system. If not, please delete the device="cuda" command in the main.py function and train this model using CPU. It is not receommanded to train the model using CPU as it is time consuming.

Note that this model requires the collaboration of Python and Matlab. Please make sure that Matlab is correctly installed and the package `matlab.engine` exists, which can be installed via:
```
pip install matlab.engine
```

### Code Overview
The main function of each file is described breifly in the following.
* amp.py : The code for Action Mapping Network (AMN) of GDRL.
* arg_parser.py : Hyperparameter settings like the number of LEO/HAPS/UE and some training related hyperparameters.
* CallBack.py : Customized callback function to store offloading/allocation actions, reward and achieved latency.
* Channel_Model.py : The channel model as detailed in our paper.
* Environment_baseline.py : The environment used to train GDRL. It is built on gym environment.
* Feature.py : The GFEN of GDRL. Specifically, this is the customized feature extraction network that implements GCN.
* Generate_adj_matrix.py : This function generates the connection status of the network.
* Generate_inital_environment.py : This function generates the initial environment. It fulfiles the reset function required by gym.
* groundtospace.m : This is the matlab function that complements the ground to space channel model used in this work. For detail, please refer to our work.
* HAPSLEO_Status.py : This function generates the initial status like avalible number of VMs for each LEO/HAPS.
* main.py : This is the main function.
* Rate_Calculation.py : This function calculates the transmission rate for each user based on the channel model.
* UpdateVariable.py : This function updates the status of LEO/HAPS during training based on the offloading/allocation decisions.
* UserRequest.py : This function generates the user requests used to training the model.
* UserStatus.py : This function generates a collection of user requests based on UserRequest.py.

To customize this code, you need to change Environment_baseline.py to your own environment and Feature.py to your own feature extraction network. Data need to be generated as the same shape as that generated by UserStatus.py and HAPSLEO_Status.py and the batch size need to be customized in arg_parser.py. 

### Model
![GDRL](https://github.com/user-attachments/assets/a03b1de4-4ea9-4e8e-9ae8-547de1b7f3dd)

This figure shows the general achitecture of GDRL. Please refer to our work for further details. The code for each part is lised in the previous code overview section.

### Train and Results
The model can be trained by running the following code
```
python main.py
```
It is recommanded to run this code using pycharm. Navigate to the main.py function and run this script will start the training. Training results will be saved in a newly created folder named trpo_tensorboard and the result name can be changed by modifying tb_log_name="first_run". Here, the default name is set to first_run. Note that only the final reward is valid as the training of GDRL is different from conventional TRPO. Other results are not customized and cannot reflect the true perfromance of GDRL.

The trained model will be saved in a new folder with the name model_save. It will be automatically created once the training is finished. You can customize the name by modifying model.save("./model_save/trpo_trained"). By default, the model saved is named as trpo_trained. Addtionally, discrete and continous AMN are saved seperatly under the same folder. One can customize the name by modifying the name torch.save(autoencoder_dis.state_dict(), './model_save/autoencoder_dis.pth'), torch.save(autoencoder_con.state_dict(), './model_save/autoencoder_con.pth'). Specifically, the discrete AMN is saved as autoencoder_dis and continous one as autoencoder_con. 

### Note
* The batch size is set to 100 in this code and can be customized by modifying arg_parser.py.
* Please check your CUDA version before installing torch.
* New pacekge may need to be installed for enabling the collaboartion between Matlab and Python. 
