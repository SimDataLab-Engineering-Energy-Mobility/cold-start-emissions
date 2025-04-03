# Table of Contents
  - [Description](#description)
  - [Getting Started](#getting-started)
  - [Installation](#installation)
    - [Requirements](#requirements)
    - [Example Data](#example-data)
    - [Resources](#resources)
  - [Example Data](#Example Data)
  - [License](#license)
  - [Citing](#citing)
  - [Acknowledgements](#acknowledgements)
  - [Authors](#authors)

# Description 

This repository contains two independent scripts: 
- `ENC_DEC_LSTM.ipynb`: Implements an Encoder-Decoder LSTM architecture.
- `MLP.ipynb`: Implements a Multi-Layer Perceptron (MLP) for supervised learning.

Each script can be run independently and requires user input to select the epoch with the smallest validation loss, which is conveniently displayed on the screen for reference. 

This abovementioned software was used to predict hazardous gaseous emissions such as CO, NOx and unburned hydrocarbons (UHC) from passenger cars. The predictions are based on seven measured features (2,500 time steps at a 5 Hz rate) as input. The three emissions (outputs) are predicted simultaneously within the same model. The measurements were conducted after the cars had been stationary for an entire "night," meaning the engine was in a cold state at the start of the measurements (hence, a cold-start). 

The cold-start-emissions software consists of two independent machine learning models. The first is a multilayer perceptron (MLP) model that processes all 2,500 time-steps simultaneously and predicts emissions for those time-steps. The second model is an Encoder-Decoder architecture based on long short-term memory (LSTM) networks — one LSTM for the encoder and one for the decoder. This model performs real-time predictions by taking the last 16 seconds (80 time-steps) as input and predicting emissions for 1 second (5 time-steps). Therefore, if integrated into the car's software, this model can be used for real-time predictions. 

A detailed description of the models can be found in the corresponding scientific publication; please refer to the file "CITATION.md" for further information. 


# Getting Started
For the development and use of the software, we utilized one of the two supercomputers: 
https://uc2-jupyter.scc.kit.edu/ and 
https://hk-jupyter.scc.kit.edu/
with "jupyter/ai" as JupyterLab-Basemodule. 

# Requirements

## Basics
- python >= ???3.9
- pytorch >= ????2.0.0
- scipy???
- mathplotlib ??? 


## GPU support
In order to do computations on your GPU(s):
- your CUDA or ROCm installation must match your hardware and its drivers;
- your [PyTorch installation](https://pytorch.org/get-started/locally/) must be compiled with CUDA/ROCm support.


## Example Data
Due to an agreement regarding data distribution, we are unable to present the actual data here. Instead, we have prepared artificial data with the same shape and structure as the real data. These synthetic datasets can be used for testing or modifying the software.  


# License
Licensed under the GNU General Public License v3.0, see our LICENSE file.


## Authors
- Manoj Mangipudi  
- Jordan A. Denev  
Karlsruhe Institute of Technology (KIT)  
Scientific Computing Center (SCC)
Department of Scientific Computing and Mathematics 


# Citing 
If you find our software helpful for your research, please mention it in your publications. Example how you can cite it is given in file CITATION.md 


# Acknowledgements
*This work was performed on the HoreKa supercomputer at the Scientific Computing Center (SCC) of the Karlsruhe Institute of Technology, with support from the National High-Performance Computing Center project (NHR@KIT). Both HoreKa and NHR@KIT are funded by the Ministry of Science, Research and the Arts Baden-Württemberg and the German Federal Ministry of Education and Research.* 
