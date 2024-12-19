# README #

This software accompanies the 2017 NIPS [paper](https://papers.nips.cc/paper/7231-protein-interface-prediction-using-graph-convolutional-networks) and [poster](https://zenodo.org/record/1134154), Protein Interface Prediction using Graph Convolutional Networks.
We implemented multiple versions of graph convolution and applied them to the problem of protein interface prediction.
This work was supported by the National Science Foundation under grant no DBI-1564840.

## Setup ##
### Requirements ###

- python 2.7
- PyYAML 3.12
- numpy 1.13.3
- scikit-learn 0.19.1
- tensorflow 1.0.1

### Detail ###
Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main  
_openmp_mutex             5.1                       1_gnu  
blas                      1.0                         mkl  
ca-certificates           2024.9.24            h06a4308_0  
certifi                   2020.6.20          pyhd3eb1b0_3  
funcsigs                  1.0.2                    pypi_0    pypi
intel-openmp              2023.1.0         hdb19cb5_46306  
libffi                    3.4.4                h6a678d5_1  
libgcc-ng                 11.2.0               h1234567_1  
libgfortran-ng            7.5.0               ha8ba4b0_17  
libgfortran4              7.5.0               ha8ba4b0_17  
libgomp                   11.2.0               h1234567_1  
libstdcxx-ng              11.2.0               h1234567_1  
mkl                       2018.0.3                      1  
mkl_fft                   1.0.6            py27h7dd41cf_0  
mkl_random                1.0.1            py27h4414c95_1  
mock                      3.0.5                    pypi_0    pypi
ncurses                   6.4                  h6a678d5_0  
numpy                     1.13.3           py27hdbf6ddf_4  
pip                       19.3.1                   py27_0  
protobuf                  3.17.3                   pypi_0    pypi
python                    2.7.18               h42bf7aa_3  
pyyaml                    3.12             py27h2d70dd7_1  
readline                  8.2                  h5eee18b_0  
scikit-learn              0.19.1           py27hedc7406_0  
scipy                     1.1.0            py27hd20e5f9_0  
setuptools                44.0.0                   py27_0  
six                       1.16.0                   pypi_0    pypi
sqlite                    3.45.3               h5eee18b_0  
tensorflow                1.0.1                    pypi_0    pypi
tk                        8.6.14               h39e8969_0  
wheel                     0.37.1             pyhd3eb1b0_0  
yaml                      0.2.5                h7b6447c_0  
zlib                      1.2.13               h5eee18b_1  


### Environment Variables ###
The software assumes the following environment variables are set:

- PL_DATA: full path of data directory (where data files are kept)
- PL_OUT: full path of output directory (where experiment results are placed)
- PL_EXPERIMENTS: full path of experiment library (YAML files)

An alternative to setting these variables is to edit the portions of configuration.py which reference these environment variables.
#### ####
To add environment variables in Linux, you can follow these steps. This guide will help you set the required environment variables (`PL_DATA`, `PL_OUT`, and `PL_EXPERIMENTS`) either temporarily for the current session or permanently for all future sessions.

### Setting Environment Variables in Linux

#### 1. Temporary Environment Variables

If you want to set the environment variables for the current terminal session only, you can use the `export` command. This will last until you close the terminal.

Open your terminal and run the following commands:

export PL_DATA=/path/to/your/data/directory
export PL_OUT=/path/to/your/output/directory
export PL_EXPERIMENTS=/path/to/your/experiment/library

Replace `/path/to/your/data/directory`, `/path/to/your/output/directory`, and `/path/to/your/experiment/library` with the actual paths you want to use.

#### 2. Permanent Environment Variables

To set environment variables permanently, you can add them to your shell's configuration file. The file you need to edit depends on the shell you are using. Common shells include bash

##### For Bash Users

1. Open the .bashrc file in your home directory with a text editor (e.g., nano, vim, or gedit):
   nano ~/.bashrc

2. Add the following lines at the end of the file:
   export PL_DATA=/path/to/your/data/directory
   export PL_OUT=/path/to/your/output/directory
   export PL_EXPERIMENTS=/path/to/your/experiment/library

3. Save the file and exit the editor (in `nano`, you can do this by pressing CTRL + X, then Y, and `Enter`).

4. To apply the changes, run:
   source ~/.bashrc
   
### Alternative: Editing `configuration.py`

If you prefer not to set environment variables, you can directly edit the configuration.py file in your software's directory. Look for the sections that reference PL_DATA, PL_OUT, and PL_EXPERIMENTS, and replace them with the full paths you want to use.


### CUDA Setup ###
Consider setting the following environment variables for CUDA use:

- LD_LIBRARY_PATH: path to cuda libraries
- CUDA_VISIBLE_DEVICES: Specify (0, 1, etc.) which GPU to use or set to "" to force CPU

### Data ###

To run the provided experiments, you need the pickle files found [here](https://zenodo.org/record/1127774#.WkLewGGnGcY).


## Running Experiments ##

Simply run:
```python experiment_runner.py <experiment>```.
Where ```<experiment>``` is the name of the experiment file (including .yml extension) in the experiments directory.
Alternatively you may run ```run_experiments.sh```, which contains expressions for all provided experiments.
