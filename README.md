# Molecular Dynamics with Neural Networks

This project was part of a six month research assistant position with Prof. Enrico Tapavicza. The main goal is to speed up the prediction of forces and the energy of a second-generation Feringa-type molecular nanomotor. In our case this is 9-(2’-methyl-2’,3’-dihydro-1’H-cyclopenta[a]naphthalen-1’-ylidene)-9H-xanthene (CPNX). 

## Outline

The general structure of the project is as follows:

1. [md_with_schnet](#md_with_schnet)
   - [preprocessing](#preprocessing)
   - [data_analysis](#data_analysis)
   - [neural_net](#neural_net)
2. [deprecated](#deprecated)

## md_with_schnet

This directory contains everything of the project so far, as we have only worked with SchNetPack so far. 


## deprecated

This directory contains every outdated piece of code, that will at some point be removed from this project.


## Installation

Once you cloned this project you can use the environment.yaml file to build the conda environment needed to execute the project code.


```bash
git clone
conda env create -f environment.yml
conda activate schnet
```

Note that paths need to be updated depending on the local setup especially of the data. 

## Contributing

If you would like to contribute to the project, please open an issue or a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

