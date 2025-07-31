# Molecular Dynamics with Neural Networks

This project was part of a six-month, part-time research assistant position under the supervision of Professor Enrico Tapavicza. The main goal is to speed up the simulation of molecular dynamics (MDs) of a second-generation Feringa-type molecular nanomotor. In our case this is 9-(2’-methyl-2’,3’-dihydro-1’H-cyclopenta[a]naphthalen-1’-ylidene)-9H-xanthene (CPNX) (see for example the [paper](https://pubs.rsc.org/en/content/articlepdf/2025/cp/d5cp01063b)). 

## Outline

1. [Project Structure](#project-structure)
2. [Ground State Molecular Dynamics](#ground-state-molecular-dynamics)
   - [Data Overview](#data-overview)
   - [Installation](#installation)
   - [Workflow](#workflow)
3. [Exited State Molecular Dynamics](#exited-state-molecular-dynamics)
   - [Data Overview](#data-overview)
   - [Installation](#installation)
   - [Workflow](#workflow)
4. [Resources](#resources)
5. [Contributing](#contributing)
6. [License](#license)

# Project Structure

This project is split into two main parts, namely the **ground_state_md** and **exited_state_md** directories. The training of neural networks (NNs) to predict ground state MDs is done by employing the [SchNetPack](https://github.com/atomistic-machine-learning/schnetpack) package, whereas the [SPaiNN](https://pubs.rsc.org/en/content/articlepdf/2024/sc/d4sc04164j) package was used to train NNs for the prediction of exited state trajectories. The main differences between these two tasks are summarized below

| Feature / Task                       | Ground State MD (`ground_state_md`)               | Excited State MD (`exited_state_md`)                      |
|--------------------------------------|---------------------------------------------------|-----------------------------------------------------------|
| Target Properties                    | Potential energy S0 and resulting forces          | Potential energies S0 and S1, resulting forces and non‑adiabatic couplings S0 -> S1 |
| Training Data                        | Ground‑state trajectories using xTB with REMD     | Ground MD with xTB + excited‑state MD with TDDFT and FSSH |
| ML Framework                         | SchNetPack + ASE Interface                        | SPaiNN (PaiNN + SchNetPack + SHARC interface) |
| Dynamics Setup                       | Adiabatic MD on a single PES                      | Non‑adiabatic MD using FSSH  (via SHARC)  |

Since I started with the code for ground state MD without knowing the exact direction of this project, much of the code for the exited state MD is based on or directly using code from ground_state_md.
The "deprecated" directory contains every outdated piece of code, that at some point may be removed from this project.

# Ground State Molecular Dynamics

## Data Overview

The dataset used for training the neural network consists of five replica exchange molecular dynamics (REMD) simulations performed with the extended tight binding (xTB) software. The simulations were carried out on the CPNX nanomotor which consists of 48 atoms, and the data includes information about atomic positions, energies, forces and velocities. Different dihedral angles were defined in order to cluster the sampled structures into the following four conformations: "syn-M", "anti-M", "syn-P" and "anti-P". See the [paper](https://pubs.rsc.org/en/content/articlepdf/2025/cp/d5cp01063b) by Lucia-Tamudo et al. for more details on the underlying data and this clustering. The evolution of the dihedral angles for one of the simulations (T300_1) can be viewed [here](https://FerdinandToelkes.github.io/whk/dihedral_angles_MOTOR_MD_XTB_T300_1.html). As one can see, the configurations of this simulations are mostly in anti-M conformation. Throughout the following we will only focus on the T300_1 data when training our networks.


## Installation

Once you have cloned this project, you can use the environment.yaml file within the ground_state_md folder to build the conda environment needed to execute the project code. The needed commands are as follows:


```bash
git clone git@github.com:FerdinandToelkes/whk.git
cd /path/to/cloned/directory
conda env create -f ground_state_md/environment.yml
conda activate schnet
```

## Workflow

Each script should include an example of how to execute it at the top. All python scripts are to be executed from the root directory of the project. The "target_dir" as well as the "trajectory_dir" parameters have to be set relative to the data directory (see also set_data_prefix within utils.py). In my case, we could have target_dir = MOTOR_MD_XTB/T300_1.

### Preprocessing

- Obtain gradients, positions and velocities from mdlog.i files with the extract.py script:
```bash
python -m ground_state_md.preprocessing.extract \
    --property gradients \
    --target_dir path/to/dir/with/mdlog.i/files
```
- Obtain energies from mdlog.i files with Turbomole by executing directly in the directory with the mdlog.i files (make sure that Turbomole is installed and enabled):
```bash
log2egy > energies.txt
```
- Transform the extracted properties into a .db file (which is the format used within SchNetPack) by employing the prepare_xtb_data.py script
```bash
python -m ground_state_md.preprocessing.prepare_xtb_data \
    --trajectory_dir path/to/dir/with/mdlog.i/files \
    --num_atoms 48 --position_unit angstrom \
    --energy_unit kcal/mol --time_unit fs
```
- Define how the data later should be splitted into training, validation and test data via the create_splits.py script:
```bash
python -m ground_state_md.preprocessing.create_splits \
    --trajectory_dir path/to/dir/with/mdlog.i/files \
    --units angstrom_kcal_per_mol_fs 
```
- If needed, compute the mean and standard deviation of the various properties in the training set via the compute_means_and_stds.py script:
```bash
python -m ground_state_md.preprocessing.compute_means_and_stds \
    --trajectory_dir path/to/dir/with/mdlog.i/files \
    --num_atoms=48 --units angstrom_kcal_per_mol_fs
```
Note that paths need to be updated depending on the local setup especially of the data. 

### Training and Inference

- Use train.py to train a neural network via SchNetPack (adjust parameters via the command line or the .yml config file if necessary)
```bash
screen -dmS xtb_train sh -c 'python -m ground_state_md.neural_net.train \ 
    --trajectory_dir path/to/dir/with/mdlog.i/files --epochs 1000  \ 
    --batch_size 100 --learning_rate 0.0001 --seed 42 \
    --config_name train_config_default_transforms 
    --units angstrom_kcal_per_mol_fs; exec bash'
```
- Use get_test_metrics.py to predict the energies, forces and gradients of the test set with the trained model
```bash
python -m ground_state_md.neural_net.get_test_metrics \
    --model_dir MOTOR_MD_XTB/T300_1/epochs_1000_bs_100_lr_0.0001_seed_42
```
- Run inference_with_ase.py to generate a MD trajectory starting from a configuration within the test dataset
```bash
screen -dmS inference_xtb sh -c 'python -m ground_state_md.neural_net.inference_with_ase \
    --model_dir MOTOR_MD_XTB/T300_1/epochs_1000_bs_100_lr_0.0001_seed_42 \
    --units angstrom_kcal_per_mol_fs --md_steps 100 --time_step 0.5 ; exec bash'
```
- Execute ~~order 66~~ the plot_interactive_md_ase_sim.py script in order to gain an overview of the various energies from the two trajectories as well as their correlation 
```bash
python -m ground_state_md.neural_net.plot_interactive_md_ase_sim \
    --model_dir MOTOR_MD_XTB/T300_1/epochs_1000_bs_100_lr_0.0001_seed_42 \
    --simulation_name  md_sim_steps_5000_time_step_1.0_seed_42 \
    --n_samples 5000 --units angstrom_kcal_per_mol_fs
```

## Results

Here is a quick overview of results for training a neural network on the MOTOR_MD_XTB/T300_1 dataset. We used the trained model to run a MD and the plots show a comparison between the model's prediction for the energies with predictions made by xTB that can be viewed [here](https://FerdinandToelkes.github.io/whk/angstrom_kcal_per_mol_fs/MOTOR_MD_XTB/T300_1/epochs_1000_bs_100_lr_0.0001_seed_42/md_sim_steps_5000_time_step_1.0_seed_42/interactive_properties_plot.html) and the corresponding rolling correlation between the energies, that is displayed in [this plot](https://FerdinandToelkes.github.io/whk/angstrom_kcal_per_mol_fs/MOTOR_MD_XTB/T300_1/epochs_1000_bs_100_lr_0.0001_seed_42/md_sim_steps_5000_time_step_1.0_seed_42/interactive_rolling_corr_plot.html)
 
# Exited State Molecular Dynamics

## Data Overview

The dataset used for training the neural network consists of five replica exchange molecular dynamics (REMD) simulations performed with the extended tight binding (xTB) software. The simulations were carried out on the CPNX nanomotor which consists of 48 atoms, and the data includes information about atomic positions, energies, forces and velocities. Different dihedral angles were defined in order to cluster the sampled structures into the following four conformations: "syn-M", "anti-M", "syn-P" and "anti-P". See the [paper](https://pubs.rsc.org/en/content/articlepdf/2025/cp/d5cp01063b) by Lucia-Tamudo et al. for more details on the underlying data and this clustering. The evolution of the dihedral angles for one of the simulations (T300_1) can be viewed [here](https://FerdinandToelkes.github.io/whk/dihedral_angles_MOTOR_MD_XTB_T300_1.html). As one can see, the configurations of this simulations are mostly in anti-M conformation. Throughout the following we will only focus on the T300_1 data when training our networks.


## Installation

Once you have cloned this project, you can use the environment.yaml file within the exited_state_md folder to build the conda environment needed to execute the project code. The needed commands are as follows:


```bash
git clone git@github.com:FerdinandToelkes/whk.git
cd /path/to/cloned/directory
conda env create -f exited_state_md/environment.yml
conda activate schnet
```

## Workflow

Each script should include an example of how to execute it at the top. All python scripts are to be executed from the root directory of the project. The "target_dir" as well as the "trajectory_dir" parameters have to be set relative to the data directory (see also set_data_prefix within utils.py). In my case, we could have target_dir = MOTOR_MD_XTB/T300_1.

### Preprocessing

- Obtain gradients, positions and velocities from mdlog.i files with the extract.py script:
```bash
python -m ground_state_md.preprocessing.extract \
    --property gradients \
    --target_dir path/to/dir/with/mdlog.i/files
```
- Obtain energies from mdlog.i files with Turbomole by executing directly in the directory with the mdlog.i files (make sure that Turbomole is installed and enabled):
```bash
log2egy > energies.txt
```
- Transform the extracted properties into a .db file (which is the format used within SchNetPack) by employing the prepare_xtb_data.py script
```bash
python -m ground_state_md.preprocessing.prepare_xtb_data \
    --trajectory_dir path/to/dir/with/mdlog.i/files \
    --num_atoms 48 --position_unit angstrom \
    --energy_unit kcal/mol --time_unit fs
```
- Define how the data later should be splitted into training, validation and test data via the create_splits.py script:
```bash
python -m ground_state_md.preprocessing.create_splits \
    --trajectory_dir path/to/dir/with/mdlog.i/files \
    --units angstrom_kcal_per_mol_fs 
```
- If needed, compute the mean and standard deviation of the various properties in the training set via the compute_means_and_stds.py script:
```bash
python -m ground_state_md.preprocessing.compute_means_and_stds \
    --trajectory_dir path/to/dir/with/mdlog.i/files \
    --num_atoms=48 --units angstrom_kcal_per_mol_fs
```
Note that paths need to be updated depending on the local setup especially of the data. 

### Training and Inference

- Use train.py to train a neural network via SchNetPack (adjust parameters via the command line or the .yml config file if necessary)
```bash
screen -dmS xtb_train sh -c 'python -m ground_state_md.neural_net.train \ 
    --trajectory_dir path/to/dir/with/mdlog.i/files --epochs 1000  \ 
    --batch_size 100 --learning_rate 0.0001 --seed 42 \
    --config_name train_config_default_transforms 
    --units angstrom_kcal_per_mol_fs; exec bash'
```
- Use get_test_metrics.py to predict the energies, forces and gradients of the test set with the trained model
```bash
python -m ground_state_md.neural_net.get_test_metrics \
    --model_dir MOTOR_MD_XTB/T300_1/epochs_1000_bs_100_lr_0.0001_seed_42
```
- Run inference_with_ase.py to generate a MD trajectory starting from a configuration within the test dataset
```bash
screen -dmS inference_xtb sh -c 'python -m ground_state_md.neural_net.inference_with_ase \
    --model_dir MOTOR_MD_XTB/T300_1/epochs_1000_bs_100_lr_0.0001_seed_42 \
    --units angstrom_kcal_per_mol_fs --md_steps 100 --time_step 0.5 ; exec bash'
```
- Execute ~~order 66~~ the plot_interactive_md_ase_sim.py script in order to gain an overview of the various energies from the two trajectories as well as their correlation 
```bash
python -m ground_state_md.neural_net.plot_interactive_md_ase_sim \
    --model_dir MOTOR_MD_XTB/T300_1/epochs_1000_bs_100_lr_0.0001_seed_42 \
    --simulation_name  md_sim_steps_5000_time_step_1.0_seed_42 \
    --n_samples 5000 --units angstrom_kcal_per_mol_fs
```

## Results

Here is a quick overview of results for training a neural network on the MOTOR_MD_XTB/T300_1 dataset. We used the trained model to run a MD and the plots show a comparison between the model's prediction for the energies with predictions made by xTB that can be viewed [here](https://FerdinandToelkes.github.io/whk/angstrom_kcal_per_mol_fs/MOTOR_MD_XTB/T300_1/epochs_1000_bs_100_lr_0.0001_seed_42/md_sim_steps_5000_time_step_1.0_seed_42/interactive_properties_plot.html) and the corresponding rolling correlation between the energies, that is displayed in [this plot](https://FerdinandToelkes.github.io/whk/angstrom_kcal_per_mol_fs/MOTOR_MD_XTB/T300_1/epochs_1000_bs_100_lr_0.0001_seed_42/md_sim_steps_5000_time_step_1.0_seed_42/interactive_rolling_corr_plot.html)

# Resources

## Data
- [First principles prediction of wavelength-dependent isomerization quantum yields of a second-generation molecular nanomotor](https://pubs.rsc.org/en/content/articlepdf/2025/cp/d5cp01063b)

## Ground State Dynamics
- [SchNetPack: A Deep Learning Toolbox For Atomistic Systems](https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.8b00908)
- [SchNetPack documentation](https://schnetpack.readthedocs.io/en/latest/)
- [SchNetPack GitHub Page](https://github.com/atomistic-machine-learning/schnetpack)
- [SchNet – A deep learning architecture for molecules and materials](https://pubs.aip.org/aip/jcp/article/148/24/241722/962591/SchNet-A-deep-learning-architecture-for-molecules)

## Exited State Dynamics
- [SPAINN: equivariant message passing for excited-state nonadiabatic molecular dynamics](https://pubs.rsc.org/en/content/articlepdf/2024/sc/d4sc04164j)
- [SpaiNN documentation](https://spainn.readthedocs.io/en/latest/index.html)
- [SpaiNN GitHub Page](https://github.com/CompPhotoChem/SPaiNN)
- [Nonadiabatic dynamics: The SHARC approach](https://wires.onlinelibrary.wiley.com/doi/pdfdirect/10.1002/wcms.1370)
- [Pedagogical Overview of the Fewest Switches Surface Hopping Method](https://pubs.acs.org/doi/pdf/10.1021/acsomega.2c04843?ref=article_openPDF)
- [Ab initio non-adiabatic molecular dynamics](https://pubs.rsc.org/en/content/articlepdf/2013/cp/c3cp51514a)

## General Methods
- [Dynamic Filter Networks](https://proceedings.neurips.cc/paper/2016/file/8bf1211fd4b7b94528899de0a43b9fb3-Paper.pdf)
- [Neural Message Passing for Quantum Chemistry](https://proceedings.mlr.press/v70/gilmer17a/gilmer17a.pdf)
- [Graph Neural Networks Series (Blog post)](https://medium.com/the-modern-scientist/graph-neural-networks-series-part-4-the-gnns-message-passing-over-smoothing-e77ffee523cc)
- [Lecture Notes on Data Analysis](https://indico.in2p3.fr/event/2086/contributions/22818/attachments/18562/22658/cowan_statnote.pdf)

# Unfinished Thoughts on Change of Units

TODO: try to add handwritten document.

# Contributing

If you would like to contribute to the project, please open an issue or a pull request.

# License

This project is licensed under the MIT License - see the LICENSE file for details.

