# Molecular Dynamics with Neural Networks

This project was part of a six-month, part-time research assistant position under the supervision of Professor Enrico Tapavicza. The main goal is to speed up the prediction of forces and the energy of a second-generation Feringa-type molecular nanomotor. In our case this is 9-(2’-methyl-2’,3’-dihydro-1’H-cyclopenta[a]naphthalen-1’-ylidene)-9H-xanthene (CPNX). 

## Outline

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Workflow](#workflow)
4. [Contributing](#contributing)
5. [License](#license)

## Project Structure

The "md_with_schnet" directory contains everything of the project, as we have only worked with [SchNetPack](https://github.com/atomistic-machine-learning/schnetpack) so far. The "deprecated" directory contains every outdated piece of code, that at some point will be removed from this project.


## Installation

Once you have cloned this project, you can use the environment.yaml file to build the conda environment needed to execute the project code. The needed commands are as follows:


```bash
git clone git@github.com:FerdinandToelkes/whk.git
conda env create -f path/to/environment.yml
conda activate schnet
```

## Workflow

Each script should include an example of how to execute it at the top.

### preprocessing

- Obtain gradients, positions and velocities from mdlog.i files with the extract.py script
- Transform the extracted properties into a .db file (which is the format used within SchNetPack) by employing the prepare_xtb.py script
- Define how the data later should be splitted into training, validation and test data via the create_splits.py script

Note that paths need to be updated depending on the local setup especially of the data. 

### neural_net

- Use train.py to train a neural network via SchNetPack (adjust parameters via the command line or the train_config.yml if necessary)
- Run inference_with_ase.py to generate a MD trajectory starting from a configuration within the test dataset
- Execute ~~order 66~~ the plot_interactive_md_ase_sim.py script in order to gain an overview of the various energies from the two trajectories as well as their correlation 

## Interactive Plots

Here is a quick overview of results for training a neural network on the MOTOR_MD_XTB/T300_1 dataset. We used the trained model to run a MD and the plots show a comparison between the model's prediction for the energies with predictions made by xTB that can be viewed [here](https://FerdinandToelkes.github.io/whk/MOTOR_MD_XTB_T300_1_epochs_1000_bs_100_lr_0.0001_seed_42/md_sim_steps_5000_time_step_1.0_seed_42/interactive_properties_plot.html) and the corresponding rolling correlation between the energies, that is displayed in [this plot](https://FerdinandToelkes.github.io/whk/MOTOR_MD_XTB_T300_1_epochs_1000_bs_100_lr_0.0001_seed_42/md_sim_steps_5000_time_step_1.0_seed_42/interactive_rolling_corr_plot.html)
 

## Contributing

If you would like to contribute to the project, please open an issue or a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

