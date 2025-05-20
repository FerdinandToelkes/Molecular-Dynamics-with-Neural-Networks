import torch
from schnetpack.md import System
from schnetpack.md.calculators import MDCalculator

from ase.build import molecule
from xtb_ase import XTB

class AseMDCalculator(MDCalculator):
    def __init__(self, ase_calc: XTB, atoms: molecule, force_key: str = "forces", energy_key: str | None = None, required_properties: list[str] = [], **kwargs):
        super().__init__(
            required_properties=required_properties,
            energy_unit="eV",
            position_unit="Angstrom", # TODO Think about units
            force_key=force_key,
            energy_key=energy_key,
            **kwargs
        )
        self.ase_calc = ase_calc  # your xtb ASE calculator
        atoms.calc = ase_calc
        self.atoms = atoms

    def calculate(self, system: System):
        # Convert SchNetPack System → ASE Atoms
        self.atoms.set_positions(system.positions[0].cpu().numpy())

        # Run ASE calculation
        e = self.atoms.get_potential_energy()
        f = self.atoms.get_forces()

        # # Update SchNetPack system
        # if self.energy_key:
        #     system.set_property(self.energy_key, torch.tensor(e, dtype=torch.float64))
        # forces: (n_atoms, 3)
        self.results = {
            self.force_key: torch.tensor(f, dtype=torch.float64).unsqueeze(0),
            self.energy_key: torch.tensor(e, dtype=torch.float64).unsqueeze(0),
        }

        for p in self.required_properties:
            if p not in self.results:
                raise RuntimeError( # otherwise copy error class from base_calculator
                    "Requested property {:s} not in " "results".format(p)
                )
            else:
                dim = self.results[p].shape
                # Bring to general structure of MD code. Second dimension can be n_mol or n_mol x n_atoms.
                system.properties[p] = (
                    self.results[p].view(system.n_replicas, -1, *dim[1:])
                    * self.property_conversion[p]
                )

        # Set the forces for the system (at this point, already detached)
        self._set_system_forces(system)

        # Store potential energy to system if requested:
        if self.energy_key is not None:
            self._set_system_energy(system)

        # Set stress of the system if requested:
        if self.stress_key is not None:
            self._set_system_stress(system) 



def main():
    atoms = molecule("H2O")
    atoms.calc = XTB()

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    print(f"Energy: {energy:.6f} eV")
    print("Forces:", forces)

    
if __name__ == "__main__":
    main()
