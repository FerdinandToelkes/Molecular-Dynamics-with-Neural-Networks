"""
Physical and unit conversion constants. The hard coded reference values are taken from Wikipedia
"""
import numpy as np
import torch

from ase import units

# atomic units
BOHR_TO_ANGSTROM = units.Bohr  # 1 Bohr = 0.529177210544 Å
HARTREE_TO_KCAL_PER_MOL = units.Hartree * units.mol / units.kcal  # 1 Hartree = 627.5094740631 kcal/mol
HARTREE_TO_EV = units.Hartree # 1 eV = 27.211386245988 Hartree
AUT_TO_FS = units._aut * 1e15 # 1 AUT = 2.4188843265864e-2 fs
AUT_TO_S = units._aut * 1e-15  # 1 AUT = 2.4188843265864e-17 s

KCAL_PER_MOL_TO_EV = units.kcal / units.mol  # 1 kcal/mol = 0.0433641 eV


# Tolerance for floating point comparison
FLOAT_TOL = 1e-6

assert abs(BOHR_TO_ANGSTROM - 0.529177210544) < FLOAT_TOL, "BOHR_TO_ANGSTROM is not set correctly"
assert abs(HARTREE_TO_KCAL_PER_MOL - 627.5094740631) < FLOAT_TOL, "HARTREE_TO_KCAL_PER_MOL is not set correctly"
assert abs(HARTREE_TO_EV - 27.211386245988) < FLOAT_TOL, "HARTREE_TO_EV is not set correctly"
assert abs(AUT_TO_FS - 2.4188843265864e-2) < FLOAT_TOL, "AUT_TO_FS is not set correctly"
assert abs(AUT_TO_S - 2.4188843265864e-17) < FLOAT_TOL, "AUT_TO_S is not set correctly"
assert abs(KCAL_PER_MOL_TO_EV - 0.0433641) < FLOAT_TOL, "KCAL_PER_MOL_TO_EV is not set correctly"


def convert_energies(energies: np.ndarray | torch.Tensor, from_units: str, to_units: str) -> np.ndarray | torch.Tensor:
    """ Convert energies between different units.
    Args:
        energies (np.ndarray or torch.Tensor): Energies to convert.
        from_units (str): Units of the input energies.
        to_units (str): Units to convert the energies to.
    Returns:
        np.ndarray or torch.Tensor: Converted energies.
    """
    if from_units == "hartree" and to_units == "kcal/mol":
        return energies * HARTREE_TO_KCAL_PER_MOL
    elif from_units == "kcal/mol" and to_units == "hartree":
        return energies / HARTREE_TO_KCAL_PER_MOL
    elif from_units == "hartree" and to_units == "ev":
        return energies * HARTREE_TO_EV
    elif from_units == "ev" and to_units == "hartree":
        return energies / HARTREE_TO_EV
    elif from_units == "kcal/mol" and to_units == "ev":
        return energies * KCAL_PER_MOL_TO_EV
    elif from_units == "ev" and to_units == "kcal/mol":
        return energies / KCAL_PER_MOL_TO_EV
    else:
        raise ValueError(f"Unsupported conversion from {from_units} to {to_units}")
    
def convert_forces(forces: np.ndarray | torch.Tensor, from_units: str, to_units: str) -> np.ndarray | torch.Tensor:
    """ Convert forces between different units.
    Args:
        forces (np.ndarray or torch.Tensor): Forces to convert.
        from_units (str): Units of the input forces.
        to_units (str): Units to convert the forces to.
    Returns:
        np.ndarray or torch.Tensor: Converted forces.
    """
    if from_units == "hartree/bohr" and to_units == "kcal/mol/angstrom":
        return forces * HARTREE_TO_KCAL_PER_MOL / BOHR_TO_ANGSTROM
    elif from_units == "kcal/mol/angstrom" and to_units == "hartree/bohr":
        return forces * BOHR_TO_ANGSTROM / HARTREE_TO_KCAL_PER_MOL
    elif from_units == "hartree/bohr" and to_units == "ev/angstrom":
        return forces * HARTREE_TO_EV / BOHR_TO_ANGSTROM
    elif from_units == "ev/angstrom" and to_units == "hartree/bohr":
        return forces * BOHR_TO_ANGSTROM / HARTREE_TO_EV
    elif from_units == "kcal/mol/angstrom" and to_units == "ev/angstrom":
        return forces * KCAL_PER_MOL_TO_EV
    elif from_units == "ev/angstrom" and to_units == "kcal/mol/angstrom":
        return forces / KCAL_PER_MOL_TO_EV
    else:
        raise ValueError(f"Unsupported conversion from {from_units} to {to_units}")
    
def convert_distances(distances: np.ndarray | torch.Tensor, from_units: str, to_units: str) -> np.ndarray | torch.Tensor:
    """ Convert distances between different units.
    Args:
        distances (np.ndarray or torch.Tensor): Distances to convert.
        from_units (str): Units of the input distances.
        to_units (str): Units to convert the distances to.
    Returns:
        np.ndarray or torch.Tensor: Converted distances.
    """
    if from_units == "bohr" and to_units == "angstrom":
        return distances * BOHR_TO_ANGSTROM
    elif from_units == "angstrom" and to_units == "bohr":
        return distances / BOHR_TO_ANGSTROM
    else:
        raise ValueError(f"Unsupported conversion from {from_units} to {to_units}")
    
def convert_velocities(velocities: np.ndarray | torch.Tensor, from_units: str, to_units: str) -> np.ndarray | torch.Tensor:
    """ Convert velocities between different units.
    Args:
        velocities (np.ndarray or torch.Tensor): Velocities to convert.
        from_units (str): Units of the input velocities.
        to_units (str): Units to convert the velocities to.
    Returns:
        np.ndarray or torch.Tensor: Converted velocities.
    """
    if from_units == "bohr/fs" and to_units == "angstrom/fs":
        return velocities * BOHR_TO_ANGSTROM
    elif from_units == "angstrom/fs" and to_units == "bohr/fs":
        return velocities / BOHR_TO_ANGSTROM
    elif from_units == "angstrom/aut" and to_units == "angstrom/fs":
        return velocities / AUT_TO_FS
    elif from_units == "angstrom/fs" and to_units == "angstrom/aut":
        return velocities * AUT_TO_FS
    elif from_units == "bohr/aut" and to_units == "angstrom/fs":
        return velocities * BOHR_TO_ANGSTROM / AUT_TO_FS
    elif from_units == "angstrom/fs" and to_units == "bohr/aut":
        return velocities * AUT_TO_FS / BOHR_TO_ANGSTROM
    # elif from_units == "angstrom/fs" and to_units == "angstrom/ase_time":
    #     return velocities * AUT_TO_FS 
    else:
        raise ValueError(f"Unsupported conversion from {from_units} to {to_units}")
    
def convert_time(time: float, from_units: str, to_units: str) -> float:
    """ Convert time between different units.
    Args:
        time (float): Time to convert.
        from_units (str): Units of the input time.
        to_units (str): Units to convert the time to.
    Returns:
        float: Converted time.
    """
    if from_units == "aut" and to_units == "fs":
        return time * AUT_TO_FS
    elif from_units == "fs" and to_units == "aut":
        return time / AUT_TO_FS
    elif from_units == "aut" and to_units == "s":
        return time * AUT_TO_S
    elif from_units == "s" and to_units == "aut":
        return time / AUT_TO_S
    else:
        raise ValueError(f"Unsupported conversion from {from_units} to {to_units}")
    
if __name__ == "__main__":
    # Example usage
    energies = np.array([1.0, 2.0, 3.0])  # in Hartree
    converted_energies = convert_energies(energies, "hartree", "kcal/mol")
    reconverted_energies = convert_energies(converted_energies, "kcal/mol", "hartree")
    assert np.allclose(energies, reconverted_energies), "Energy conversion failed"

    forces = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=float)  # in Hartree/Bohr
    converted_forces = convert_forces(forces, "hartree/bohr", "kcal/mol/angstrom")
    reconverted_forces = convert_forces(converted_forces, "kcal/mol/angstrom", "hartree/bohr")
    assert torch.allclose(forces, reconverted_forces), "Forces conversion failed"

    distances = np.array([1.0, 2.0, 3.0])  # in Bohr
    converted_distances = convert_distances(distances, "bohr", "angstrom")
    reconverted_distances = convert_distances(converted_distances, "angstrom", "bohr")
    assert np.allclose(distances, reconverted_distances), "Distance conversion failed"

    velocities = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=float)  # in Bohr/fs
    converted_velocities = convert_velocities(velocities, "bohr/fs", "angstrom/fs")
    reconverted_velocities = convert_velocities(converted_velocities, "angstrom/fs", "bohr/fs")
    assert torch.allclose(velocities, reconverted_velocities), "Velocity conversion failed"

