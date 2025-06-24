"""
Physical and unit conversion constants. The hard coded reference values are taken from Wikipedia
"""

from ase import units

# Unit conversions
BOHR_TO_ANGSTROM = units.Bohr  # 1 Bohr = 0.529177210544 Å
HARTREE_TO_KCAL_MOL = units.Hartree * units.mol / units.kcal  # 1 Hartree = 627.5094740631 kcal/mol
HARTREE_TO_EV = units.Hartree # 1 eV = 27.211386245988 Hartree
AUT_TO_FS = units._aut * 1e15 # 1 AUT = 2.4188843265864e-2 fs
AUT_TO_S = units._aut * 1e-15  # 1 AUT = 2.4188843265864e-17 s


# Tolerance for floating point comparison
FLOAT_TOL = 1e-6

assert abs(BOHR_TO_ANGSTROM - 0.529177210544) < FLOAT_TOL, "BOHR_TO_ANGSTROM is not set correctly"
assert abs(HARTREE_TO_KCAL_MOL - 627.5094740631) < FLOAT_TOL, "HARTREE_TO_KCAL_MOL is not set correctly"
assert abs(HARTREE_TO_EV - 27.211386245988) < FLOAT_TOL, "HARTREE_TO_EV is not set correctly"
assert abs(AUT_TO_FS - 2.4188843265864e-2) < FLOAT_TOL, "AUT_TO_FS is not set correctly"
assert abs(AUT_TO_S - 2.4188843265864e-17) < FLOAT_TOL, "AUT_TO_S is not set correctly"
