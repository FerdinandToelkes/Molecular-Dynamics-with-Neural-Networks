import torch

from schnetpack.transform import Transform


# for reference, see the schnetpack/transform/atomistic.py file in schnetpack

class StandardizeEnergy(Transform):
    """
    Standardize energy in the input data.
    This transform normalizes the energy using the provided mean and standard deviation values.
    """
    def __init__(
            self, 
            e_key: str, 
            mean_E: torch.Tensor, 
            std_E: torch.Tensor, 
        ):
        """
        Args:
            e_key (str): Key for energy in the input data.
            f_key (str): Key for forces in the input data.
            mean_E (torch.Tensor): Mean value of energy for standardization.
            std_E (torch.Tensor): Standard deviation of energy for standardization.
        """
        super().__init__()
        self.e_key = e_key
        self.mean_E = mean_E
        self.std_E = std_E

    def forward(self, data: dict) -> dict:
        """
        Standardize the energy in the input data.
        Args:
            data (dict): Input data containing energy.
        Returns:
            dict: Data with standardized energy.
        """
        data[self.e_key] = (data[self.e_key] - self.mean_E) / self.std_E
        return data


class StandardizeForces(Transform):
    """
    Standardize forces in the input data.
    This transform normalizes the forces using the provided mean and standard deviation values.
    """
    def __init__(
            self, 
            f_key: str, 
            mean_F: torch.Tensor, 
            std_F: torch.Tensor
        ):
        """
        Args:
            f_key (str): Key for forces in the input data.
            mean_F (torch.Tensor): Mean values of forces for standardization.
            std_F (torch.Tensor): Standard deviations of forces for standardization.
        """
        super().__init__()
        self.f_key = f_key
        self.mean_F = mean_F
        self.std_F = std_F

    def forward(self, data: dict) -> dict:
        """
        Standardize the forces in the input data.
        Args:
            data (dict): Input data containing forces.
        Returns:
            dict: Data with standardized forces.
        """
        data[self.f_key] = (data[self.f_key] - self.mean_F) / self.std_F
        return data
