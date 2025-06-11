import torch

from schnetpack.transform import Transform


# for reference, see the schnetpack/transform/atomistic.py file in schnetpack

class StandardizeProperty(Transform):
    """
    Standardize scalar (e.g., energy, forces) and vector (e.g., positions) properties in the input data.
    This transform standardizes the properties using the provided mean and standard deviation values.
    """
    def __init__(
            self, 
            property_key: str, 
            property_mean: torch.Tensor, 
            property_std: torch.Tensor, 
        ):
        """
        Args:
            property_key (str): Key for property in the input data.
            property_mean (torch.Tensor): Mean value of property for standardization.
            property_std (torch.Tensor): Standard deviation of property for standardization.
        """
        super().__init__()
        self.property_key = property_key
        self.property_mean = property_mean
        self.property_std = property_std

    def forward(self, data: dict) -> dict:
        """
        Standardize the property in the input data.
        Args:
            data (dict): Input data containing property.
        Returns:
            dict: Data with standardized property.
        """
        data[self.property_key] = (data[self.property_key] - self.property_mean) / self.property_std
        return data

class RescaleProperty(Transform):
    """
    Rescale scalar (e.g., energy, forces) and vector (e.g., positions) properties in the input data.
    This transform rescales the properties using the provided mean and std values.
    """
    def __init__(
            self, 
            property_key: str, 
            property_mean: torch.Tensor, 
            property_std: torch.Tensor, 
        ):
        """
        Args:
            property_key (str): Key for property in the input data.
            property_mean (torch.Tensor): Mean value of property for standardization.
            property_std (torch.Tensor): Standard deviation of property for standardization.
        """
        super().__init__()
        self.property_key = property_key
        self.property_mean = property_mean
        self.property_std = property_std

    def forward(self, data: dict) -> dict:
        """
        Rescale the property in the input data.
        Args:
            data (dict): Input data containing property.
        Returns:
            dict: Data with rescaled property.
        """
        data[self.property_key] = (data[self.property_key] * self.property_std) + self.property_mean
        return data
