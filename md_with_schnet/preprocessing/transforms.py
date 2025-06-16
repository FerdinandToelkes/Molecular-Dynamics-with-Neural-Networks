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
            property_mean: torch.Tensor | None = None, 
            property_std: torch.Tensor | None = None, 
            path_to_stats: str = None,
        ):
        """
        Args:
            property_key (str): Key for property in the input data.
            property_mean (torch.Tensor | None): Mean value of property for standardization.
            property_std (torch.Tensor | None): Standard deviation of property for standardization.
            path_to_stats (str | None): Path to the statistics file (mean and std) for standardization.
        """
        super().__init__()
        self.property_key = property_key

        if path_to_stats is not None:
            # Load the mean and std from the stats file
            stats = torch.load(path_to_stats, weights_only=True)
            self.property_mean = stats[f'{property_key}_mean']
            self.property_std = stats[f'{property_key}_std']
        else:
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
            property_mean: torch.Tensor | None = None, 
            property_std: torch.Tensor | None = None, 
            path_to_stats: str | None = None,
        ):
        """
        Args:
            property_key (str): Key for property in the input data.
            property_mean (torch.Tensor | None): Mean value of property for standardization.
            property_std (torch.Tensor | None): Standard deviation of property for standardization.
            path_to_stats (str | None): Path to the statistics file (mean and std) for standardization.
        """
        super().__init__()
        self.property_key = property_key
        
        if path_to_stats is not None:
            # Load the mean and std from the stats file
            stats = torch.load(path_to_stats, weights_only=True)
            self.property_mean = stats[f'{property_key}_mean']
            self.property_std = stats[f'{property_key}_std']
        else:
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
    
