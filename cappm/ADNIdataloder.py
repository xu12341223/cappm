from data_utils import MultiModalDataset

class ADNI(MultiModalDataset):
    """ADNI features use medians and standardization fitted only on training data."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, standardize=True, **kwargs)
