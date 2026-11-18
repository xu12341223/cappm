from data_utils import MultiModalDataset

class ANM(MultiModalDataset):
    """ANMerge features retain their original scale; missing values use training medians."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, standardize=False, **kwargs)
