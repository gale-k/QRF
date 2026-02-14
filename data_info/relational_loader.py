import data_info.datasets.toy_dataset
from data_info.datasets.toy_dataset import ToyRelationalQRF


def load_relational_dataset(name: str):

    if name in ["toy_machines", "toy_father", "toy_cancer"]:
        return ToyRelationalQRF(name)

    raise ValueError(f"Unknown relational dataset: {name}")
