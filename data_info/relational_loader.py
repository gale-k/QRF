from data_info.datasets.toy_dataset import ToyRelationalQRF


def load_relational_dataset(name: str):

    if name in ["boston_housing", "california_housing", "citeseer", "cora", "drug_interactions", 
                "financial_nlp_small", "icml", "nell_sports", "roofworld20", "toy_cancer", 
                "toy_father", "toy_machines", "uwcse", "webkb"]:
        return ToyRelationalQRF(name)

    raise ValueError(f"Unknown relational dataset: {name}")
