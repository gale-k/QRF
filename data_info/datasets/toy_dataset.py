from relational_datasets import load
from data_info.encoder import AngleEncoder
import numpy as np
import re

# for each relational fact:
    # machine_has_component(m1,c2)
# computes:
    # QRF_Attention(m1, c2)
# where label is derived from:
    # defective(m1) ∈ pos ?

# imports for relational datasets: https://pypi.org/project/relational-datasets/

class RelationalQRF:

    def __init__(self, dataset_name):

        print(f"[INFO] Loading {dataset_name}...")
        self.train_ds, self.test_ds = load(dataset_name)

        self.dataset = self.train_ds

        self.facts = self.dataset.facts
        self.pos = set(self.dataset.pos)
        self.neg = set(self.dataset.neg)

        self.query_encoder = AngleEncoder(0, 1)
        self.key_encoder = AngleEncoder(0, 1)
        self.label_encoder = AngleEncoder(0, 1)

        print(f"[INFO] Facts: {len(self.facts)}")
        print(f"[INFO] Pos labels: {len(self.pos)}")
        print(f"[INFO] Neg labels: {len(self.neg)}")

    def __len__(self):
        return len(self.facts)

    def _parse_fact(self, fact_str):
        """
        Example:
            'machine_has_component(m1,c2)'
        Returns:
            predicate, arg1, arg2
        """

        match = re.match(r"(\w+)\(([^,]+),([^)]+)\)", fact_str)
        if not match:
            return None, None, None

        return match.group(1), match.group(2), match.group(3)

    def _entity_to_scalar(self, entity):
        """
        Deterministic numeric encoding for symbolic entities.
        """
        return (abs(hash(entity)) % 1000) / 1000.0

    def get_pair(self, idx):

        fact = self.facts[idx]

        predicate, arg1, arg2 = self._parse_fact(fact)

        if predicate is None:
            return 0, 0, 0

        query_scalar = self._entity_to_scalar(arg1)
        key_scalar = self._entity_to_scalar(arg2)

        query_angle = self.query_encoder.encode(query_scalar)
        key_angle = self.key_encoder.encode(key_scalar)

        # Label: check if unary predicate holds for arg1
        label_str = f"{predicate.split('_')[0]}({arg1})"

        if label_str in self.pos:
            label_scalar = 1
        elif label_str in self.neg:
            label_scalar = 0
        else:
            label_scalar = 0

        label = self.label_encoder.encode(label_scalar)

        return query_angle, key_angle, label


