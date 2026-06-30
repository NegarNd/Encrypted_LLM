from dataclasses import dataclass

@dataclass
class Counter:
    rotations: int = 0
    ct_pt_mult: int = 0
    ct_ct_mult: int = 0

    def reset(self):
        self.rotations = 0
        self.ct_pt_mult = 0
        self.ct_ct_mult = 0

counter = Counter()