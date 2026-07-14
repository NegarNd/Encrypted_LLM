from dataclasses import dataclass

@dataclass
class Counter:
    rotations: int = 0
    ct_pt_mult: int = 0
    ct_ct_mult: int = 0
    conjugations: int = 0

    def reset(self):
        self.rotations = 0
        self.ct_pt_mult = 0
        self.ct_ct_mult = 0
        self.conjugations = 0

    def snapshot(self) -> dict:
        """Return a plain dict copy of the current op counts."""
        return {
            "rotations": self.rotations,
            "ct_pt_mult": self.ct_pt_mult,
            "ct_ct_mult": self.ct_ct_mult,
            "conjugations": self.conjugations,
        }

    def delta_since(self, snap: dict) -> dict:
        """Return the per-field increase since a previous snapshot()."""
        cur = self.snapshot()
        return {k: cur[k] - snap[k] for k in cur}

counter = Counter()