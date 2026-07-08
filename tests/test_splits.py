import unittest
from pathlib import Path
import sys

# Ensure core package is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.splits import make_splits, SPLIT_SEED

class TestSplits(unittest.TestCase):
    def setUp(self):
        # Generate 200 dummy participant IDs: P0001 to P0200
        self.dummy_pids = [f"P{i:04d}" for i in range(1, 201)]
        
    def test_determinism(self):
        """Verify that make_splits with same input and seed produces identical lists."""
        tr1, val1, te1 = make_splits(self.dummy_pids, seed=42)
        tr2, val2, te2 = make_splits(self.dummy_pids, seed=42)
        
        self.assertEqual(tr1, tr2)
        self.assertEqual(val1, val2)
        self.assertEqual(te1, te2)
        
    def test_different_seed_different_split(self):
        """Verify that changing the seed yields a different partition."""
        tr1, val1, te1 = make_splits(self.dummy_pids, seed=42)
        tr2, val2, te2 = make_splits(self.dummy_pids, seed=100)
        
        # Highly unlikely to be identical with different random seeds
        self.assertNotEqual(tr1, tr2)
        
    def test_disjointness_and_exhaustiveness(self):
        """Verify that splits have zero overlap and sum to the total coverage."""
        tr, val, te = make_splits(self.dummy_pids, seed=SPLIT_SEED)
        
        set_tr = set(tr)
        set_val = set(val)
        set_te = set(te)
        
        # Verify mutual exclusivity
        self.assertEqual(len(set_tr & set_val), 0)
        self.assertEqual(len(set_tr & set_te), 0)
        self.assertEqual(len(set_val & set_te), 0)
        
        # Verify exhaustiveness
        self.assertEqual(sorted(tr + val + te), sorted(self.dummy_pids))
        
    def test_stable_sizes(self):
        """Verify that split sizes are exactly 70% train, 15% val, and 15% test."""
        tr, val, te = make_splits(self.dummy_pids, seed=SPLIT_SEED)
        
        self.assertEqual(len(tr), 140)
        self.assertEqual(len(val), 30)
        self.assertEqual(len(te), 30)

if __name__ == "__main__":
    unittest.main()
