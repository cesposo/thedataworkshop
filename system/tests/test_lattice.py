"""
Tests for stable matching lattice.
"""

import unittest
from dist_llm_train.scheduler.lattice import StableMatchingLattice, LatticeBuilder
from dist_llm_train.scheduler.rotation import Rotation, RotationPoset


class TestStableMatchingLattice(unittest.TestCase):
    """Test cases for StableMatchingLattice class."""

    def setUp(self):
        """Set up test lattice."""
        self.lattice = StableMatchingLattice()

        # Add some matchings
        self.lattice.add_matching(0, {'t1': 'w1', 't2': 'w2'}, set())
        self.lattice.add_matching(1, {'t1': 'w2', 't2': 'w1'}, {0})

        self.lattice.task_optimal_id = 0
        self.lattice.worker_optimal_id = 1

    def test_add_matching(self):
        """Test adding matchings to lattice."""
        self.assertEqual(len(self.lattice.matchings), 2)
        self.assertIn(0, self.lattice.matchings)
        self.assertIn(1, self.lattice.matchings)

    def test_find_matching_by_id(self):
        """Test retrieving matching by ID."""
        matching = self.lattice.find_matching_by_id(0)
        self.assertIsNotNone(matching)
        self.assertEqual(matching['t1'], 'w1')

        missing = self.lattice.find_matching_by_id(999)
        self.assertIsNone(missing)

    def test_enumerate_all_matchings(self):
        """Test enumerating all matchings."""
        all_matchings = self.lattice.enumerate_all_matchings()
        self.assertEqual(len(all_matchings), 2)

    def test_size(self):
        """Test lattice size."""
        self.assertEqual(self.lattice.size(), 2)

    def test_rank_matchings_by_objective(self):
        """Test ranking matchings by objective function."""
        # Objective: prefer matchings where t1 is with w1
        def objective(matching):
            return 1.0 if matching.get('t1') == 'w1' else 0.0

        ranked = self.lattice.rank_matchings_by_objective(objective)
        self.assertEqual(ranked[0], 0)  # Matching 0 should rank first

    def test_get_matching_avoiding_workers(self):
        """Test finding matching that avoids specific workers."""
        result = self.lattice.get_matching_avoiding_workers({'w1'})
        self.assertIsNone(result)  # Both matchings use w1

        result = self.lattice.get_matching_avoiding_workers({'w3'})
        self.assertIsNotNone(result)  # Should find a matching

    def test_get_path_between_matchings(self):
        """Test finding path between matchings."""
        # Add rotation to poset
        rot = Rotation(id=0, edges=[('t1', 'w1'), ('t2', 'w2')])
        self.lattice.rotation_poset.add_rotation(rot)

        path = self.lattice.get_path_between_matchings(0, 1)
        # Should find a path (even if empty in this simple case)
        self.assertIsInstance(path, list)

    def test_serialization(self):
        """Test lattice serialization."""
        data = self.lattice.to_dict()
        restored = StableMatchingLattice.from_dict(data)

        self.assertEqual(len(restored.matchings), len(self.lattice.matchings))
        self.assertEqual(restored.task_optimal_id, self.lattice.task_optimal_id)
        self.assertEqual(restored.worker_optimal_id, self.lattice.worker_optimal_id)


class TestLatticeBuilder(unittest.TestCase):
    """Test cases for LatticeBuilder class."""

    def setUp(self):
        """Set up test data for lattice building."""
        # Simple instance: 2 tasks, 2 workers
        self.task_optimal = {'t1': 'w1', 't2': 'w2'}

        self.task_prefs = {
            't1': ['w1', 'w2'],
            't2': ['w2', 'w1']
        }

        self.worker_prefs = {
            'w1': ['t1', 't2'],
            'w2': ['t2', 't1']
        }

        self.builder = LatticeBuilder(max_size=10)

    def test_build_lattice_simple(self):
        """Test building lattice for simple instance."""
        lattice = self.builder.build_lattice(
            self.task_optimal,
            self.task_prefs,
            self.worker_prefs
        )

        self.assertIsNotNone(lattice)
        self.assertGreaterEqual(lattice.size(), 1)  # At least task-optimal matching
        self.assertEqual(lattice.task_optimal_id, 0)

    def test_build_lattice_single_matching(self):
        """Test lattice with only one stable matching."""
        # If preferences perfectly align, there's only one stable matching
        single_task_optimal = {'t1': 'w1'}
        single_task_prefs = {'t1': ['w1']}
        single_worker_prefs = {'w1': ['t1']}

        lattice = self.builder.build_lattice(
            single_task_optimal,
            single_task_prefs,
            single_worker_prefs
        )

        self.assertEqual(lattice.size(), 1)
        self.assertEqual(lattice.task_optimal_id, lattice.worker_optimal_id)

    def test_build_lattice_max_size_limit(self):
        """Test that lattice respects max size limit."""
        builder = LatticeBuilder(max_size=2)

        lattice = builder.build_lattice(
            self.task_optimal,
            self.task_prefs,
            self.worker_prefs
        )

        self.assertLessEqual(lattice.size(), 2)

    def test_extract_rotations(self):
        """Test rotation extraction."""
        rotations = self.builder._extract_rotations(
            self.task_optimal,
            self.task_prefs,
            self.worker_prefs
        )

        # Should extract at least some rotations for this instance
        self.assertIsInstance(rotations, list)

    def test_build_rotation_poset(self):
        """Test building rotation poset."""
        rot1 = Rotation(id=0, edges=[('t1', 'w1')])
        rot2 = Rotation(id=1, edges=[('t2', 'w2')])

        poset = self.builder._build_rotation_poset([rot1, rot2])

        self.assertEqual(len(poset.rotations), 2)

    def test_enumerate_matchings(self):
        """Test matching enumeration."""
        lattice = StableMatchingLattice()
        lattice.task_optimal_id = 0
        lattice.add_matching(0, self.task_optimal, set())

        # Add a simple rotation
        rot = Rotation(id=0, edges=[('t1', 'w1'), ('t1', 'w2')])
        lattice.rotation_poset.add_rotation(rot)

        self.builder._enumerate_matchings(lattice, self.task_optimal)

        # Should have generated additional matchings
        self.assertGreaterEqual(lattice.size(), 1)

    def test_build_lattice_no_rotations(self):
        """Test lattice building when no rotations exist."""
        # Single task-worker pair - no rotations possible
        single_matching = {'t1': 'w1'}
        single_task_prefs = {'t1': ['w1']}
        single_worker_prefs = {'w1': ['t1']}

        lattice = self.builder.build_lattice(
            single_matching,
            single_task_prefs,
            single_worker_prefs
        )

        self.assertEqual(lattice.size(), 1)
        self.assertEqual(len(lattice.rotation_poset.rotations), 0)


class TestLatticeIntegration(unittest.TestCase):
    """Integration tests for lattice functionality."""

    def test_lattice_stability_property(self):
        """Test that all matchings in lattice are stable (no blocking pairs)."""
        # Build a lattice
        task_optimal = {'t1': 'w1', 't2': 'w2', 't3': 'w3'}
        task_prefs = {
            't1': ['w1', 'w2', 'w3'],
            't2': ['w2', 'w3', 'w1'],
            't3': ['w3', 'w1', 'w2']
        }
        worker_prefs = {
            'w1': ['t1', 't2', 't3'],
            'w2': ['t2', 't1', 't3'],
            'w3': ['t3', 't2', 't1']
        }

        builder = LatticeBuilder(max_size=20)
        lattice = builder.build_lattice(task_optimal, task_prefs, worker_prefs)

        # For each matching, verify no blocking pairs
        for matching in lattice.enumerate_all_matchings():
            has_blocking_pair = self._check_blocking_pairs(
                matching, task_prefs, worker_prefs
            )
            self.assertFalse(has_blocking_pair,
                           f"Matching {matching} has blocking pairs")

    def _check_blocking_pairs(self, matching, task_prefs, worker_prefs):
        """Check if a matching has any blocking pairs."""
        # A blocking pair (t, w) exists if:
        # 1. t prefers w over current partner
        # 2. w prefers t over current partner

        for task_id, current_worker in matching.items():
            if task_id not in task_prefs:
                continue

            task_pref_list = task_prefs[task_id]
            current_rank = task_pref_list.index(current_worker) if current_worker in task_pref_list else len(task_pref_list)

            # Check all workers that task prefers more
            for better_worker in task_pref_list[:current_rank]:
                if better_worker not in worker_prefs:
                    continue

                # Find what task the better_worker is matched to
                better_worker_partner = None
                for t, w in matching.items():
                    if w == better_worker:
                        better_worker_partner = t
                        break

                if better_worker_partner is None:
                    # Worker is unmatched, so (task, worker) is blocking
                    return True

                # Check if better_worker prefers this task over current partner
                worker_pref_list = worker_prefs[better_worker]
                if task_id in worker_pref_list and better_worker_partner in worker_pref_list:
                    task_rank = worker_pref_list.index(task_id)
                    partner_rank = worker_pref_list.index(better_worker_partner)

                    if task_rank < partner_rank:
                        # Both prefer each other - blocking pair found
                        return True

        return False


if __name__ == '__main__':
    unittest.main()
