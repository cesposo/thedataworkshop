"""
Tests for rotation data structures.
"""

import unittest
from dist_llm_train.scheduler.rotation import Rotation, RotationPoset


class TestRotation(unittest.TestCase):
    """Test cases for Rotation class."""

    def test_rotation_creation(self):
        """Test basic rotation creation."""
        rot = Rotation(
            id=0,
            edges=[('t1', 'w1'), ('t2', 'w2'), ('t3', 'w3')],
            eliminated_pairs={('t1', 'w2'), ('t2', 'w3')}
        )
        self.assertEqual(rot.id, 0)
        self.assertEqual(len(rot.edges), 3)
        self.assertEqual(len(rot.eliminated_pairs), 2)

    def test_rotation_serialization(self):
        """Test rotation to_dict and from_dict."""
        rot = Rotation(
            id=1,
            edges=[('t1', 'w1'), ('t2', 'w2')],
            eliminated_pairs={('t1', 'w3')}
        )
        data = rot.to_dict()
        restored = Rotation.from_dict(data)

        self.assertEqual(restored.id, rot.id)
        self.assertEqual(restored.edges, rot.edges)
        self.assertEqual(restored.eliminated_pairs, rot.eliminated_pairs)

    def test_rotation_repr(self):
        """Test rotation string representation."""
        rot = Rotation(id=0, edges=[('t1', 'w1'), ('t2', 'w2')])
        repr_str = repr(rot)
        self.assertIn('Rotation', repr_str)
        self.assertIn('t1', repr_str)


class TestRotationPoset(unittest.TestCase):
    """Test cases for RotationPoset class."""

    def setUp(self):
        """Set up test rotation poset."""
        self.poset = RotationPoset()

        # Create simple rotations
        self.rot0 = Rotation(id=0, edges=[('t1', 'w1'), ('t2', 'w2')])
        self.rot1 = Rotation(id=1, edges=[('t2', 'w2'), ('t3', 'w3')])
        self.rot2 = Rotation(id=2, edges=[('t3', 'w3'), ('t4', 'w4')])

        self.poset.add_rotation(self.rot0)
        self.poset.add_rotation(self.rot1)
        self.poset.add_rotation(self.rot2)

    def test_add_rotation(self):
        """Test adding rotations to poset."""
        self.assertEqual(len(self.poset.rotations), 3)
        self.assertIn(0, self.poset.rotations)
        self.assertIn(1, self.poset.rotations)
        self.assertIn(2, self.poset.rotations)

    def test_add_precedence(self):
        """Test adding precedence constraints."""
        # rot0 must come before rot1
        self.poset.add_precedence(0, 1)
        self.assertIn(0, self.poset.precedence_graph[1])

        # rot1 must come before rot2
        self.poset.add_precedence(1, 2)
        self.assertIn(1, self.poset.precedence_graph[2])

    def test_get_executable_rotations(self):
        """Test getting executable rotations."""
        # Initially, only rot0 is executable (no predecessors)
        self.poset.add_precedence(0, 1)
        self.poset.add_precedence(1, 2)

        executable = self.poset.get_executable_rotations(set())
        self.assertIn(0, executable)
        self.assertNotIn(1, executable)
        self.assertNotIn(2, executable)

        # After eliminating rot0, rot1 becomes executable
        executable = self.poset.get_executable_rotations({0})
        self.assertNotIn(0, executable)
        self.assertIn(1, executable)
        self.assertNotIn(2, executable)

        # After eliminating rot0 and rot1, rot2 becomes executable
        executable = self.poset.get_executable_rotations({0, 1})
        self.assertIn(2, executable)

    def test_apply_rotation(self):
        """Test applying rotation to a matching."""
        matching = {'t1': 'w1', 't2': 'w2', 't3': 'w3'}

        # Apply rot0: t1->w1, t2->w2 should cycle
        # After rotation: t1 gets w2's partner (w2), t2 gets w1's partner (w1)
        new_matching = self.poset.apply_rotation(matching, 0)

        # The rotation should swap assignments in the cycle
        self.assertIsNotNone(new_matching)
        # Matching should still have same tasks
        self.assertEqual(set(new_matching.keys()), set(matching.keys()))

    def test_topological_sort(self):
        """Test topological sorting of rotations."""
        # Create linear precedence: 0 -> 1 -> 2
        self.poset.add_precedence(0, 1)
        self.poset.add_precedence(1, 2)

        topo_order = self.poset.topological_sort()

        self.assertEqual(len(topo_order), 3)
        self.assertEqual(topo_order[0], 0)
        self.assertEqual(topo_order[1], 1)
        self.assertEqual(topo_order[2], 2)

    def test_topological_sort_parallel_rotations(self):
        """Test topological sort with parallel rotations."""
        # Create diamond: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
        rot3 = Rotation(id=3, edges=[('t5', 'w5')])
        self.poset.add_rotation(rot3)

        self.poset.add_precedence(0, 1)
        self.poset.add_precedence(0, 2)
        self.poset.add_precedence(1, 3)
        self.poset.add_precedence(2, 3)

        topo_order = self.poset.topological_sort()

        # 0 should come first
        self.assertEqual(topo_order[0], 0)
        # 3 should come last
        self.assertEqual(topo_order[3], 3)
        # 1 and 2 should be in the middle (order doesn't matter)
        self.assertIn(1, topo_order[1:3])
        self.assertIn(2, topo_order[1:3])

    def test_get_path_length(self):
        """Test path length calculation."""
        from_set = {0}
        to_set = {0, 1, 2}

        length = self.poset.get_path_length(from_set, to_set)
        self.assertEqual(length, 2)  # Need to add rotations 1 and 2

    def test_poset_serialization(self):
        """Test poset to_dict and from_dict."""
        self.poset.add_precedence(0, 1)
        self.poset.add_precedence(1, 2)

        data = self.poset.to_dict()
        restored = RotationPoset.from_dict(data)

        self.assertEqual(len(restored.rotations), len(self.poset.rotations))
        self.assertEqual(
            restored.precedence_graph[1],
            self.poset.precedence_graph[1]
        )

    def test_empty_poset(self):
        """Test operations on empty poset."""
        empty_poset = RotationPoset()

        executable = empty_poset.get_executable_rotations(set())
        self.assertEqual(executable, [])

        topo_order = empty_poset.topological_sort()
        self.assertEqual(topo_order, [])


if __name__ == '__main__':
    unittest.main()
