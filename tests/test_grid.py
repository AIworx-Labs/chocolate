import unittest
from unittest.mock import MagicMock

from chocolate import Grid
from chocolate.space import Space, uniform, quantized_uniform, quantized_log
from chocolate.sample.grid import ParameterGrid


class TestParameterGrid(unittest.TestCase):
    def test_init_continuous(self):
        s = {"a": uniform(1, 2)}
        self.assertRaises(AssertionError, ParameterGrid, Space(s))

    def test_len(self):
        s = {"a": quantized_uniform(1, 2, 1)}
        grid = ParameterGrid(Space(s))
        self.assertEqual(len(grid), 1)

        s = {"a": quantized_uniform(1, 5, 1),
             "b": quantized_log(1, 5, 1, 10)}
        grid = ParameterGrid(Space(s))
        self.assertEqual(len(grid), 16)

        s = {"a": quantized_uniform(1, 5, 1),
             "b": {"c": quantized_log(1, 5, 1, 10),
                   "d": quantized_uniform(0, 1, 1)}}
        grid = ParameterGrid(Space(s))
        self.assertEqual(len(grid), 20)

        s = {"a": quantized_uniform(1, 5, 1),
             "b": {"c": quantized_log(1, 5, 1, 10),
                   "d": quantized_uniform(0, 10, 1)}}

        grid = ParameterGrid(Space(s))
        self.assertEqual(len(grid), 56)

    def test_getitem(self):
        s = {"a": quantized_uniform(1, 2, 1)}
        grid = ParameterGrid(Space(s))

        r = grid[0]
        self.assertAlmostEqual(r[0], 0.0)
        r = grid[-1]
        self.assertAlmostEqual(r[0], 0.0)

        s = {"a": quantized_uniform(1, 5, 1),
             "b": quantized_log(1, 5, 1, 10)}
        grid = ParameterGrid(Space(s))

        r = grid[0]
        self.assertAlmostEqual(r[0], 0.0)
        self.assertAlmostEqual(r[1], 0.0)

        r = grid[1]
        self.assertAlmostEqual(r[0], 0.25)
        self.assertAlmostEqual(r[1], 0.0)

        s = {"a": quantized_uniform(1, 5, 1),
             "b": {"c": quantized_log(1, 5, 1, 10),
                   "d": quantized_uniform(0, 10, 1)}}
        grid = ParameterGrid(Space(s))

        r = grid[0]
        self.assertEqual(len(r), 4)
        self.assertAlmostEqual(r[0], 0.0)
        self.assertAlmostEqual(r[1], 0.0)
        self.assertAlmostEqual(r[2], 0.0)
        self.assertAlmostEqual(r[3], None)

        r = grid[1]
        self.assertEqual(len(r), 4)
        self.assertAlmostEqual(r[0], 0.25)
        self.assertAlmostEqual(r[1], 0.0)
        self.assertAlmostEqual(r[2], 0.0)
        self.assertAlmostEqual(r[3], None)

        r = grid[16]
        self.assertEqual(len(r), 4)
        self.assertAlmostEqual(r[0], 0.0)
        self.assertAlmostEqual(r[1], 0.5)
        self.assertAlmostEqual(r[2], None)
        self.assertAlmostEqual(r[3], 0.0)

        r = grid[20]
        self.assertEqual(len(r), 4)
        self.assertAlmostEqual(r[0], 0.0)
        self.assertAlmostEqual(r[1], 0.5)
        self.assertAlmostEqual(r[2], None)
        self.assertAlmostEqual(r[3], 0.1)

        r = grid[-1]
        self.assertEqual(len(r), 4)
        self.assertAlmostEqual(r[0], 0.75)
        self.assertAlmostEqual(r[1], 0.5)
        self.assertAlmostEqual(r[2], None)
        self.assertAlmostEqual(r[3], 0.9)

    def test_invalid_getitem(self):
        s = {"a": quantized_uniform(1, 2, 1)}
        grid = ParameterGrid(Space(s))
        self.assertRaises(IndexError, grid.__getitem__, 1)
        self.assertRaises(IndexError, grid.__getitem__, -2)

    def test_empty_grid(self):
        grid = ParameterGrid(Space(dict()))
        self.assertRaises(IndexError, grid.__getitem__, 0)

    def test_empty_subspace(self):
        grid = ParameterGrid(Space({"a": {"b": None}}))
        self.assertRaises(IndexError, grid.__getitem__, 0)


class TestGrid(unittest.TestCase):
    def setUp(self):
        s = {"a": quantized_uniform(1, 10, 1)}

        self.mock_conn = MagicMock(name="connection")
        self.mock_conn.get_space.return_value = Space(s)

        self.grid = Grid(self.mock_conn, s)

    def test_next(self):
        self.mock_conn.count_results.return_value = 0
        token, p = self.grid.next()

        self.assertEqual(token, {"_chocolate_id": 0})
        self.assertEqual(p, {"a": 1})

        self.mock_conn.count_results.return_value = 1
        token, p = self.grid.next()
        self.assertEqual(token, {"_chocolate_id": 1})
        self.assertEqual(p, {"a": 2})

    def test_stop_iteration(self):
        self.mock_conn.count_results.return_value = 9
        self.assertRaises(StopIteration, self.grid.next)
