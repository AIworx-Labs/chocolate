
import unittest

import numpy

from chocolate.space import *

class Obj1(object):
    pass

class Obj2(object):
    pass

class TestSpace(unittest.TestCase):
    def test_numpy_array(self):
        u = uniform(0.0005, 0.1)
        l = log(1, 10, 2)
        s = Space({"a" : u,
                   "b" : l})

        x = numpy.array([0, 0.5])
        self.assertEqual(s(x), {"a" : 0.0005, "b" : 2**5.5})

    def test_empty_space(self):
        s = Space(dict())
        self.assertEqual(len(s), 0)

    def test_continuous(self):
        u = uniform(0.0005, 0.1)
        l = log(1, 10, 2)
        s = Space({"a" : u,
                   "b" : l})

        self.assertEqual(2, len(s))
        self.assertEqual(s.names(), ["a", "b"])
        self.assertEqual(s.isdiscrete(), False)
        self.assertEqual(s.subspaces(), [[u, l]])

        x = [0, 0.5]
        self.assertEqual(s(x), {"a" : 0.0005, "b" : 2**5.5})


    def test_discrete(self):
        qu = quantized_uniform(0.01, 0.1, 0.02)
        ql = quantized_log(1, 10, 1, 10)
        s = Space({"a" : qu,
                   "b" : ql})

        self.assertEqual(2, len(s))
        self.assertEqual(s.names(), ["a", "b"])
        self.assertEqual(s.isdiscrete(), True)
        self.assertEqual(s.subspaces(), [[qu, ql]])

        x = [0, 0.5]
        self.assertEqual(s(x), {"a" : 0.01, "b" : 10**5})

    def test_contitional_continuous(self):
        u = uniform(0.01, 0.1)
        l = log(1, 10, 10)
        qu = quantized_uniform(0.01, 0.1, 0.02)
        ql = quantized_log(1, 10, 1, 10)
        s = Space([{"k1" : "a", "k2" : "b",
                        "a" : u,
                        "b" : l},
                   {"k1" : "a", "k2" : "c",
                        "a" : qu,
                        "c" : ql}])

        self.assertEqual(5, len(s))
        self.assertEqual(s.names(), ["_subspace", "k1_a_k2_b_a", "k1_a_k2_b_b", "k1_a_k2_c_a", "k1_a_k2_c_c"])
        self.assertEqual(s.isdiscrete(), False)
        self.assertEqual(s.subspaces(), [[0.0, u, l, None, None], [0.5, None, None, qu, ql]])

        x = [0, 0.1, 0.7, 0.0, 0.0]
        res = s(x)
        self.assertEqual(res["k1"], "a")
        self.assertEqual(res["k2"], "b")
        self.assertAlmostEqual(res["a"], 0.019)
        self.assertAlmostEqual(res["b"], 10**7.3)

        x = [0.5, 0.0, 0.0, 0.25, 0.7]
        res = s(x)
        self.assertEqual(res["k1"], "a")
        self.assertEqual(res["k2"], "c")
        self.assertAlmostEqual(res["a"], 0.03)
        self.assertAlmostEqual(res["c"], 10**7)

    def test_conditional_discrete(self):
        qu1 = quantized_uniform(0.01, 0.1, 0.02)
        ql1 = quantized_log(1, 10, 1, 10)
        qu2 = quantized_uniform(0.1, 1, 0.2)
        ql2 = quantized_log(0, 9, 1, 10)

        s = Space([{"k1" : "a", "k2" : "b",
                           "a" : qu1,
                           "b" : ql1},
                   {"k1" : "a", "k2" : "c",
                           "a" : qu2,
                           "c" : ql2}])

        self.assertEqual(5, len(s))
        self.assertEqual(s.names(), ["_subspace", "k1_a_k2_b_a", "k1_a_k2_b_b", "k1_a_k2_c_a", "k1_a_k2_c_c"])
        self.assertEqual(s.isdiscrete(), True)
        self.assertEqual(s.subspaces(), [[0.0, qu1, ql1, None, None], [0.5, None, None, qu2, ql2]])

        x = [0, 0.3, 0.7, 0.0, 0.0]
        res = s(x)
        self.assertEqual(res["k1"], "a")
        self.assertEqual(res["k2"], "b")
        self.assertAlmostEqual(res["a"], 0.03)
        self.assertAlmostEqual(res["b"], 10**7)

        x = [0.5, 0.0, 0.0, 0.25, 0.7]
        res = s(x)
        self.assertEqual(res["k1"], "a")
        self.assertEqual(res["k2"], "c")
        self.assertAlmostEqual(res["a"], 0.3)
        self.assertAlmostEqual(res["c"], 10**6)

    def test_multi_conditional(self):
        l1 = log(low=-3, high=5, base=10)
        l2 = log(low=-2, high=3, base=10)
        u = uniform(low=-1, high=1)
        qu = quantized_uniform(low=1, high=20, step=1)
        s = Space([{"algo" : {"svm" : {"C" : l1,
                                       "kernel" : {"linear" : None,
                                                   "rbf" : {"gamma" : l2}},
                                       "cond2" : {"aa" : None,
                                                  "bb" : {"abc" : u}}},
                              "knn" : {"n_neighbors" : qu}}}])

        self.assertEqual(7, len(s))
        self.assertEqual(s.names(), ['algo__subspace', 'algo_algo_knn_n_neighbors', 'algo_algo_svm_C', 'algo_algo_svm_cond2__subspace', 'algo_algo_svm_cond2_cond2_bb_abc', 'algo_algo_svm_kernel__subspace', 'algo_algo_svm_kernel_kernel_rbf_gamma'])
        self.assertEqual(s.isdiscrete(), False)
        self.assertEqual(s.subspaces(), [[0.0, qu, None, None, None, None, None], [0.5, None, l1, 0.0, None, 0.0, None], [0.5, None, l1, 0.0, None, 0.5, l2], [0.5, None, l1, 0.5, u, 0.0, None], [0.5, None, l1, 0.5, u, 0.5, l2]])

        x = [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
        res = s(x)
        self.assertEqual(res["algo"], "knn")
        self.assertAlmostEqual(res["n_neighbors"], 10)
        self.assertNotIn("C", res)

        x = [0.5, 0.0, 0.5, 0.0, 0.0, 0.5, 0.2]
        res = s(x)
        self.assertEqual(res["algo"], "svm")
        self.assertAlmostEqual(res["C"], 10)
        self.assertAlmostEqual(res["gamma"], 10**-1)
        self.assertEqual(res["cond2"], "aa")
        self.assertNotIn("aa", res)
        self.assertNotIn("bb", res)

        x = [0.5, 0.0, 0.5, 0.5, 0.3, 0.5, 0.2]
        res = s(x)
        self.assertEqual(res["algo"], "svm")
        self.assertAlmostEqual(res["C"], 10)
        self.assertAlmostEqual(res["gamma"], 10**-1)
        self.assertEqual(res["cond2"], "bb")
        self.assertAlmostEqual(res["abc"], -0.4)
        self.assertNotIn("aa", res)
        self.assertNotIn("bb", res)

    def test_equal(self):
        s1 = {"a" : uniform(1, 2),
              "b" : {"c" : {"c1" : quantized_log(0, 5, 1, 10)},
                     "d" : {"d1" : quantized_log(0, 5, 1, 2)}}}

        s2 = {"a" : uniform(1, 2),
              "b" : {"c" : {"c1" : quantized_log(0, 5, 1, 10)},
                     "d" : {"d1" : quantized_log(0, 5, 1, 2)}}}

        self.assertEqual(Space(s1), Space(s2))

    def test_equal_None(self):
        s1 = {"a" : uniform(1, 2),
              "b" : {"c" : {"c1" : quantized_log(0, 5, 1, 10)},
                     "d" : {"d1" : quantized_log(0, 5, 1, 2)}}}

        self.assertNotEqual(Space(s1), None)

    def test_not_equal(self):
        s1 = {"a" : uniform(1, 2),
              "b" : {"c" : {"c1" : quantized_log(0, 5, 1, 10)},
                     "d" : {"d1" : quantized_log(0, 5, 1, 2)}}}

        s2 = {"a" : uniform(1, 2),
              "b" : {"c" : {"c1" : quantized_log(0, 5, 1, 10)},
                     "d" : {"d2" : quantized_log(0, 5, 1, 2)}}}

        s3 = {"a" : uniform(1, 2),
              "b" : {"c" : {"c1" : quantized_log(0, 5, 1, 10)},
                     "d" : {"d1" : quantized_log(0, 5, 1, 8)}}}

        self.assertNotEqual(Space(s1), Space(s2))
        self.assertNotEqual(Space(s1), Space(s3))
        self.assertFalse(Space(s1) == Space(s2))
        self.assertFalse(Space(s1) == Space(s3))
        self.assertTrue(Space(s1) != Space(s2))
        self.assertTrue(Space(s1) != Space(s3))

    def test_invalid_parameter_name(self):
        s1 = {"" : uniform(1, 2)}
        s2 = {"a" : uniform(1, 2),
              "b" : {"c" : {"" : quantized_log(0, 5, 1, 10)},
                     "d" : {"d1" : quantized_log(0, 5, 1, 2)}}}

        self.assertRaises(RuntimeError, Space, s1)
        self.assertRaises(RuntimeError, Space, s2)

    def test_uniform(self):
        low, high = 0.01, 0.1
        dist = uniform(low, high)

        self.assertAlmostEqual(dist(0.0), low)
        self.assertAlmostEqual(dist(0.5), 0.5 * (high - low) + low)
        self.assertAlmostEqual(dist(0.999), 0.999 * (high - low) + low)

    def test_invalid_uniform(self):
        low, high = 1, 0
        self.assertRaises(AssertionError, uniform, low, high)

    def test_quantized_uniform(self):
        low, high, step = 0.01, 0.1, 0.02
        dist = quantized_uniform(low, high, step)
        # [0.01, 0.03, 0.05, 0.07, 0.09]

        self.assertAlmostEqual(dist(0.0), low)
        self.assertAlmostEqual(dist(0.5), 0.05)
        self.assertAlmostEqual(dist(0.999), 0.09)

        n_steps = (high - low) / step
        stop = numpy.ceil(n_steps) / n_steps
        target = numpy.linspace(0, stop, num=int(numpy.ceil(n_steps)), endpoint=False)
        for i, (t, it) in enumerate(zip(target, list(dist))):
            self.assertAlmostEqual(t, it)
            self.assertAlmostEqual(t, dist[i])

        self.assertEqual(5, len(dist))

    def test_invalid_quantized_uniform(self):
        low, high, step = 1, 0, 1
        self.assertRaises(AssertionError, quantized_uniform, low, high, step)

        low, high, step = 0, 1, 0
        self.assertRaises(AssertionError, quantized_uniform, low, high, step)

    def test_log(self):
        low, high, base = 1, 10, 2
        dist = log(low, high, base)

        # 2^low
        self.assertAlmostEqual(dist(0.0), 2**low)
        # 2^(0.5 * (high - low) + low)
        self.assertAlmostEqual(dist(0.5), 2**(0.5 * (high - low) + low))
        # 2^(0.999 * (high - low) + low)
        self.assertAlmostEqual(dist(0.999), 2**(0.999 * (high - low) + low))

    def test_invalid_log(self):
        low, high, base = 10, 1, 2
        self.assertRaises(AssertionError, log, low, high, base)

        low, high, base = 0, 10, 0
        self.assertRaises(AssertionError, log, low, high, base)

        low, high, base = 0, 10, 1
        self.assertRaises(AssertionError, log, low, high, base)


    def test_quantized_log(self):
        low, high, step, base = 1, 10, 3, 2
        dist = quantized_log(low, high, step, base)
        # [2**1, 2**4, 2**7]

        self.assertAlmostEqual(dist(0.0), 2**low)
        self.assertAlmostEqual(dist(0.5), 2**4)
        self.assertAlmostEqual(dist(0.999), 2**7)

        n_steps = (high - low) / step
        stop = numpy.ceil(n_steps) / n_steps
        target = numpy.linspace(0, stop, num=int(numpy.ceil(n_steps)), endpoint=False)
        for i, (t, it) in enumerate(zip(target, list(dist))):
            self.assertAlmostEqual(t, it)
            self.assertAlmostEqual(t, dist[i])

        self.assertEqual(3, len(dist))

    def test_invalid_quantized_log(self):
        low, high, step, base = 10, 1, 2, 10
        self.assertRaises(AssertionError, quantized_log, low, high, step, base)

        low, high, step, base = 0, 10, 0, 10
        self.assertRaises(AssertionError, quantized_log, low, high, step, base)

        low, high, step, base = 0, 10, 1, 0
        self.assertRaises(AssertionError, quantized_log, low, high, step, base)

        low, high, step, base = 0, 10, 1, 1
        self.assertRaises(AssertionError, quantized_log, low, high, step, base)

    def test_choice(self):
        data = ["a", 2, Obj1, [1,2,3]]
        dist = choice(data)

        self.assertAlmostEqual(dist(0.0), data[0])
        self.assertAlmostEqual(dist(0.34), data[1])
        self.assertAlmostEqual(dist(0.67), data[2])
        self.assertAlmostEqual(dist(0.999), data[3])

        target = numpy.linspace(0, 1, num=len(data), endpoint=False)
        for i, (t, it) in enumerate(zip(target, list(dist))):
            self.assertAlmostEqual(t, it)
            self.assertAlmostEqual(t, dist[i])

        self.assertEqual(4, len(dist))

    def test_invalid_choices(self):
        self.assertRaises(AssertionError, choice, [])

    def test_invalid_choices_call(self):
        dist = choice([1, 2, 3])
        self.assertRaises(AssertionError, dist, 1)
