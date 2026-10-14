# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""CPU-only guards for the cross-interpreter simulator boundary."""

import importlib.util
import pickle
import unittest
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[3] / "rlinf/envs/isaaclab/portable_arrays.py"
SPEC = importlib.util.spec_from_file_location("portable_arrays", PATH)
CODEC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CODEC)


class PortableArraysTests(unittest.TestCase):
    def test_observation_and_terminal_tree_roundtrip(self):
        value = (
            {
                "policy": {
                    "rgb": np.arange(48, dtype=np.uint8).reshape(2, 2, 4, 3),
                    "state": np.ones((2, 8), dtype=np.float32),
                }
            },
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([True, False]),
            np.array([False, True]),
            {"log": {}},
        )
        wire = CODEC.encode(value)
        restored = CODEC.decode(pickle.loads(pickle.dumps(wire, protocol=4)))
        self.assertIsInstance(restored, tuple)
        np.testing.assert_array_equal(
            restored[0]["policy"]["rgb"], value[0]["policy"]["rgb"]
        )
        np.testing.assert_array_equal(restored[2], value[2])
        self.assertEqual(restored[1].dtype, value[1].dtype)
        self.assertFalse(np.shares_memory(restored[1], value[1]))

    def test_reset_ids_keep_int64_and_empty_shape(self):
        value = {"env_ids": np.array([], dtype=np.int64), "seed": 0}
        decoded = CODEC.decode(CODEC.encode(value))
        self.assertEqual(decoded["env_ids"].dtype, np.dtype("int64"))
        self.assertEqual(decoded["env_ids"].shape, (0,))
        self.assertEqual(decoded["seed"], 0)

    def test_reject_python_object_arrays(self):
        with self.assertRaisesRegex(TypeError, "object arrays"):
            CODEC.encode(np.array([object()], dtype=object))
        with self.assertRaisesRegex(TypeError, "object arrays"):
            CODEC.decode({"__ndarray_v1__": ("O", (1,), b"12345678")})

    def test_wire_uses_only_builtin_types_not_numpy_pickle_classes(self):
        value = CODEC.encode(np.ones((2, 7), dtype=np.float32))
        encoded = pickle.dumps(value, protocol=4)
        self.assertNotIn(b"numpy", encoded)
        self.assertNotIn(b"torch", encoded)


if __name__ == "__main__":
    unittest.main()
