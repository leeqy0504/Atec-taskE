"""Regression checks for ACT start-index sampling without Isaac Sim."""

import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

from train_task_e import DemoDataset_ACT


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.temp.name) / "demos.hdf5")
        with h5py.File(self.path, "w") as data:
            for i in range(2):
                group = data.create_group(f"traj_{i}")
                states = np.full((40, 8), i, dtype=np.float32)
                actions = np.full((40, 8), i * 2, dtype=np.float32)
                states[12:21, 0] += np.arange(9, dtype=np.float32) * 0.01
                states[21:, 0] += 0.08
                actions[25:, 6] += 0.07
                group.create_dataset("obs", data=states)
                group.create_dataset("actions", data=actions)
                group.attrs["phase_intervals"] = json.dumps([
                    {"object": i + 1, "state": "CLOSE", "start_step": 0, "end_step": 7},
                    {"object": i + 1, "state": "LIFT", "start_step": 7, "end_step": 40},
                ])

    def tearDown(self):
        self.temp.cleanup()

    def load(self, mode="all", **kwargs):
        return DemoDataset_ACT(self.path, 8, include_rgb=False, sample_mode=mode, **kwargs)

    def test_default_mode_and_statistics(self):
        original = self.load()
        sampled = self.load("motion_phase")
        self.assertEqual(original.slices, [(i, t) for i in range(2) for t in range(40)])
        self.assertLess(len(sampled), len(original))
        for key, value in original.norm_stats.items():
            torch.testing.assert_close(value, sampled.norm_stats[key], rtol=0, atol=0)

    def test_phase_anchors_and_motion_are_retained(self):
        dataset = self.load("motion_phase")
        self.assertEqual(dataset.slices, sorted(set(dataset.slices)))
        for i in range(2):
            starts = {t for traj, t in dataset.slices if traj == i}
            self.assertTrue({0, 7, 39}.issubset(starts))
            self.assertTrue(set(range(12, 20)).issubset(starts))
            self.assertIn(24, starts)  # A gripper action change with no joint motion.
            self.assertTrue(all(0 <= t < 40 for t in starts))

    def test_chunks_follow_source_time_and_never_cross_trajectories(self):
        dataset = self.load("motion_phase")
        with h5py.File(self.path, "r") as data:
            for index, (traj, step) in enumerate(dataset.slices):
                sample = dataset[index]
                actual = (sample["actions"] * dataset.norm_stats["action_std"]
                          + dataset.norm_stats["action_mean"])
                expected = data[f"traj_{traj}/actions"][step:step + 8]
                if len(expected) < 8:
                    expected = np.concatenate([
                        expected, np.repeat(expected[-1:], 8 - len(expected), axis=0),
                    ])
                np.testing.assert_allclose(actual.numpy(), expected, atol=1e-6, rtol=1e-6)
                state = (sample["observations"]["state"] * dataset.norm_stats["state_std"][0]
                         + dataset.norm_stats["state_mean"][0])
                np.testing.assert_allclose(state.numpy(), data[f"traj_{traj}/obs"][step], atol=1e-6)

    def test_legacy_metadata_and_argument_validation(self):
        with h5py.File(self.path, "a") as data:
            for group in data.values():
                del group.attrs["phase_intervals"]
        self.assertGreater(len(self.load("motion_phase")), 0)
        with self.assertRaises(ValueError):
            self.load("unsupported")
        with self.assertRaises(ValueError):
            self.load("motion_phase", static_start_stride=0)


if __name__ == "__main__":
    unittest.main()
