"""Geometry CLI unit fixtures; no GMR solve, simulator, or policy execution."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("experiment", Path(__file__).resolve().parents[1] / "tools/experiment.py")
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


class ExperimentTests(unittest.TestCase):
    def test_output_boundary_and_overwrite(self):
        root = experiment.PROJECT
        self.assertEqual(experiment.output_path("outputs/run01"), root / "outputs/run01")
        with self.assertRaises(ValueError):
            experiment.output_path("inputs/modified.npy")
        with self.assertRaises(ValueError):
            experiment.output_path("outputs/../../outside")
        with self.assertRaises(FileExistsError):
            experiment.new_files([Path(__file__)])

    def test_common_clock_and_position_interpolation(self):
        np, _, _ = experiment.geometry()
        human = (np.arange(4) + 1) / 30.
        objects = np.arange(4) / 30.
        times, qh, qo = experiment.common_clock(human, objects, 30.)
        np.testing.assert_allclose(times, [1 / 30., 2 / 30., 3 / 30.], atol=1e-15)
        np.testing.assert_array_equal(times, qh)
        np.testing.assert_array_equal(times, qo)
        values = np.arange(12).reshape(4, 3)
        actual = experiment.linear(values, human, times)
        np.testing.assert_allclose(actual, values[:3])
        with self.assertRaises(ValueError):
            experiment.common_clock(np.array([0., 0.]), np.array([0., 1.]), 30.)

    def test_align_trusted_source_fixture(self):
        np, _, _ = experiment.geometry()
        original_project, original_datasets = experiment.PROJECT, experiment.DATASETS
        (original_project / "outputs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="geometry_test_", dir=original_project / "outputs") as temporary:
            root = Path(temporary)
            experiment.PROJECT, experiment.DATASETS = root, root / "configs/datasets.json"
            try:
                (root / "inputs").mkdir();(root / "configs").mkdir()
                count = 4
                human = np.zeros((1, count, 24, 3));human[0, :, :, 0] = np.arange(count)[:, None] * .1 + 5
                raw = dict(metadata=dict(fps=30., position_units="meter", axis_angle_units="radian", quaternion_order="wxyz",
                                         pose_joint_names=experiment.NAMES, pose_joint_parent_indices=experiment.PARENTS),
                           human=dict(joint_xyz=human, pose_aa=np.zeros((1, count, 22, 3))),
                           object=dict(floorlamp=dict(xyz=np.zeros((1, count, 3)), rotation=np.tile([1., 0., 0., 0.], (1, count, 1)))))
                np.save(root / "inputs/motion.npy", raw)
                np.savez(root / "inputs/clock.npz", measured_human_reference_time_s=(np.arange(count) + 1) / 30.,
                         derived_object_interpolation_time_s=np.arange(count) / 30., raw_recorder_index=np.arange(count))
                cfg = dict(datasets=dict(test=dict(source_npy="inputs/motion.npy", clock_npz="inputs/clock.npz", rest_mesh="inputs/mesh.ply",
                                                   clock_kind="pick_place_clock_map", scale=.9, origin_policy="aligned_first")))
                experiment.DATASETS.write_text(json.dumps(cfg))
                with contextlib.redirect_stdout(io.StringIO()):
                    experiment.align("test", "outputs/run01")
                with np.load(root / "outputs/run01/aligned.npz", allow_pickle=False) as aligned:
                    self.assertEqual(aligned["human_joint_pos"].shape, (3, 24, 3))
                    np.testing.assert_allclose(aligned["origin_xyz"], [5., 0., 0.])
                    np.testing.assert_allclose(aligned["object_quat_wxyz"], np.tile([1., 0., 0., 0.], (3, 1)))
                report = json.loads((root / "outputs/run01/alignment_report.json").read_text())
                self.assertEqual(report["status"], "PASS_NUMERIC_ALIGNMENT")
                self.assertFalse(report["extrapolation_used"])
                with self.assertRaises(FileExistsError):
                    experiment.align("test", "outputs/run01")
            finally:
                experiment.PROJECT, experiment.DATASETS = original_project, original_datasets


if __name__ == "__main__":
    unittest.main(verbosity=2)
