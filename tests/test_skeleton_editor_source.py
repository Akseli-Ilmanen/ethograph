"""The skeleton editor can draw on the dataset's position when no video overlay exists."""

from types import SimpleNamespace

import numpy as np

from ethograph.gui.plots_skeleton import pose_for_editor


class _Store:
    """A 3D pose whose y axis is flat: the editor should see x/z."""

    def __init__(self):
        t = np.linspace(0, 1, 5)
        self.axes = {"x": np.outer(t, [0, 1, 2]), "y": np.zeros((5, 3)), "z": np.outer(t, [5, 0, 3])}

    def feature_dims(self, feature):
        return {"space": ["x", "y", "z"], "keypoints": ["a", "b", "c"]}

    def select(self, feature, selection, t0, t1):
        return SimpleNamespace(data=self.axes[selection["space"]], time=np.arange(5), dim_labels=["a", "b", "c"])


def test_3d_pose_is_projected_onto_its_widest_plane():
    app_state = SimpleNamespace(data_loader=_Store(), window_bounds=SimpleNamespace(start_s=0.0, end_s=1.0))
    keypoints, positions = pose_for_editor(app_state)
    assert keypoints == ["a", "b", "c"]
    assert positions.shape == (5, 3, 2)
    np.testing.assert_array_equal(positions[:, :, 1], app_state.data_loader.axes["z"])


def test_editor_config_with_numpy_keypoint_names_saves(qapp, tmp_path):
    from ethograph.gui.dialog_skeleton_editor import SkeletonEditorDialog
    from ethograph.skeleton.library import load_skeletons, save_skeleton

    keypoints = list(np.array(["beak", "head"]))  # np.str_, as the pose loaders hand them over
    dialog = SkeletonEditorDialog(keypoints, np.zeros((2, 2, 2)))
    dialog.canvas._add_edge(0, 1)
    save_skeleton("crow", dialog.get_config(), tmp_path)
    assert load_skeletons(tmp_path)["crow"]["connections"][0]["end"] == "head"
