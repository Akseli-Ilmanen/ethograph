"""Multi-camera calibration and triangulation to 3D, through aniposelib.

aniposelib does every piece of multi-view geometry; Ethograph only reads the
2D points, hands them over as arrays and writes the result down.

    from ethograph import triangulate as tri

    tri.import_dlc_calibration("DLC/3D_project", "my_project")   # once per rig
    tri.triangulate("my_project/session_01", "my_project")       # position_3d into the session's .nc
"""

from ethograph.triangulate.calibration import (
    CalibrationError,
    TriangulationUnavailableError,
    calibration_names,
    import_dlc_calibration,
    load_calibration,
    resolve_calibration,
)
from ethograph.triangulate.frame import WorldFrame, WorldFrameError, WorldTransform, load_frame, save_frame
from ethograph.triangulate.geometry import export_geometry, write_geometry
from ethograph.triangulate.points import METHODS, reproject_points, triangulate_points
from ethograph.triangulate.session import (
    POSITION_3D,
    REPROJECTION_ERROR,
    NoViewsError,
    gather_views,
    triangulate,
    triangulate_tree,
)
