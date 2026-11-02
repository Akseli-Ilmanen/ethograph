"""Which media folder the Data wizard proposes as the session folder — Qt-free.

A drag & drop never chooses: its session is a fresh ``{project}/sessions/{timestamp}/``
(``gui/project.new_session_dir``). The wizard writes a notebook whose ``session_dir``
the user edits, and proposes the media folder of the best-ranked kind.
"""

from __future__ import annotations

#: Which kind of folder makes the best session folder, most preferred first.
#: Data files before media: the folder with the features is the natural home.
KIND_ORDER: tuple[str, ...] = ("nc", "nwb", "npz", "npy", "video", "pose", "audio", "ephys", "image", "labels")
