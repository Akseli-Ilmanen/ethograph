.. _target-nwb-alignment-api:

NWB alignment
=============

.. currentmodule:: ethograph.io.nwb_alignment

The NWB alignment file (``<project>/.ethograph/alignment.nwb``) stores trial
timing, media file paths, and stream offsets. It is the single source of
truth for "what file corresponds to what trial, and when did it start".

See :ref:`Pairing and alignment <target-nwb-alignment>` for the user-facing walkthrough.
At runtime the same interface is available via ``dt.nwb_alignment`` on a
loaded TrialTree.

----

Pairing media files
-------------------

.. currentmodule:: ethograph.io.pairing

.. autoclass:: SourceSpec

.. autofunction:: discover_media

.. autofunction:: pair_media

----

Creating alignment files
------------------------

.. currentmodule:: ethograph.io.nwb_alignment

.. autofunction:: align_media_per_trial

.. autofunction:: align_media_from_streams

.. autofunction:: make_nwb_alignment

.. autofunction:: discover_nwb

.. autofunction:: sync_acquisition_for_streams

.. autofunction:: edit_nwb

----

Reading alignment metadata
--------------------------

.. autoclass:: NWBAlignment
   :members: cameras, mics, start_time, stop_time, get_media, devices,
             resolve_media_path, stream_offset_for_trial, get_stream_rate,
             electrical_series, trials_df, has_real_timing, print_session
   :no-inherited-members:

.. autoclass:: TableAlignment
   :no-inherited-members:

.. autoclass:: EmpytAlignment
   :no-inherited-members:

----

NWB import helpers
------------------

.. currentmodule:: ethograph.io.nwb_import

.. autofunction:: read_trials_table

.. autofunction:: probe_electrical_series

.. autofunction:: probe_label_sources
