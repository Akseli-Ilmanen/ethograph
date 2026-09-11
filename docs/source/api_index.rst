.. _target-api:

API reference
=============

Organized by module group. Each page has full signatures with type hints,
examples, and cross-links to the narrative docs.

.. toctree::
   :maxdepth: 1
   :hidden:

   api/trialtree
   api/dataset
   api/pynapple_io
   api/nwb_alignment
   api/labels
   api/changepoints
   api/xr_utils

.. rubric:: At a glance

:doc:`TrialTree <api/trialtree>`
    The :class:`~ethograph.io.trialtree.TrialTree` data structure,
    :func:`~ethograph.open`, :func:`~ethograph.from_datasets`, trial
    access, iteration, filtering, and saving.

:doc:`Dataset <api/dataset>`
    Xarray-side dataset builders:
    :func:`~ethograph.io.dataset.downsample_trialtree`,
    :func:`~ethograph.io.dataset.add_changepoints_to_ds`,
    :func:`~ethograph.io.dataset.add_angle_rgb_to_ds`.

:doc:`Pynapple IO <api/pynapple_io>`
    Pynapple loaders and augmenters:
    :func:`~ethograph.io.pynapple.load_nap_data`,
    :func:`~ethograph.io.pynapple.detect_trials`,
    :func:`~ethograph.io.pynapple.add_changepoints_to_nap`,
    :func:`~ethograph.io.pynapple.add_angle_rgb_to_nap`, plus NWB-import
    probes.

:doc:`NWB alignment <api/nwb_alignment>`
    :class:`~ethograph.io.nwb_alignment.NWBAlignment`,
    :func:`~ethograph.io.pairing.discover_media`,
    :func:`~ethograph.io.pairing.pair_media`.

:doc:`Labels <api/labels>`
    Interval operations, TSV storage, dense ↔ interval conversion,
    predictions, Crowsetta/pynapple converters, export, and plotting.

:doc:`Changepoints <api/changepoints>`
    Detection (:func:`~ethograph.features.changepoints.find_peaks_binary`,
    :func:`~ethograph.features.changepoints.find_troughs_binary`,
    :func:`~ethograph.features.changepoints.find_nearest_turning_points_binary`),
    merging/time extraction, label correction, and the binary / smooth
    Laplacian / segment-ID
    :func:`~ethograph.features.changepoints.more_changepoint_features`
    used by downstream segmentation models.

:doc:`xr_utils <api/xr_utils>`
    :func:`~ethograph.utils.xr_utils.sel_valid` and
    :func:`~ethograph.utils.xr_utils.get_time_coord`.
