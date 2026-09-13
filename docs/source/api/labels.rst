.. _target-labels-api:

Labels
======

Labels are stored as interval DataFrames (``onset_s``, ``offset_s``,
``labels``, ``individual``, ``trial``, ...) in a TSV file alongside each
``.nc`` dataset. See :doc:`../advanced/labels/index` for the user-facing
workflow and the storage-format reference.

----

Interval operations
-------------------

.. currentmodule:: ethograph.labels.intervals

.. autofunction:: add_interval

.. autofunction:: delete_interval

.. autofunction:: find_interval_at

.. autofunction:: get_interval_bounds

.. autofunction:: empty_intervals

.. autofunction:: purge_short_intervals

.. autofunction:: stitch_intervals

.. autofunction:: snap_boundaries

.. autofunction:: load_label_mapping

.. autofunction:: save_label_mapping

.. autofunction:: load_mapping

----

Dense ↔ interval conversion (ML)
--------------------------------

.. currentmodule:: ethograph.labels.ml

.. autofunction:: dense_to_intervals

.. autofunction:: intervals_to_dense

.. autofunction:: get_labels_start_end_indices

----

TSV storage
-----------

.. currentmodule:: ethograph.labels.tsv_store

.. autofunction:: labels_tsv_path

.. autofunction:: load_labels_tsv

.. autofunction:: save_labels_tsv

.. autofunction:: validate_labels_tsv

.. autofunction:: init_empty_labels

.. autofunction:: get_trial_from_tsv

.. autofunction:: set_trial_in_tsv

.. autofunction:: get_trial_meta

.. autofunction:: set_trial_meta_attr

.. autofunction:: migrate_label_dt_to_tsv

----

Predictions
-----------

.. currentmodule:: ethograph.labels.predictions

.. autofunction:: prediction_to_labels_and_confidence

.. autoclass:: PredictionsStore
   :members:
   :no-inherited-members:

----

Crowsetta / pynapple converters
-------------------------------

.. currentmodule:: ethograph.labels.crowsetta_format

.. autoclass:: EthographSeq
   :members:
   :no-inherited-members:

.. currentmodule:: ethograph.labels.converters

.. autoclass:: LabelConverter
   :no-inherited-members:

.. autoclass:: CrowsettaLabelConverter
   :no-inherited-members:

.. autoclass:: NWBLabelConverter
   :no-inherited-members:

.. autoclass:: PynappleLabelConverter
   :no-inherited-members:

.. autofunction:: crowsetta_to_intervals

.. autofunction:: extract_crowsetta_labels

.. autofunction:: extend_mapping

.. autofunction:: build_mapping_from_labels

.. autofunction:: write_mapping_file

----

Export helpers
--------------

.. currentmodule:: ethograph.labels.export

.. autofunction:: enrich_labels_df

.. autofunction:: correct_offsets

.. autofunction:: correct_offsets_trial

.. autofunction:: trees_to_df

----

Plotting
--------

.. currentmodule:: ethograph.labels.plots

.. autofunction:: draw_label_rectangle

.. autofunction:: plot_label_segments

.. autofunction:: plot_label_segments_multirow
