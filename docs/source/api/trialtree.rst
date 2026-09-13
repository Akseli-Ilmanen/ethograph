.. _target-trialtree-api:
.. _target-trialtree:

TrialTree
=========

.. currentmodule:: ethograph.io.trialtree

:class:`TrialTree` is a wrapper around :class:`xarray.DataTree` that stores
one :class:`xarray.Dataset` per trial. Build one from a list of datasets,
then access each trial by ID or index:

.. code-block:: python

   import numpy as np, xarray as xr, ethograph as eto

   # Build: one xr.Dataset per trial
   datasets = []
   for i in range(1, 4):
       ds = xr.Dataset({"speed": xr.DataArray(np.random.rand(300), dims=["time"],
                                               coords={"time": np.arange(300) / 30.0})})
       ds.attrs["trial"] = i
       ds.attrs["fps"] = 30.0
       datasets.append(ds)

   dt = eto.from_datasets(datasets)

   # Access by trial ID (label-based, like xr.Dataset.sel)
   ds = dt.trial(2)
   ds.attrs["trial"]   # 2
   ds["speed"]          # the speed DataArray for trial 2

   # Access by integer index (0-based, like xr.Dataset.isel)
   ds = dt.itrial(0)
   ds.attrs["trial"]   # 1

   # List all trial IDs
   dt.trials            # [1, 2, 3]

   # Save / load
   dt.save("session.nc")
   dt = eto.open("session.nc")

For the :class:`xarray.Dataset` structure expected inside each trial, see
:doc:`../getting_started/your_data/index`.

.. autoclass:: TrialTree
   :no-members:
   :no-inherited-members:

----

Creating
--------

.. automethod:: TrialTree.open

.. automethod:: TrialTree.from_datasets

.. automethod:: TrialTree.from_continuous

For a single long recording with trial epochs, :meth:`~TrialTree.from_continuous`
slices on demand instead of copying data:

.. code-block:: python

   import pandas as pd

   epochs = pd.DataFrame({
       "trial": [1, 2, 3],
       "start_time": [0.0, 60.0, 120.0],
       "stop_time": [60.0, 120.0, 180.0],
   })
   dt = eto.from_continuous(ds, epochs)
   dt.trial(2)  # returns 60–120 s slice, time shifted to 0

.. automethod:: TrialTree.from_datatree

----

Accessing trials
----------------

.. automethod:: TrialTree.trial

.. automethod:: TrialTree.itrial

.. autoproperty:: TrialTree.trials

.. automethod:: TrialTree.get_all_trials

.. automethod:: TrialTree.get_common_attrs

.. automethod:: TrialTree.get_trial_metadata

----

Iterating over trials
---------------------

.. code-block:: python

   for trial_id, ds in dt.trial_items():
       print(f"Trial {trial_id}: {len(ds.time)} timepoints")

   # Apply a function to every trial, returning a new TrialTree
   dt_smoothed = dt.map_trials(lambda ds: smooth(ds))

.. automethod:: TrialTree.trial_items

.. automethod:: TrialTree.map_trials

----

Modifying trials
----------------

**In-place mutations** work directly through :meth:`~TrialTree.trial` because
the returned dataset shares its underlying data with the tree:

.. code-block:: python

   dt.trial(1).attrs["human_verified"] = True
   dt.trial(1)["speed"].values[:10] = 0.0

**Structural changes** (adding/removing variables) require
:meth:`~TrialTree.update_trial`:

.. code-block:: python

   dt.update_trial(1, lambda ds: ds.assign(
       smoothed_speed=ds["speed"].rolling(time=5).mean()
   ))

.. automethod:: TrialTree.update_trial

----

Filtering
---------

.. code-block:: python

   dt_tone_a = dt.filter_by_attr("stimulus", "tone_A")

.. automethod:: TrialTree.filter_by_attr

----

Saving
------

.. code-block:: python

   dt.save("path/to/session.nc")
   dt.save()  # overwrite the file it was loaded from

When saving to a new directory, the NWB alignment file is automatically
copied alongside the ``.nc``.

.. automethod:: TrialTree.save

----

See also
--------

- :doc:`dataset` — dataset builders (:func:`~ethograph.io.dataset.downsample_trialtree`,
  :func:`~ethograph.io.dataset.add_changepoints_to_ds`,
  :func:`~ethograph.io.dataset.add_angle_rgb_to_ds`).
- :doc:`pynapple_io` — pynapple loading (:func:`~ethograph.io.pynapple.load_nap_data`,
  :func:`~ethograph.io.pynapple.add_changepoints_to_nap`) and NWB-import probes.
- :doc:`nwb_alignment` — trial timing, media paths, stream offsets via
  ``dt.nwb_alignment``.
- :doc:`labels` — TSV label sidecar format and helpers.
- :doc:`changepoints` — detection, merging, time extraction, and label
  correction.
