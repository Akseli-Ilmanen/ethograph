.. _target-examples-gallery:

Examples
========

Jupyter notebooks showing how real-world datasets were converted into the
:class:`~ethograph.io.trialtree.TrialTree` format.

Every example can be run locally: open a notebook and use **Download notebook**
in the right-hand sidebar, or browse the whole folder on GitHub.

.. button-link:: https://github.com/Akseli-Ilmanen/ethograph/tree/main/examples
   :color: primary
   :outline:

   :octicon:`mark-github` Examples on GitHub

Have a workflow or dataset others could reuse? See
:ref:`Add your workflow/dataset example! <target-add-your-example>` in the
contributing guide.

.. grid:: 1 2 2 3
   :gutter: 4
   :class-container: examples-gallery

   .. grid-item-card:: Tool-using crows
      :img-top: ../_static/media/moll1.png
      :link: create_dataset_Moll25
      :link-type: doc
      :class-card: example-card

      Moll et al., 2025 — two-camera video, DeepLabCut 2D/3D pose, video
      features and kinematic features of a Carrion crow doing a tool-use task..

   .. grid-item-card:: Sound production in crickets
      :img-top: ../_static/media/cricket0.png
      :link: create_dataset_cricket
      :link-type: doc
      :class-card: example-card

      Stridulating *Pholidoptera littoralis* — video, audio and pose in one
      dataset, so movement and sound features can be explored side by side.

   .. grid-item-card:: Social rats
      :img-top: ../_static/media/pair24.png
      :link: convert_dataset_pair24
      :link-type: doc
      :class-card: example-card

      Marshall et al., 2021 (PAIR-R24M) — multi-animal 3D pose of dyadic
      interactions between laboratory rats.

   .. grid-item-card:: Flying & singing zebra finches
      :img-top: ../_static/media/birdpark0.png
      :link: convert_dataset_birdpark
      :link-type: doc
      :class-card: example-card

      Rüttimann et al., 2025 (BirdPark) — synchronized video, microphone
      arrays and backpack-mounted accelerometers for zebra finch groups.

   .. grid-item-card:: Tool-using mice
      :img-top: ../_static/media/lockbox2.gif
      :link: convert_datset_lockbox
      :link-type: doc
      :class-card: example-card

      Reiske et al., 2025 (Mouse Lockbox) — mice solving mechanical puzzles,
      filmed from top, front and side with pose for each view.

.. toctree::
   :maxdepth: 1
   :hidden:

   create_dataset_Moll25
   create_dataset_cricket
   convert_dataset_pair24
   convert_dataset_birdpark
   convert_datset_lockbox
