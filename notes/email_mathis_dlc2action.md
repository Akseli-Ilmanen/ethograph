**Subject:** Thank you for DeepLabCut and DLC2Action, and a licensing question about ethograph

Dear Prof. Mathis,

I'm writing from [Felix Moll's lab](https://felixmoll.com/), where we study behaviour in crows. Thank you for DeepLabCut and DLC2Action. We now use both in the lab.

We have been building [ethograph](https://github.com/Akseli-Ilmanen/ethograph), a GUI for visualising, labelling and segmenting behaviour. Last year we started doing action segmentation with architectures like ASFormer, which worked very well. We only came across DLC2Action in October 2025. It gave similar results to what we had, but trained much faster and offered more functionality.

Initially we wanted to keep ethograph as a labelling tool only and integrate with DLC2Action. We found that difficult, because it wasn't intuitive to us how the data should be organised and the configs selected. We work with the [movement](https://movement.neuroinformatics.dev/) library (compatible with DeepLabCut and other pose estimation tools) and wanted one streamlined path: feature engineering in movement, labelling in ethograph, then training segmentation models.

So we adopted many of the architectures from DLC2Action. We vendored the models, the losses and their default configs, unmodified, with the AGPL licence and the upstream attributions kept. We did not copy the toolbox layer. [THIRD_PARTY_NOTICES.md](https://github.com/Akseli-Ilmanen/ethograph/blob/main/THIRD_PARTY_NOTICES.md) and the [NOTICE.md](https://github.com/Akseli-Ilmanen/ethograph/blob/main/ethograph/segment/dlc2action/NOTICE.md) next to the code specify what we used, adapted and left out.

I wanted to make sure I did everything right with the licensing. If anything is missing, I would be glad to fix it. And of course, when promoting ethograph we want to encourage users to cite DLC2Action whenever they use these models. If you have a preferred citation, please let me know.

Thanks again for making this work openly available.

Best wishes,
Akseli Ilmanen
Moll Lab
