(target-segment)=
# Action segmentation

| You want to… | Read |
|---|---|
| Train a first model this afternoon, with every choice made for you | {doc}`quickstart` — three sessions, three kinematic features, one architecture, four lines |
| Understand every stage: the search, the cross-validation, samples, the materialised dataset, the architectures | {doc}`guide` |
| Look up a key you already know you need | {doc}`config` |
| Give the model what the animal *looks like it is doing*, from a pretrained video network | {doc}`video_features` |
| Fine-tune a video foundation model on the pixels alone, or feed its embeddings to this pipeline | {doc}`feral` |

Whichever route you take, the predictions are a labels TSV the GUI opens with
**File ▸ Import labels…**, and {doc}`curation <../curation/index>` is where
you review them.

```{toctree}
:maxdepth: 1

quickstart
Full guide <guide>
Config reference <config>
video_features
feral
```
