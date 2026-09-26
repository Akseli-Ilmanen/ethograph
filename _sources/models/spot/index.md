(target-spot)=
# Precise event spotting (PES) from pixels

| You want to… | Read |
|---|---|
| Train a first model this afternoon, with every choice made for you | {doc}`quickstart` — three sessions, one camera, plain E2E-Spot on pixels only |
| Understand every stage: the three ways to spot an event, the config, durations, confidence, MSAGSM | {doc}`guide` |
| Add the pose features you already have as a second input beside the pixels | {doc}`multimodal` |
| Look up a key you already know you need | {doc}`config` |

Whichever route you take, the predictions are a labels TSV the GUI opens with
**File ▸ Import labels…**, and {doc}`curation <../curation/index>` is where
you review them.

```{toctree}
:maxdepth: 1

quickstart
Full guide <guide>
Pixels + pose <multimodal>
Config reference <config>
```
