# Modeling probabilistic reduction across domains with Naive Discriminative Learning

Authors: [Anna Sophia Stein](anna.stein.com), [Kevin Tang](https://www.kevintang.org/)

This repository is the official implementation of Modeling probabilistic reduction across domains with Naive Discriminative Learning.

**Abstract**:
 The predictability of a word modulates its acoustic duration. Such probabilistic effects can compete across linguistic domains (segments, syllables and adjacent-word contexts e.g., frequent words with infrequent syllables) and across local and aggregate contexts (e.g., a generally unpredictable word in a predictable context). This study aims to tease apart competing effects using Naive Discriminative Learning, which incorporates cue competition. The model was trained on English conversational speech from the Buckeye Corpus, using words as outcomes and segments, syllables, and adjacent words as cues. The connections between cues and outcomes were used to predict acoustic word duration. Results show that a word's duration is more strongly predicted by its syllables than its segments, and a word's predictability aggregated over all contexts is a stronger predictor than its specific local contexts. Our study presents a unified approach to modeling competition in probabilistic reduction.

## Requirements

To install the requirements:

```setup
pip install -r requirements.txt
```

## Workflow

Below is a short version of what you ened to do in order to recreate my workflow. All of these files are in the `/code`folder. A longer, more thourough version of the worklofw can be found in `docs/workflow.txt`.

The code for this thesis, especially the Python code, is by no means the most efficient way to do things. However, it represents my coding journey and what I was able to do at the time. In the future I hope to add a more concise and faster version of the current code.

1. Create a **word list** for every speaker in the Buckeye corpus with `buckeye_text.py`.
2. Create table with **data** from the Buckeye corpus **for the regression analysis** with `regression_data.py`.
3. Create individual **speaker event files** with `eventfilesV1.py`.
4. Correct the previous event files with `correctingV1eventfiles.py`.
5. **Train** the NDL model with the input from 4. with `trainNDL.py`.
6. Compute **NDL predictors** for the regression analysis with `prior_activation.py`.
7. Compute **speech rate** per utterance with `speech_rate.py`.
8. Replicate the **statistical analysis** with `regression_analysis.Rmd`

# License

All source code is made available under a BSD 3-clause license. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE.md` for the full license text.

The manuscript text is not open source. The author reserves the rights to the
thesis content.
