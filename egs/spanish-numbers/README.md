# Step-by-step Training Guide Using Spanish Numbers Dataset

Spanish Numbers Dataset is a small dataset of 485 images containing handwritten sentences of Spanish numbers (298 for training and 187 for testing).

![Example](example.png "Example")

## Requirements

- Laia
- ImageMagick's convert
- Optionally: Kaldi's compute-wer

## Training

To train a new Laia model for the Spanish Numbers dataset just follow these steps. We will use the test partition as validation, given that this dataset does not provide validation partition,

- Download the Spanish Numbers dataset:
```bash
mkdir -p data/;
wget -P data/ https://www.prhlt.upv.es/corpora/spanish-numbers/Spanish_Number_DB.tgz;
tar -xvzf data/Spanish_Number_DB.tgz -C data/;
```

- Execute `steps/prepare.sh`. This script assumes that Spanish Numbers dataset is inside `data` folder. This script does the following:
  - Transforms the images from pbm to png.
  - Scales them to 128px height.
  - Creates the auxiliary files necessary for training.

- Execute the `TBD` script to create an "empty" laia model using:
```bash
TBD
```

- Use the `TBD` script to train the model:
```bash
TBD
```

## Decoding

TBD

## TL;DR

Execute `run.sh`.
