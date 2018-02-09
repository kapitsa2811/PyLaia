#!/bin/bash
set -e;
export LC_NUMERIC=C;
export PATH="$(pwd)/../../:$PATH";

overwrite=false;
batch_size=8;

# Directory where the run.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)" != "$SDIR" ] && \
  echo "Please, run this script from the experiment top directory!" && \
  exit 1;

mkdir -p data/;
# download dataset
[ -f data/Spanish_Number_DB.tgz ] || \
  wget --no-check-certificate -P data/ http://www.prhlt.upv.es/corpora/spanish-numbers/Spanish_Number_DB.tgz;
# extract it
[ -d data/Spanish_Number_DB ] || \
  tar -xzf data/Spanish_Number_DB.tgz -C data/;

./steps/prepare.sh --overwrite "$overwrite";
num_symbols=$[$(wc -l data/lang/char/symbs.txt | cut -d\  -f1) - 1];

[ -f model.t7 -a "$overwrite" = false ] || {
  echo "Create and train model";
  # pylaia-train-ctc \
  #   data/lang/char/symbs.txt \
  #   data/train.lst \
  #   data/test.lst;
}

