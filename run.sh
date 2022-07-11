#!/bin/bash

python3 model_selection.py --thresholds 0.003 0.005 0.008 0.013 0.02 --lags 1 2 3 --gap 0 --data_num 96 --data_length 1000 --worker 30

echo "[[Finished!]]"