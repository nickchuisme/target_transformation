#!/bin/bash

python3 model_selection.py --thresholds 0.003 0.005 0.008 0.013 0.02 0.04 0.07 0.12 0.2 0.25 --lags 1 2 3 4 5 --data_num 32 --data_length 300 --worker 30

echo "[[Finished!]]"