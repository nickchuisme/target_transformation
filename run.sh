#!/bin/bash

python3 model_selection.py --threshold 0.01 0.02 0.03 0.05 0.07 0.1 0.13 0.16 0.2 0.25 --lags 1 2 3 4 5 --data_num 32 --data_length 300 --worker 30

echo "[[Finished!]]"