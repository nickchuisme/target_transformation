#!/bin/bash

python3 model_selection.py --threshold_step 0.03 --lags 1 2 3 4 5 --data_num 20 --data_length 300 --worker 30

echo "[[Finished!]]"