#!/bin/bash

python model_selection.py --threshold_step 0.03 --lags 1 2 3 4 5 --data_num 20 --data_length 80 --worker 30

echo "[[Finished!]]"