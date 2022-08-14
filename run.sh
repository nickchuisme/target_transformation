#!/bin/bash

python3 model_selection.py --lags 1 12 --gap 0 --data_num 96 --data_length 100 --worker 30

echo "[[Finished!]]"