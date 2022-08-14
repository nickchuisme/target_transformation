## Target transformation

1. Run the experiment by applying transformation on the target series to anticipate improving the forecasting performance.
2. Analyse the experiment results from a saved JSON file.

### Execution:
<!-- for testing:
- lags: 1, 12
- wavelets: haar, db5, db6, db19, db20, coif4, coif5
```bash
./run_test.sh
``` -->
execute entire experiment:
- lags: 1, 12
- wavelets: haar, db5, db6, db19, db20, coif4, coif5
```bash
./run.sh
```

### Workflow
<img src="https://imgur.com/NiWCgUn.png" width="900" height="400">


1. <code>model_selection.py</code> validates the model with the combinations of lags, thresholds and model parameters. Then, select each best-validated model to fit the training data and calculate the model's performance by <code>sMAPE</code> measure.

2. Generated data from each time series will store in the JSON file. Each data includes time-series information, untransformed and transformed predictions of each model and the forecasting performance (error measures) of each model.

3. <code>test.ipynb</code> gives a simple visualisation of comparing the prediction from transformed and untransformed time series.

### Files

- <code>settings.py</code>: this file saves the available models and theirs parameters. When validating a model, the main function calls the model parameters from this file and creates possible combinations.

- <code>utils.py</code>: this file gives some toolkits.
  1. <code>Performance_metrics</code> is able to calculate the performance with a set of error measures or a specific measure.
  2. <code>Records</code> saves each time-series information into a JSON file

- <code>transformation.py</code> provides <code>dwt()</code> (Discrete Wavelet Transformation) to transform time series.
