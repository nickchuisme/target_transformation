## Target transformation

1. Run the experiment by applying transformation on the target series to anticipate improving the forecasting performance.
2. Analyse the experiment results from a saved JSON file.

### Execution:
```bash
python3 model_selection.py
```

### Workflow
![plot](http://processon.com/chart_image/62bd97fde0b34d075b187c0b.png)


1. <code>model_selection.py</code> validates the model with the combinations of lags, thresholds and model parameters. Then, select each best-validated model to fit the training data and calculate the model's performance by <code>sMAPE</code> measure.

2. Generated data from each time series will store in the JSON file. Each data includes time-series information, untransformed and transformed predictions of each model and the forecasting performance (error measures) of each model.

3. <code>test.ipynb</code> gives a simple visualisation of comparing the prediction from transformed and untransformed time series.

### Files

- <code>settings.py</code>: this file saves the available models and theirs parameters. When validating a model, the main function calls the model parameters from this file and creates possible combinations.

- <code>util.py</code>: this file gives some toolkits.
  1. <code>Performance_metrics</code> is able to calculate the performance with a set of error measures or a specific measure.
  2. <code>Records</code> saves each time-series information into a JSON file

- <code>transformation.py</code> provides <code>dwt()</code> (Discrete Wavelet Transformation) to transform time series.
