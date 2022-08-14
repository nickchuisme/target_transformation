## Target transformation
The aim is to investigate the forecasting performance of target transformation by comparing how models perform when trained on the untransformed and transformed time series.

The discrete wavelet transform (DWT) is applied to the time series data for each candidate model, such as Elastic Net (EN), Support Vector Regressor (SVR), K-Nearest Neighbours (KNN), Random Forest (RF), Multi-layer Perceptron (MLP), and ETS.

### Execution
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


1. <code>model_selection.py</code> implements the model selection with the combinations of lags, wavelets and model parameters. Then, select each best-validated model to fit the training data and calculate the model's performance by <code>sMAPE</code> measure. In this script, preprocessing method (detrending and deseasonalisation) are optional functions to remove trend and seasonal components.

2. Generated data from each time series will store in the JSON file. Each data includes time series information, untransformed and transformed predictions of each model and the forecasting performance (error measures) of each model.

### Other Files

- <code>settings.py</code>: this file saves the available models and theirs parameters. When validating a model, the main function calls the model parameters from this file and creates candidate combinations.

- <code>utils.py</code>: this file gives some toolkits.
  1. <code>Performance_metrics</code> is able to calculate the performance with a set of error measures or a specific measure.
  2. <code>Records</code> saves each time series information into a JSON file
  3. <code>DeTrendSeason</code> provides detrending and deseasonalisation methods which are suggested by M-competition.

- <code>transformation.py</code> provides:
  1. <code>dwt</code> which is used to transform a time series with a unerversal threshold and a specific wavelet.
  2. <code>dwt_feature</code> can not only gives a transformed time series but also generate the decomposed subseries.
