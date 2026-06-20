This folder contains selected figures from the electricity load forecasting experiments. The figures are included for research documentation and portfolio demonstration.

# Rolling-Origin Forecast Example — N-HiTS on ECL

<img width="3567" height="1168" alt="NHiTS_ECL_H24_LF5_ROLLING_FULL_313" src="https://github.com/user-attachments/assets/b565be09-f932-4838-bb6b-78dfbb4532dd" />


This figure shows an example of rolling-origin electricity load forecasting using the N-HiTS model on the ECL dataset. It compares actual electricity load values with model predictions for a 24-step forecast horizon.

The figure illustrates how the forecasting model follows short-term demand patterns across consecutive forecast windows. It is intended as a visual example of the model’s behavior under rolling-origin evaluation.

# N-HiTS on ECL — Error Overview

<img width="5366" height="2363" alt="NHiTS_ECL_H24_LF5_ROLLING_FULL_error_overview" src="https://github.com/user-attachments/assets/c5e495c5-9826-4e69-a700-3132d03db669" />


This figure summarizes the forecasting error distribution for the N-HiTS model on the ECL dataset. It includes the signed error histogram, per-series normalized MAE distribution, the top series with the largest maximum absolute errors, and a Pareto/Lorenz-style view showing how much of the total error is concentrated in the worst-performing series.


# N-HiTS on ECL — Selected High-Error Series

<img width="4719" height="2329" alt="NHiTS_ECL_H24_LF5_ROLLING_FULL_worst_sum_abs_err" src="https://github.com/user-attachments/assets/1682f386-094c-4071-8eeb-f8a984d9be7f" />

This figure shows selected ECL time series with high accumulated forecasting errors. Each subplot compares the true normalized load values with the N-HiTS forecasts. The figure helps identify where the model follows the general daily pattern well and where it struggles with abrupt changes, irregular behavior, or unusual demand drops.
