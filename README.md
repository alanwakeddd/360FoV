# FoV
FoV prediction for 360 videos.

Shanghai Tech dataset 
http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_Gaze_Prediction_in_CVPR_2018_paper.pdf

Tsinghua dataset
https://wuchlei-thu.github.io/
1. use dataModder_v2.py to clean data and create new csv files.
2. use dataIO2.py to create dictionary pickle files from csv files.
3. use function get_data() in utility.py to convert dictionary pickles into numpy arrays. Now the get_data() supports overlapping(stride=1) chuncks. 

classify users and split training and testing.


## Model variants

| model                                 | input     | output    | sample&refeed? | loss |auc (a=1)| auc (a=1.25) |
| ------------------------------------- | --------- | --------- | -------------- | ---- | ---- | ---- |
| fclstm seq2seq                        | mean,var | mean,var | no             | mse  |0.5765|0.6921|
| fclstm seq2seq                        | raw and mean,var | mean,var | no            | mse  |0.5876|0.7021|
| fclstm seq2seq+MLP model others gt | raw and mean,var | mean,var | no             | mse  |0.6276|0.7486|
| fclstm seq2seq+bi-LSTM model others gt | raw and mean,var | mean,var | no             | mse  |0.6342|0.7539|
| fclstm seq2seq+bi-LSTM model others gt+mlp mixing | raw and mean,var | mean,var | no             | mse  |0.6226|0.7449|
| fclstm seq2seq+LSTM model others gt | raw and mean,var | mean,var | no             | mse  |0.6299|0.7503|
| fclstm seq2seq+MLP mixing (others gt) | mean,var | mean,var | no             | mse  |0.6281|0.7479|
| * fclstm seq2seq+MLP mixing (others gt) | raw and mean,var | mean,var | no             | mse  |0.6378|0.7566|
| fclstm seq2seq+ 3 layer MLP mixing (others gt) | raw and mean,var | mean,var | no             | mse  |0.6343|0.7521|
| convlstm seq2seq                      | raw       | raw       | no             | mse  |  0.2272|0.3254|
| convlstm seq2seq pred mean/var               | mean,var | mean,var      | no            | mse  |0.5686|0.6901|
| convlstm seq2seq pred mean/var               | raw, raw | mean,var      | yes             | mse  |0.5795|0.6992|
| convlstm seq2seq+convlstm others      | mean,var ,raw and mean var | all mean, var | no             | mse  |0.4890|0.6172|
| convlstm seq2seq+convlstm others      | raw, raw, raw | all mean,var | yes            | mse  |0.5329|0.6539|
| single LSTM | raw, raw| mean, var | yes             | mse  |-|0.6790 (a=1.25)|
| Averaging other users | raw, raw| mean, var | no             | mse  |0.4552|-|




##### For Tsinghua Dataset:
| model                                 | input     | output    | sample&refeed? | loss |auc|
| ------------------------------------- | --------- | --------- | -------------- | ---- | ---- |
| fclstm seq2seq+ 3 layer MLP mixing (others gt) | raw and mean,var | mean,var | no             | mse  |0.8514|
| fclstm seq2seq+bi-LSTM model others gt | raw and mean,var | mean,var | no             | mse  |0.8372|
| fclstm seq2seq                        | raw and mean,var | mean,var | no            | mse  |0.8383|
| convlstm seq2seq pred mean/var               | raw, raw | mean,var      | yes             | mse  |0.8308|



##### Plus visual saliency inputs (Shanghaitech Dataset):
| model                                 | input     | output    | sample&refeed? | loss |auc|
| ------------------------------------- | --------- | --------- | -------------- | ---- | ---- |
| MLP mixing fc_mlpmixing_saliency_residual_sep11 | raw and mean,var | mean,var | no             | mse  |0.7277|
| MLP mixing fc_mlpmixing_shanghai_plussaliency_stride10_bs8_sep11| raw and mean,var | mean,var | no             | mse  |0.7149|




## Observations and Insights

#### Baselines
We have compared with two baselines: the persistence model and the truncated linear extrapolation. The persistence model is simply repeating the last known FoV locations. The truncated linear model requires that we first detect the last monotonic line segment of the historic trajectory and then do linear extrapolation. We used the python function InterpolatedUnivariateSpline from scipy.interpolate package to do order 1 linear extrapolation, we found the performance of this function is very similar with truncated linear. 

#### At the beginning of the prediction time, i.e. within 0-2 seconds (60 frames), 
The truncated linear baseline model has the best performance, and the persistence baseline model is the second best. This observation indicates that users' viewing trajectories are very linear in a very short time span. However, both baseline models drop very rapidly as the prediction time increases. The truncated linear model drops to around 50% hitrate in the 10th second prediction while LSTM based models have around 70%. 


#### Evaluation metircs
Mean square loss 
hit rate 


