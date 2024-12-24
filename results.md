xLSTM sMAPE on 3 series: 17.6,
time to train: 38.69 seconds

    "time_to_train": 27.95
for 5 series with embedding dim = 64 and 1 epoch


286.54 seconds for 5 series and 10 epochs, dim=32

-> 1 epoch now
23.83 sec for dim=32 and 5 series

dim=64, 5 series, 28.11 seconds
-> dim has an effect

let's keep dim at 64 but try to reduce xlstm stack complexity
reduce blocks to 6
-> Training completed in 22.47 seconds

let's also reduce dim to 32

Training completed in 22.32 seconds
-> no big effect

back to 64:
-> 21.48 sec

reduce to 4 blocks
-> 15.40 seconds

reducing num heads to 4
13.82 seconds

Use a smaller conv1d_kernel_size (e.g., from 6 to 4) to reduce the receptive field slightly.
Set qkv_proj_blocksize to a smaller value (e.g., 4) for finer granularity.
Simplify Feedforward Layers:
Decrease proj_factor in FeedForwardConfig (e.g., from 5.0 to 3.0). A lower projection factor results in smaller intermediate representations, reducing computational cost.

conv1d_kernel_size from 6 to 4:
14.51 seconds

qkv_proj_blocksize 8 to 4
15.50 seconds

return conv1d to 6, blocksize to 8, reduce proj factor to 3:
Training completed in 14.56 seconds

bias_init to standard
Training completed in 14.89 seconds

change dropout to 0.1
Training completed in 14.67 seconds
-> no effect

remove slstm
Training completed in 2.35 seconds

remove mlstm
Training completed in 28.68 seconds

-> slstm slows down significantly


mlstm only with latest config trains model
in 212.38 seconds (for 1 epoch)
"SMAPE": 19.74,


okay let's reduce num_heads in slstm to 2, back to 5 series:
latest result is around 14 seconds
Training completed in 14.54 seconds


reduce to only 1 slstm block
Training completed in 8.68 seconds
sMAPE 17.57

-> each slstm adds about 6 seconds
let's add the second back
Training completed in 13.96 seconds
sMAPE 16.24

### slstm is not parallelizable

unfortunately according to the paper only using mlstm is subpar.
mlstm removes memory mixing.

adding the 7:1 mlstm:slstm ratio
Training completed in 9.94 seconds
sMAPE 12.26
so the additional 3 blocks added around 3 seconds

it might also be not worth using slstm at all 
according to some results of the paper

for bigger models it performs better without the slstm


7 blocks only mlstm:
Training completed in 4.46 seconds
sMAPE 16.78

sLSTM with increased scale increases context and decreases performance
mLSTM the opposite

okay let's train the entire model now with 7 blocks

Training completed in 372.39 seconds
sMAPE 31.03
-> much worse results with 7 blocks

4 blocks mlstm only 5 series
Training completed in 2.33 seconds
sMAPE 16.82

7 blocks mlstm only 5 series
Training completed in 3.68 seconds
sMAPE 24.01
Training completed in 3.70 seconds
sMAPE 18.7

-> doesnt seem the additional blocks help

4 blocks + 1 slstm
Training completed in 8.38 seconds
sMAPE 16.95
Training completed in 8.37 seconds
sMAPE 19.87
Training completed in 8.91 seconds
sMAPE 15.38

7 + 1 
Training completed in 10.08 seconds
sMAPE 21.31
Training completed in 9.90 seconds
sMAPE 19.31

-> it's a bit random on the results

either 4 blocks only mlstm or 4:1

4:1 on all series
Training completed in 803.73 seconds
sMAPE 20.8

-> it doesn't seem the slstm is helping at all
let's try 4 mlstm blocks with more complexity num heads 8

5 series:
Training completed in 2.88 seconds
sMAPE 16.09

all 414 series:
Training completed in 292.31 seconds
sMAPE 20.42

-> head doesn't affect too much

let's try 10 epochs
Training completed in 2901.36 seconds
sMAPE 16.33

direct:
SIMPLERNN Total Weighted SMAPE: 20.18
COMPLEXLSTM Total Weighted SMAPE: 20.51
TIMESERIESTRANSFORMER Total Weighted SMAPE: 20.57
XLSTM Total Weighted SMAPE: 20.82

recursive:
SIMPLERNN Total Weighted SMAPE: 18.33
COMPLEXLSTM Total Weighted SMAPE: 20.44
TIMESERIESTRANSFORMER Total Weighted SMAPE: 19.51
XLSTM Total Weighted SMAPE: 21.65