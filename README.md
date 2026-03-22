# Deep learning experiments - ETSIINF
# FFNNs (1st report)
## Léo
### EXPERIMENT 1 - file: ffnn_exp1_leo.ipynb
#### changes : default, without changing the template
#### Results : 
Mean Accuracy: 37.013%
Mean Recall: 24.446%
Mean Precision: 30.915%

### EXPERIMENT 2 - file: ffnn_exp2_leo.ipynb
#### changes : adding a normalization of the data -> nn converges better with data between 0 and 1 instead of 0 and 255 for the pixels
#### Results : 
Mean Accuracy: 37.173%
Mean Recall: 36.797%
Mean Precision: 34.686%
#### comments : 
overfitting, needs more layers 

### EXPERIMENT 2 - file: ffnn_exp3_leo.ipynb
#### changes : adding more hidden layers
#### Results : 

#### comments : 
more parameters

## Melen


## Adrián

### General number of instances per class:

| Class               | Samples |
|---------------------|----------|
| Building            | 3594     |
| Small car           | 3324     |
| Truck               | 2210     |
| Bus                 | 1768     |
| Shipping container  | 1523     |
| Storage tank        | 1469     |
| Dump truck          | 1236     |
| Motorboat           | 1069     |
| Excavator           | 789      |
| Fishing vessel      | 706      |
| Cargo plane         | 635      |
| Pylon               | 312      |
| Helipad             | 111      |

### 1 - Original arquitecture

#### Arquitecture:
Flatten -> Activation ReLU -> Dense(13) -> Activation Softmax

#### Hyperparameters:
Train/Test ratio: 0.2
Optimizer: Adam -> learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=True, clipnorm=1.0
Loss: categorical_crossentropy
Batch size: 16
Epochs: 20

##### Callbacks:
model_checkpoint = ModelCheckpoint('model.keras', monitor='val_accuracy', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau('val_accuracy', factor=0.1, patience=10, verbose=1)
early_stop = EarlyStopping('val_accuracy', patience=40, verbose=1)
terminate = TerminateOnNaN()

#### Results:
accuracy: 0.5100 - loss: 836.0410 - val_accuracy: 0.4192 - val_loss: 1378.4482 - learning_rate: 1.0000e-04

#### Issues:

1) Class unbalance
2) The activation ReLU post Flatten doesn´t do anything
3) Categorical Crossentropy could be changed to Focal Loss to treat with class unbalance?
4) Test ReLU variants with no dying ReLU problem, e.g, Leaky ReLU or ELU

#### Confussion Matrix:

##### Best Classes:

| Class to Predict   | Correct Prediction | Most Confused With | Wrong Prediction |
|--------------------|-------------------|--------------------|------------------|
| Cargo Plane        | 0.65              | Building           | 0.32             |
| Small Car          | 0.77              | Motor Boat         | 0.07             |
| Motor Boat         | 0.45              | Small Car          | 0.19             |
| Building           | 0.54              | Cargo Plane        | 0.11             |
| Shopping Center    | 0.49              | Building           | 0.12             |
| Pylon              | 0.50              | Building           | 0.39   

##### Worst Classes:

| Class to Predict   | Correct Prediction | Most Confused With | Wrong Prediction |
|--------------------|-------------------|--------------------|------------------|
| Helipad        | 0.00              | Building           | 0.54             |
| Storage Tank          | 0.04              | Building         | 0.47             |
| Truck         | 0.40              | Small Car          | 0.23             |
| Fishing Vessel           | 0.17              | Building        | 0.23             |


### 2 - 128 Extra Dense Layer

Flatten -> Dense(128) -> Activation ReLU -> Dense(13) -> Activation Softmax

#### Hyperparameters:
Train/Test ratio: 0.2
Optimizer: Adam -> learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=True, clipnorm=1.0
Loss: categorical_crossentropy
Batch size: 16
Epochs: 20

##### Callbacks:
model_checkpoint = ModelCheckpoint('model.keras', monitor='val_accuracy', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau('val_accuracy', factor=0.1, patience=10, verbose=1)
early_stop = EarlyStopping('val_accuracy', patience=40, verbose=1)
terminate = TerminateOnNaN()

#### Results:
accuracy: 0.1685 - loss: 1116.3430 - val_accuracy: 0.1925 - val_loss: 2.3859 - learning_rate: 0.0010

#### Issues:

1) Increassing Complexity of the network without changing any other hyperparameter result in lower results
2) Maybe due to low number of epochs?
3) Maybe due to big first dense layer?

#### Confussion Matrix:

##### Worst Classes:

| Class to Predict   | Correct Prediction | Most Confused With | Wrong Prediction |
|--------------------|-------------------|--------------------|------------------|
| All        | ~ 0.00              | Building           | ~ 1.00             |

### 3 - 32 Extra Dense Layer

Flatten -> Dense(32) -> Activation ReLU -> Dense(13) -> Activation Softmax

#### Hyperparameters:
Train/Test ratio: 0.2
Optimizer: Adam -> learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=True, clipnorm=1.0
Loss: categorical_crossentropy
Batch size: 16
Epochs: 20

##### Callbacks:
model_checkpoint = ModelCheckpoint('model.keras', monitor='val_accuracy', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau('val_accuracy', factor=0.1, patience=10, verbose=1)
early_stop = EarlyStopping('val_accuracy', patience=40, verbose=1)
terminate = TerminateOnNaN()

#### Results:
accuracy: 0.1606 - loss: 475.6181 - val_accuracy: 0.1925 - val_loss: 2.3802 - learning_rate: 0.0010

#### Issues:

1) Lowering Complexity of the network without changing any other hyperparameter result in simmilar results than the more complex network
2) Maybe due to low number of epochs?
3) Maybe due to big first dense layer?

#### Confussion Matrix:

##### Worst Classes:

| Class to Predict   | Correct Prediction | Most Confused With | Wrong Prediction |
|--------------------|-------------------|--------------------|------------------|
| All        | ~ 0.00              | Building           | ~ 1.00             |

### 4 - 32 Extra Dense Layer, higher batch, higher epochs

Flatten -> Dense(32) -> Activation ReLU -> Dense(13) -> Activation Softmax

#### Hyperparameters:
Train/Test ratio: 0.2
Optimizer: Adam -> learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=True, clipnorm=1.0
Loss: categorical_crossentropy
Batch size: 32
Epochs: 40

##### Callbacks:
model_checkpoint = ModelCheckpoint('model.keras', monitor='val_accuracy', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau('val_accuracy', factor=0.1, patience=10, verbose=1)
early_stop = EarlyStopping('val_accuracy', patience=40, verbose=1)
terminate = TerminateOnNaN()

#### Results:
accuracy: 0.1934 - loss: 2.3202 - val_accuracy: 0.1925 - val_loss: 2.3181 - learning_rate: 1.0000e-04

#### Issues:

1) Lowering Complexity of the network changing epochs and batches get better results but still worse than the first approach


#### Confussion Matrix:

##### Worst Classes:

| Class to Predict   | Correct Prediction | Most Confused With | Wrong Prediction |
|--------------------|-------------------|--------------------|------------------|
| All        | ~ 0.00              | Building           | ~ 1.00             |


### 5 - No Extra Dense Layer, higher batch, higher epochs

#### Arquitecture:
Flatten -> Activation ReLU -> Dense(13) -> Activation Softmax

#### Hyperparameters:
Train/Test ratio: 0.2
Optimizer: Adam -> learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=True, clipnorm=1.0
Loss: categorical_crossentropy
Batch size: 32
Epochs: 4 0

##### Callbacks:
model_checkpoint = ModelCheckpoint('model.keras', monitor='val_accuracy', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau('val_accuracy', factor=0.1, patience=10, verbose=1)
early_stop = EarlyStopping('val_accuracy', patience=40, verbose=1)
terminate = TerminateOnNaN()

#### Results:
accuracy: 0.3942 - loss: 1901.7007 - val_accuracy: 0.3971 - val_loss: 1970.3037 - learning_rate: 0.0010

#### Issues:
1) Class unbalance
2) Maybe instead of doing a deeper NN doing a wider hidden layer?
3) This architecture confirm that the limmit of the original NN is a val_accuracy ~40%

#### Confussion Matrix:

##### Best Classes:

| Class to Predict   | Correct Prediction | Most Confused With | Wrong Prediction |
|--------------------|-------------------|--------------------|------------------|
| Cargo Plane        | 0.50              | Building           | 0.30             |
| Small Car          | 0.88              | Motor Boat         | 0.06             |
| Motor Boat         | 0.37              | Small Car          | 0.31             |
| Dump truck           | 0.46              | Small car        | 0.21             |
| Building    | 0.51              | Storage Tank           | 0.18             |
| Storage Tank              | 0.34              | Building           | 0.29   

##### Worst Classes:

| Class to Predict   | Correct Prediction | Most Confused With | Wrong Prediction |
|--------------------|-------------------|--------------------|------------------|
| Helipad        | 0.00              | Building           | 0.50             |
| Storage Tank          | 0.04              | Building         | 0.47             |
| Truck         | 0.00              | Small Car          | 0.42             |
| Fishing Vessel           | 0.06              | Bus        | 0.20             |
| Excavator           | 0.03              | Dump Truck        | 0.51             |
| Pylon           | 0.03              | Building        | 0.69             |

##### Global Metrics:

| Metric          | Value     |
|----------------|----------|
| Mean Accuracy  | 39.707%  |
| Mean Recall    | 28.639%  |
| Mean Precision | 29.928%  |

##### Per Class Metrics:

| Class              | Recall   | Precision | Specificity | Dice     |
|--------------------|----------|-----------|-------------|----------|
| Cargo plane        | 49.587%  | 53.571%   | 98.567%     | 51.502%  |
| Small car          | 87.557%  | 47.064%   | 79.004%     | 61.220%  |
| Bus                | 30.163%  | 31.624%   | 92.904%     | 30.876%  |
| Truck              | 0.225%   | 16.667%   | 99.849%     | 0.443%   |
| Motorboat          | 37.156%  | 24.923%   | 93.092%     | 29.834%  |
| Fishing vessel     | 6.207%   | 34.615%   | 99.528%     | 10.526%  |
| Dump truck         | 45.923%  | 30.836%   | 93.176%     | 36.897%  |
| Excavator          | 2.924%   | 25.000%   | 99.581%     | 5.236%   |
| Building           | 50.831%  | 51.401%   | 88.540%     | 51.114%  |
| Helipad            | 0.000%   | 0.000%    | 99.946%     | 0.000%   |
| Storage tank       | 34.173%  | 23.058%   | 90.870%     | 27.536%  |
| Shipping container | 24.342%  | 38.542%   | 96.576%     | 29.839%  |
| Pylon              | 3.226%   | 11.765%   | 99.593%     | 5.063%   |

### 6 - 13 Extra Dense Layer, higher batch, higher epochs

#### Arquitecture:

Flatten -> Dense(13) -> Activation ReLU -> Dense(13) -> Activation Softmax

#### Hyperparameters:
Train/Test ratio: 0.2
Optimizer: Adam -> learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=True, clipnorm=1.0
Loss: categorical_crossentropy
Batch size: 32
Epochs: 40

##### Callbacks:
model_checkpoint = ModelCheckpoint('model.keras', monitor='val_accuracy', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau('val_accuracy', factor=0.1, patience=10, verbose=1)
early_stop = EarlyStopping('val_accuracy', patience=40, verbose=1)
terminate = TerminateOnNaN()

#### Results:
accuracy: 0.1746 - loss: 576.1681 - val_accuracy: 0.1925 - val_loss: 2.4056 - learning_rate: 0.0010

#### Issues:

1) Lowering Complexity of the network without changing any other hyperparameter result in simmilar results than the more complex network

#### Confussion Matrix:

##### Worst Classes:

| Class to Predict   | Correct Prediction | Most Confused With | Wrong Prediction |
|--------------------|-------------------|--------------------|------------------|
| All        | ~ 0.00              | Building           | ~ 1.00             |


### 7 - 13 Extra Dense Layer, higher batch, higher epochs, modified Adam 2e-3

#### Arquitecture:

Flatten -> Dense(13) -> Activation ReLU -> Dense(13) -> Activation Softmax

#### Hyperparameters:
Train/Test ratio: 0.2
Optimizer: Adam -> learning_rate=2e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=True, clipnorm=1.0
Loss: categorical_crossentropy
Batch size: 32
Epochs: 40

##### Callbacks:
model_checkpoint = ModelCheckpoint('model.keras', monitor='val_accuracy', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau('val_accuracy', factor=0.1, patience=10, verbose=1)
early_stop = EarlyStopping('val_accuracy', patience=40, verbose=1)
terminate = TerminateOnNaN()

#### Results:
accuracy: 0.1895 - loss: 2.4112 - val_accuracy: 0.1925 - val_loss: 2.3657 - learning_rate: 0.0020

#### Issues:

1) Modifying the learning rate of the Optimizer didn't result in better predictions

#### Confussion Matrix:

##### Worst Classes:

| Class to Predict   | Correct Prediction | Most Confused With | Wrong Prediction |
|--------------------|-------------------|--------------------|------------------|
| All        | ~ 0.00              | Building           | ~ 1.00             |

### 8 - 13 Extra Dense Layer, higher batch, higher epochs, modified Adam 3e-3

#### Arquitecture:

Flatten -> Dense(13) -> Activation ReLU -> Dense(13) -> Activation Softmax

#### Hyperparameters:
Train/Test ratio: 0.2
Optimizer: Adam -> learning_rate=3e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=True, clipnorm=1.0
Loss: categorical_crossentropy
Batch size: 32
Epochs: 40

##### Callbacks:
model_checkpoint = ModelCheckpoint('model.keras', monitor='val_accuracy', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau('val_accuracy', factor=0.1, patience=10, verbose=1)
early_stop = EarlyStopping('val_accuracy', patience=40, verbose=1)
terminate = TerminateOnNaN()

#### Results:
accuracy: 0.1952 - loss: 2.3256 - val_accuracy: 0.1925 - val_loss: 2.3244 - learning_rate: 0.0030

#### Issues:

1) Modifying the learning rate of the Optimizer didn't result in better predictions

#### Confussion Matrix:

##### Worst Classes:

| Class to Predict   | Correct Prediction | Most Confused With | Wrong Prediction |
|--------------------|-------------------|--------------------|------------------|
| All        | ~ 0.00              | Building           | ~ 1.00             |

---

``` Once tested deeper networks, and different batch and epochs numbers such as Adam Optimizer tweeks, the best model is the original arquitecture. Different Optimizers will be tested.```

---

### 9 - Original Arquitecture, LossScaleOptimizer

#### Arquitecture:

Flatten -> Activation ReLU -> Dense(13) -> Activation Softmax

#### Hyperparameters:
- Train/Test ratio: 0.2

- Optimizer: Adam -> learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=True, clipnorm=1.0

    LossScaleOptimizer -> Adam, initial_scale=(2.0 ** 15) ,dynamic_growth_steps=2000

- Loss: categorical_crossentropy

- Batch size: 32

- Epochs: 40

##### Callbacks:
model_checkpoint = ModelCheckpoint('model.keras', monitor='val_accuracy', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau('val_accuracy', factor=0.1, patience=10, verbose=1)
early_stop = EarlyStopping('val_accuracy', patience=40, verbose=1)
terminate = TerminateOnNaN()

#### Results:
accuracy: 0.4738 - loss: 884.4443 - val_accuracy: 0.4141 - val_loss: 825.4595 - learning_rate: 1.0000e-04

#### Issues:

1) Unbalance

#### Confussion Matrix:

##### Best Classes:

| Class to Predict   | Correct Prediction | Most Confused With | Wrong Prediction |
|--------------------|-------------------|--------------------|------------------|
| Cargo Plane        | 0.44              | Building           | 0.26             |
| Small Car          | 0.74              | Bus         | 0.06             |
| Bus              | 0.32              | Small Car           | 0.18   
| Motor Boat         | 0.39              | Small Car          | 0.17             |
| Fising Vessel         | 0.35              | Storage Tank          | 0.14             |
| Dump Truck         | 0.31              | Bus          | 0.15             |
| Excavator           | 0.53              | Truck        | 0.09             |
| Building    | 0.36             | Storage Tank           | 0.25             |
| Storage Tank    | 0.41             | Building           | 0.23             |
| Shipping Container    | 0.38             | Truck           | 0.12             |
| Pylons    | 0.56             | Building           | 0.18             |

##### Worst Classes:

| Class to Predict   | Correct Prediction | Most Confused With | Wrong Prediction |
|--------------------|-------------------|--------------------|------------------|
| Truck        | 0.15              | Small Car           | 0.21             |
| Helipad          | 0.00              | Storage Tank         | 0.46             |


##### Global Metrics:

| Metric          | Value     |
|----------------|----------|
| Mean Accuracy  | 41.413%  |
| Mean Recall    | 38.073%  |
| Mean Precision | 36.265%  |

##### Per Class Metrics:

| Class              | Recall   | Precision | Specificity | Dice     |
|--------------------|----------|-----------|-------------|----------|
| Cargo plane        | 43.802%  | 51.456%   | 98.622%     | 47.321%  |
| Small car          | 74.355%  | 62.580%   | 90.521%     | 67.961%  |
| Bus                | 32.337%  | 35.417%   | 93.584%     | 33.807%  |
| Truck              | 15.056%  | 23.509%   | 93.404%     | 18.356%  |
| Motorboat          | 38.532%  | 31.579%   | 94.847%     | 34.711%  |
| Fishing vessel     | 35.172%  | 33.117%   | 97.143%     | 34.114%  |
| Dump truck         | 30.901%  | 34.123%   | 96.048%     | 32.432%  |
| Excavator          | 53.216%  | 44.828%   | 96.871%     | 48.663%  |
| Building           | 36.288%  | 59.545%   | 94.122%     | 45.095%  |
| Helipad            | 0.000%   | 0.000%    | 98.873%     | 0.000%   |
| Storage tank       | 41.007%  | 22.485%   | 88.681%     | 29.045%  |
| Shipping container | 37.829%  | 35.168%   | 93.848%     | 36.450%  |
| Pylon              | 56.452%  | 37.634%   | 98.427%     | 45.161%  |


### 10 - Original Arquitecture, SGD Optimizer

#### Arquitecture:

Flatten -> Activation ReLU -> Dense(13) -> Activation Softmax

#### Hyperparameters:
- Train/Test ratio: 0.2

- Optimizer: SGD -> learning_rate=0.01, momentum=0.0, nesterov=False, weight_decay=None, clipnorm=None, clipvalue=None, global_clipnorm=None, use_ema=False, ema_momentum=0.99, ema_overwrite_frequency=None, loss_scale_factor=None,  gradient_accumulation_steps=None, name='SGD'

- Loss: categorical_crossentropy

- Batch size: 32

- Epochs: 40

##### Callbacks:
model_checkpoint = ModelCheckpoint('model.keras', monitor='val_accuracy', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau('val_accuracy', factor=0.1, patience=10, verbose=1)
early_stop = EarlyStopping('val_accuracy', patience=40, verbose=1)
terminate = TerminateOnNaN()

#### Results:
accuracy: 0.3543 - loss: 637468.6875 - val_accuracy: 0.4328 - val_loss: 675838.7500 - learning_rate: 0.0100

#### Issues:

1) Unbalance

#### Confussion Matrix:

##### Best Classes:

| Class to Predict   | Correct Prediction | Most Confused With | Wrong Prediction |
|--------------------|-------------------|--------------------|------------------|
| Cargo Plane        | 0.55              | Building           | 0.41             |
| Small Car          | 0.81              | Bus         | 0.07             |
| Bus              | 0.39              | Small Car           | 0.23   
| Motor Boat         | 0.39              | Small Car          | 0.25             |
| Excavator           | 0.78              | Building        | 0.09             |
| Building    | 0.68              | Excavator           | 0.07             |

##### Worst Classes:

| Class to Predict   | Correct Prediction | Most Confused With | Wrong Prediction |
|--------------------|-------------------|--------------------|------------------|
| Helipad        | 0.00              | Building           | 0.67             |
| Storage Tank          | 0.09              | Building         | 0.58             |
| Truck         | 0.00              | Small Car          | 0.28             |
| Dump Truck           | 0.17             | Excavator        | 0.41            |
| Excavator           | 0.03              | Dump Truck        | 0.51             |
| Shipping Container           | 0.28              | Building        | 0.21             |
| Pylon           | 0.06              | Building        | 0.87             |

##### Global Metrics:

| Metric          | Value     |
|----------------|----------|
| Mean Accuracy  | 43.280%  |
| Mean Recall    | 33.173%  |
| Mean Precision | 34.054%  |

##### Per Class Metrics:

| Class              | Recall   | Precision | Specificity | Dice     |
|--------------------|----------|-----------|-------------|----------|
| Cargo plane        | 55.372%  | 48.551%   | 98.044%     | 51.737%  |
| Small car          | 81.032%  | 55.683%   | 86.250%     | 66.007%  |
| Bus                | 38.587%  | 35.149%   | 92.253%     | 36.788%  |
| Truck              | 0.000%   | 0.000%    | 100.000%    | 0.000%   |
| Motorboat          | 38.991%  | 27.778%   | 93.743%     | 32.443%  |
| Fishing vessel     | 11.034%  | 57.143%   | 99.667%     | 18.497%  |
| Dump truck         | 16.738%  | 44.828%   | 98.635%     | 24.375%  |
| Excavator          | 77.778%  | 29.103%   | 90.947%     | 42.357%  |
| Building           | 68.283%  | 45.354%   | 80.383%     | 54.505%  |
| Helipad            | 0.000%   | 0.000%    | 99.973%     | 0.000%   |
| Storage tank       | 9.353%   | 35.616%   | 98.646%     | 14.815%  |
| Shipping container | 27.632%  | 44.444%   | 96.953%     | 34.077%  |
| Pylon              | 6.452%   | 19.048%   | 99.539%     | 9.639%   |


### 11 - Original Arquitecture, LossScaleOptimizer, Focal Loss

#### Arquitecture:

Flatten -> Activation ReLU -> Dense(13) -> Activation Softmax

#### Hyperparameters:
- Train/Test ratio: 0.2

- Optimizer: Adam -> learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=True, clipnorm=1.0

    LossScaleOptimizer -> Adam, initial_scale=(2.0 ** 15) ,dynamic_growth_steps=2000

- Loss: categorical_focal_crossentropy

- Batch size: 32

- Epochs: 40

##### Callbacks:
model_checkpoint = ModelCheckpoint('model.keras', monitor='val_accuracy', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau('val_accuracy', factor=0.1, patience=10, verbose=1)
early_stop = EarlyStopping('val_accuracy', patience=40, verbose=1)
terminate = TerminateOnNaN()

#### Results:
accuracy: 0.0660 - loss: 3.7630 - val_accuracy: 0.0723 - val_loss: 3.7374 - learning_rate: 0.0010

#### Issues:

1) Unbalance

#### Confussion Matrix:

##### Worst Classes:

| Class to Predict   | Correct Prediction | Most Confused With | Wrong Prediction |
|--------------------|-------------------|--------------------|------------------|
| All        | ~ 0.00              | Motor Boat           | ~ 1.00             |

### 12 - 13 Extra Dense Layer, higher batch, higher epochs, LossScaleOptimizer

#### Arquitecture:

Flatten -> Dense(32) -> Activation elu -> Dense(13) -> Activation Softmax

#### Hyperparameters:
- Train/Test ratio: 0.2

- Optimizer: Adam -> learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=True, clipnorm=1.0

    LossScaleOptimizer -> Adam, initial_scale=(2.0 ** 15) ,dynamic_growth_steps=2000

- Loss: categorical_crossentropy

- Batch size: 32

- Epochs: 40

##### Callbacks:
model_checkpoint = ModelCheckpoint('model.keras', monitor='val_accuracy', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau('val_accuracy', factor=0.1, patience=10, verbose=1)
early_stop = EarlyStopping('val_accuracy', patience=40, verbose=1)
terminate = TerminateOnNaN()

#### Results:
accuracy: 0.1409 - loss: 260.0886 - val_accuracy: 0.1925 - val_loss: 2.3251 - learning_rate: 0.0010

#### Issues:

1) Unbalance

#### Confussion Matrix:

##### Worst Classes:

| Class to Predict   | Correct Prediction | Most Confused With | Wrong Prediction |
|--------------------|-------------------|--------------------|------------------|
| All        | ~ 0.00              | Motor Boat           | ~ 1.00             |

### 13 - Resizing Layer, 256 Extra Dense Layer, higher batch, higher epochs, LossScaleOptimizer

#### Arquitecture:

Resizing(128,128) -> Flatten -> Dense(256) -> Activation ReLU -> Dense(13) -> Activation Softmax

#### Hyperparameters:
- Train/Test ratio: 0.2

- Optimizer: Adam -> learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=True, clipnorm=1.0

    LossScaleOptimizer -> Adam, initial_scale=(2.0 ** 15) ,dynamic_growth_steps=2000

- Loss: categorical_crossentropy

- Batch size: 32

- Epochs: 40

##### Callbacks:
model_checkpoint = ModelCheckpoint('model.keras', monitor='val_accuracy', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau('val_accuracy', factor=0.1, patience=10, verbose=1)
early_stop = EarlyStopping('val_accuracy', patience=40, verbose=1)
terminate = TerminateOnNaN()

#### Results:
accuracy: 0.1409 - loss: 260.0886 - val_accuracy: 0.1925 - val_loss: 2.3251 - learning_rate: 0.0010

#### Issues:

1) Unbalance

#### Confussion Matrix:

##### Worst Classes:

| Class to Predict   | Correct Prediction | Most Confused With | Wrong Prediction |
|--------------------|-------------------|--------------------|------------------|
| All        | ~ 0.00              | Motor Boat           | ~ 1.00             |


# REGs (2nd report)
## Léo
### EXPERIMENT 1
#### Hyperparameters : 
#### Results : 

### EXPERIMENT 2
#### Hyperparameters : 
#### Results : 

## Melen

# Last Assignment
## Adrián

### EXPERIMENT 1
#### Hyperparameters : 
```
 opt = SGD(learning_rate=5e-4, momentum=0.9, nesterov=True, global_clipnorm=10.0)
model.compile(optimizer=opt, classification_loss='binary_crossentropy', box_loss='ciou', jit_compile=False)
```
```
model_checkpoint = ModelCheckpoint('model.keras', monitor='val_loss', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.5, patience=4, verbose=1)
early_stop = EarlyStopping('val_loss', patience=12, verbose=1)
terminate = TerminateOnNaN()
callbacks = [model_checkpoint, reduce_lr, early_stop, terminate]
```

```
data_augmentation = tf.keras.Sequential(
    layers=[keras_cv.layers.RandomFlip(mode='horizontal_and_vertical', bounding_box_format='xyxy'),
            keras_cv.layers.RandomShear(x_factor=0.2, y_factor=0.2, bounding_box_format='xyxy'),
            keras_cv.layers.RandomColorDegeneration(factor=0.5)])

epochs = 40
batch_size = 4

```

#### Model:
```
prediction_decoder = keras_cv.layers.NonMaxSuppression(bounding_box_format='xyxy', from_logits=False, confidence_threshold=0.2, iou_threshold=0.7)
model = keras_cv.models.YOLOV8Detector.from_preset(preset='yolo_v8_xs_backbone_coco', num_classes=len(categories), load_weights=True, bounding_box_format='xyxy', prediction_decoder=prediction_decoder)
model.summary()
```

#### Results :

##### Confussion Matrix:

| Annotation  \ Prediction | BACKGROUND | Small car | Bus  | Truck | Building |
|-------------------------|------------|-----------|------|-------|----------|
| BACKGROUND              | 0.00       | 0.31      | 0.01 | 0.00  | 0.68     |
| Small car               | 0.56       | 0.44      | 0.00 | 0.00  | 0.00     |
| Bus                     | 0.90       | 0.00      | 0.10 | 0.00  | 0.00     |
| Truck                   | 1.00       | 0.00      | 0.00 | 0.00  | 0.00     |
| Building                | 0.49       | 0.00      | 0.00 | 0.00  | 0.51     |

##### Metrics:

```
> Small car: Recall: 43.535% Precision: 57.917% AP: 32.940%
> Bus: Recall: 10.469% Precision: 32.683% AP: 4.655%
> Truck: Recall: 0.479% Precision: 41.667% AP: 0.363%
> Building: Recall: 50.851% Precision: 54.603% AP: 40.963%
mAccuracy: 33.921%
mRecall: 21.067%
mPrecision: 37.374%
mAP: 19.730%
```

### EXPERIMENT 2
#### Hyperparameters : 
```
 opt = SGD(learning_rate=5e-4, momentum=0.9, nesterov=True, global_clipnorm=10.0)
model.compile(optimizer=opt, classification_loss='binary_crossentropy', box_loss='ciou', jit_compile=False)
```
```
model_checkpoint = ModelCheckpoint('model.keras', monitor='val_loss', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.5, patience=3, verbose=1)
early_stop = EarlyStopping('val_loss', patience=8, verbose=1)
terminate = TerminateOnNaN()
callbacks = [model_checkpoint, reduce_lr, early_stop, terminate]
```

```
data_augmentation = tf.keras.Sequential(
    layers=[keras_cv.layers.RandomFlip(mode='horizontal_and_vertical', bounding_box_format='xyxy'),
            keras_cv.layers.RandomShear(x_factor=0.2, y_factor=0.2, bounding_box_format='xyxy'),
            keras_cv.layers.RandomColorDegeneration(factor=0.5)])

epochs = 40
batch_size = 4

```

#### Model:
```
# Load architecture
from types import SimpleNamespace

import numpy as np
import torch
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

print('Load model')
# Unit 2 describes Faster R-CNN as a two-stage detector: first generate proposals, then refine and classify them.
# We keep transfer learning because the xView subset is much smaller than the original COCO detection benchmark.
class TorchFasterRCNNWrapper:
    def __init__(self, num_classes, max_detections=100, micro_batch_size=1):
        self.num_classes = num_classes
        self.max_detections = max_detections
        self.micro_batch_size = micro_batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector = self._build_detector()
        self.optimizer = None
        self.clip_norm = 10.0

    def _build_detector(self):
        common_kwargs = dict(min_size=640, max_size=640, box_detections_per_img=self.max_detections, box_score_thresh=0.2, trainable_backbone_layers=2)
        try:
            detector = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, **common_kwargs)
            init_msg = 'COCO pretrained Faster R-CNN'
        except Exception as exc:
            print(f'COCO weights unavailable ({exc}). Trying ImageNet backbone...')
            try:
                detector = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=ResNet50_Weights.DEFAULT, **common_kwargs)
                init_msg = 'ImageNet pretrained backbone'
            except Exception as exc_backbone:
                print(f'Backbone weights unavailable ({exc_backbone}). Falling back to random initialization...')
                detector = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, **common_kwargs)
                init_msg = 'random initialization'
        in_features = detector.roi_heads.box_predictor.cls_score.in_features
        detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes + 1)
        detector.to(self.device)
        print(f'Using {init_msg} on {self.device}.')
        return detector

    def summary(self):
        print(self.detector)

    def compile(self, learning_rate=1e-4, momentum=0.9, weight_decay=5e-4, clip_norm=10.0):
        params = [param for param in self.detector.parameters() if param.requires_grad]
        self.optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        self.clip_norm = clip_norm

    def save_weights(self, filepath):
        payload = {'model_state_dict': self.detector.state_dict()}
        if self.optimizer is not None:
            payload['optimizer_state_dict'] = self.optimizer.state_dict()
        torch.save(payload, filepath)

    def load_weights(self, filepath):
        payload = torch.load(filepath, map_location=self.device)
        state_dict = payload['model_state_dict'] if isinstance(payload, dict) and 'model_state_dict' in payload else payload
        self.detector.load_state_dict(state_dict)
        self.detector.to(self.device)
        return self

    def _split_batch(self, images, bbox_dict):
        if hasattr(images, 'numpy'):
            images = images.numpy()
        boxes = bbox_dict['boxes'].numpy() if hasattr(bbox_dict['boxes'], 'numpy') else bbox_dict['boxes']
        classes = bbox_dict['classes'].numpy() if hasattr(bbox_dict['classes'], 'numpy') else bbox_dict['classes']

        image_list, target_list = [], []
        for idx in range(images.shape[0]):
            image = torch.from_numpy(images[idx].astype('float32') / 255.0).permute(2, 0, 1)
            box = boxes[idx].astype('float32')
            label = classes[idx].astype('int64')
            valid = (box[:, 2] > box[:, 0]) & (box[:, 3] > box[:, 1])
            box = box[valid]
            label = label[valid] + 1

            image_list.append(image)
            target_list.append({'boxes': torch.as_tensor(box, dtype=torch.float32), 'labels': torch.as_tensor(label, dtype=torch.int64)})
        return image_list, target_list

    def _train_or_validate(self, dataset, steps, training=True):
        losses = []
        self.detector.train()
        iterator = dataset.take(steps)
        context = torch.enable_grad() if training else torch.no_grad()

        with context:
            for images, bbox_dict in iterator:
                image_list, target_list = self._split_batch(images, bbox_dict)
                for start in range(0, len(image_list), self.micro_batch_size):
                    imgs = [img.to(self.device) for img in image_list[start:start + self.micro_batch_size]]
                    tgts = [{key: value.to(self.device) for key, value in tgt.items()} for tgt in target_list[start:start + self.micro_batch_size]]

                    if training:
                        self.optimizer.zero_grad()

                    loss_dict = self.detector(imgs, tgts)
                    loss = sum(value for value in loss_dict.values())
                    loss_value = float(loss.detach().cpu().item())

                    if training:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.detector.parameters(), self.clip_norm)
                        self.optimizer.step()

                    losses.append(loss_value)

        self.detector.eval()
        return float(np.mean(losses)) if losses else np.inf

    def _monitor_name(self, callback, default='val_loss'):
        return getattr(callback, 'monitor', default) if callback is not None else default

    def _extract_callback(self, callbacks, name):
        for callback in callbacks:
            if callback.__class__.__name__ == name:
                return callback
        return None

    def fit(self, dataset, steps_per_epoch=None, validation_data=None, validation_steps=None, epochs=1, callbacks=None, verbose=1):
        if self.optimizer is None:
            raise RuntimeError('Call compile() before fit().')

        callbacks = callbacks or []
        checkpoint_cb = self._extract_callback(callbacks, 'ModelCheckpoint')
        reduce_lr_cb = self._extract_callback(callbacks, 'ReduceLROnPlateau')
        early_stop_cb = self._extract_callback(callbacks, 'EarlyStopping')

        history = {'loss': [], 'val_loss': []}
        best_metric = np.inf
        checkpoint_saved = False
        reduce_wait = 0
        early_wait = 0

        for epoch in range(epochs):
            train_loss = self._train_or_validate(dataset, steps_per_epoch, training=True)
            history['loss'].append(train_loss)

            metric_name = 'loss'
            metric_value = train_loss
            if validation_data is not None:
                val_loss = self._train_or_validate(validation_data, validation_steps, training=False)
                history['val_loss'].append(val_loss)
                metric_name = 'val_loss'
                metric_value = val_loss

            current_lr = self.optimizer.param_groups[0]['lr']
            if verbose:
                if validation_data is not None:
                    print(f'Epoch {epoch + 1}/{epochs} - loss: {train_loss:.4f} - val_loss: {metric_value:.4f} - lr: {current_lr:.6f}')
                else:
                    print(f'Epoch {epoch + 1}/{epochs} - loss: {train_loss:.4f} - lr: {current_lr:.6f}')

            if not np.isfinite(metric_value):
                print('TerminateOnNaN: stopping training because the monitored loss is not finite.')
                break

            if metric_value < best_metric:
                best_metric = metric_value
                reduce_wait = 0
                early_wait = 0
                if checkpoint_cb is not None:
                    self.save_weights(checkpoint_cb.filepath)
                    checkpoint_saved = True
            else:
                reduce_wait += 1
                early_wait += 1

                if reduce_lr_cb is not None and self._monitor_name(reduce_lr_cb) == metric_name and reduce_wait >= reduce_lr_cb.patience:
                    for group in self.optimizer.param_groups:
                        group['lr'] *= reduce_lr_cb.factor
                    reduce_wait = 0
                    print(f'ReduceLROnPlateau: reducing learning rate to {self.optimizer.param_groups[0]["lr"]:.6f}')

                if early_stop_cb is not None and self._monitor_name(early_stop_cb) == metric_name and early_wait >= early_stop_cb.patience:
                    print(f'EarlyStopping: stopping at epoch {epoch + 1}')
                    break

        if checkpoint_cb is not None and checkpoint_saved:
            self.load_weights(checkpoint_cb.filepath)

        return SimpleNamespace(history=history)

    def predict(self, images, verbose=0):
        if hasattr(images, 'numpy'):
            images = images.numpy()

        batch = []
        for idx in range(images.shape[0]):
            image = torch.from_numpy(images[idx].astype('float32') / 255.0).permute(2, 0, 1).to(self.device)
            batch.append(image)

        self.detector.eval()
        with torch.no_grad():
            outputs = self.detector(batch)

        output = outputs[0]
        boxes = output['boxes'].detach().cpu().numpy()
        scores = output['scores'].detach().cpu().numpy()
        labels = output['labels'].detach().cpu().numpy() - 1

        keep = labels >= 0
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        num = min(len(scores), self.max_detections)
        padded_boxes = np.zeros((1, self.max_detections, 4), dtype=np.float32)
        padded_scores = np.zeros((1, self.max_detections), dtype=np.float32)
        padded_labels = np.zeros((1, self.max_detections), dtype=np.int32)

        if num > 0:
            padded_boxes[0, :num] = boxes[:num]
            padded_scores[0, :num] = scores[:num]
            padded_labels[0, :num] = labels[:num]

        return {'boxes': padded_boxes, 'confidence': padded_scores, 'classes': padded_labels, 'num_detections': np.array([num], dtype=np.int32)}

model = TorchFasterRCNNWrapper(num_classes=len(categories), max_detections=100, micro_batch_size=1)
model.summary()
```

#### Results :

##### Confussion Matrix:

| Annotation  \ Prediction | BACKGROUND | Small car | Bus  | Truck | Building |
|-------------------------|------------|-----------|------|-------|----------|
| BACKGROUND              | 0.00       | 0.31      | 0.01 | 0.00  | 0.68     |
| Small car               | 0.53       | 0.47      | 0.00 | 0.00  | 0.00     |
| Bus                     | 0.92       | 0.00      | 0.08 | 0.00  | 0.00     |
| Truck                   | 1.00       | 0.00      | 0.00 | 0.00  | 0.00     |
| Building                | 0.50       | 0.00      | 0.00 | 0.00  | 0.50     |

##### Metrics:

```
> Small car: Recall: 47.077% Precision: 59.811% AP: 36.542%
> Bus: Recall: 7.969% Precision: 30.357% AP: 3.023%
> Truck: Recall: 0.192% Precision: 22.222% AP: 0.096%
> Building: Recall: 49.890% Precision: 54.232% AP: 40.001%
mAccuracy: 34.452%
mRecall: 21.026%
mPrecision: 33.325%
mAP: 19.916%
```




