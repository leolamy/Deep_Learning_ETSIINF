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


## Adrian

