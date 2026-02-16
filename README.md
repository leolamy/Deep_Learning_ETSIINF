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

### EXPERIMENT 3 - file: ffnn_exp3_leo.ipynb
#### changes : adding more hidden layers 
#### Results : 
Mean Accuracy: 20.213%
Mean Recall: 15.045%
Mean Precision: 5.799%

#### comments : 
1 layer of 1024 neurons + relu
1 layer of 512 neurons + relu
154 millions of parameters
Image is too big when it is flattened (150 528 pixels dimensions(224*224*3))
raw pixels + too much parameters -> optimizer is lost
the model is not enough complex to handle so much parameter -> increase nb of layers (not enough computation?)
Solution -> reduce image size (pooling)

### EXPERIMENT 4 - file: ffnn_exp4_leo.ipynb
#### changes : adding pooling to reduce image size -> reduce amount of parameters
#### results
Mean Accuracy: 38.827%
Mean Recall: 39.409%
Mean Precision: 42.877%
#### comments : 
Best results so far
resolution lack on small objects
truck: 4.1% recall
Fishing vessel: 5.4% recall
small car: 33% recall 

### EXPERIMENT 5 - file: ffnn_exp5_leo.ipynb
#### changes : test with another pooling for smaller objects (MaxPooling + 2x2   pooling instead of 4x4 -> have better shapes)
    -> more parameters but more precise to distinct smaller objects (the issue we met)
    -> ~10,000,000 to 17,354,765 parameters
#### results
MaxPooling(3,3) = 
Mean Accuracy: 37.600%
Mean Recall: 30.620%
Mean Precision: 34.519%
17,354,765 parameters

MaxPooling(2,2) = 

Mean Accuracy: 34.400%
Mean Recall: 37.734%
Mean Precision: 33.915%

Total params: 39,067,661 

to much noise -> the model does not converge

### EXPERIMENT 6 - file : ffnn_exp6_leo.ipynb
#### changes : adding batch norm -> stabilize model to help it to converge (noyé dans le bruit actuellement)
#### results 
Mean Accuracy: 34.933%

Mean Recall: 25.160%

Mean Precision: 27.214%

Model starts to learn -> 74% training accuracy but 57% validation accuracy => OVERFITTING
Mean accuracy low -> ignore complicated classes and focuses on simple classes / sacrifies diversity for score
=> SOLUTION : Regularization L2 -> pushes the model not to be too precise on details -> learn more robust and general patterns // punishes big weights // the training accuracy will drop but the avg recall will improve

Some non-trainable params (not the case before): due to batchnorm -> the network is observing mean and variance of those non trainable params in order to stabilize the network -> statistical memory for each step of the network
 Total params: 39,073,805 (149.05 MB)
 Trainable params: 39,070,733 (149.04 MB)
 Non-trainable params: 3,072 (12.00 KB)

 ### EXPERIMENT 7 - file : ffnn_exp7_leo.ipynb
#### changes : adding regularization to force the model to generalize => improve mean recall (performance on all classes and not only on the easiest one)
#### results : 
Mean Accuracy: 27.787%
Mean Recall: 14.680%
Mean Precision: 17.266% 
=> not working

## RECAP FIRST 7 EXPERIMENTS
Exp,Change,Result,Status,Why?
1 & 2,Baseline + Normalization,~37% Acc,Limit Reached,Model too simple to learn complex features.
3,Added Hidden Layers,~20% Acc,Failed,Exploding parameters (154M). Input image is too big for a dense net; optimizer cannot converge.
4,"AveragePooling(4,4)",~38.8% Acc,Mixed,"Best stability so far, but blind to small objects (Trucks/Boats) due to blurring."
5,"MaxPooling(2,2)",~34% Acc,Unstable,Higher resolution brought back too much noise/parameters. Model struggled to converge.
6,Added BatchNorm,74% Train / 57% Val,Overfitting,Model learned too well. It memorized the training data but failed to generalize (high variance).
7,Added L2 Regularization,~27% Acc,Failed,"Penalty too harsh. The model collapsed (underfitting), predicting only one class to minimize the penalty."
SOLUTION => remove penalty L2, keep batch norm, add dropout, reduce size layers (summarize info, keep only the important one)

### EXPERIMENT 8 - ffnn_exp8_leo.ipynb
#### changes - remove penalty L2, reducing layers, no dropout, pooling 3x3
#### results : 
NO DROPOUT
training: 0.71 accuracy max
Validation : Mean Accuracy: 35.253%
Mean Recall: 25.108%
Mean Precision: 28.520%
WITH DROPOUT
training: 0.542
validation: 
Mean Accuracy: 40.853%
Mean Recall: 31.170%
Mean Precision: 40.354%

### EXPERIMENT 9 - ffnn_exp9_leo.ipynb
#### changes - epoch 20->30 / batch_size : 16->32 / adding layers (1024->512->256 to generalize the model)
#### results : 
validation accuracy: 60.6% best one
mean recall: 26% -> fail on little objects (hard to see the shapes but normal for a ffNN not a CNN)

### ARCHITECTURE PROPOSITIONS (we have to provide 3 architectures) 
- EXPERIMENT 10
    - epoch = 42
    - batch_size = 32
    - 3 layers (1024->512->256)
    - swish activation
```python
# Load architecture
# Load architecture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Input, AveragePooling2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.losses import CategoricalFocalCrossentropy
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.activations import swish

model = Sequential()
model.add(Input(shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(3, 3))) # pooling to reduce amount of parameters
model.add(Flatten())

# Layer 1
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('swish')) 
#model.add(Dropout(0.4)) 

# Layer 2
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('swish'))
#model.add(Dropout(0.4))

# Layer 3
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('swish'))
#model.add(Dropout(0.4))

# Output
model.add(Dense(len(categories)))
model.add(Activation('softmax'))

# Scheduler & Compiler
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001, 
    decay_steps=40 * (len(anns_train) // 32)
)

model.compile(
    optimizer=Adam(learning_rate=lr_schedule, weight_decay=1e-4),
    loss=CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0),
    metrics=['accuracy']
)
model.summary()
```
scheduler -> regulate speed at which model converges
No regularization -> adding normalization / loss
99% train 57% validation 

- EXPERIMENT 11 
    - epoch = 42
    - batch_size = 32
    - 3 layers (512->256->128)
    - selu activation
    - Adam Optimizer
goal : generalize model and try not to diverge and fin little classes
```python
# Load architecture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Input, AveragePooling2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.losses import CategoricalFocalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import swish

model = Sequential()
model.add(Input(shape=(224, 224, 3)))
# Réduit le bruit et le nombre de paramètres drastiquement
model.add(MaxPooling2D(pool_size=(4, 4))) 
model.add(Flatten())

# Layer 1
model.add(Dense(512, kernel_initializer='lecun_normal')) 
#model.add(BatchNormalization())
model.add(Activation('selu')) 
#model.add(Dropout(0.4)) 

# Layer 2
model.add(Dense(256, kernel_initializer='lecun_normal')) 
#model.add(BatchNormalization())
model.add(Activation('selu'))
#model.add(Dropout(0.4))

# Layer 3
model.add(Dense(128, kernel_initializer='lecun_normal')) 
#model.add(BatchNormalization())
model.add(Activation('selu'))
#model.add(Dropout(0.4))

# Output
model.add(Dense(len(categories)))
model.add(Activation('softmax'))

# Learning rate légèrement plus bas car SELU est très dynamique
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.0005, 
    decay_steps=40 * (len(anns_train) // 32)
)

model.compile(
    optimizer=Adam(learning_rate=lr_schedule),
    # Gamma plus fort pour récupérer les "Fishing Vessels" la ou recall plus bas 
    loss=CategoricalFocalCrossentropy(alpha=0.25, gamma=3.0), 
    metrics=['accuracy']
)
model.summary()
```
SELU -> données "s'auto-normalisent", utiliser une initialisation type lecun pour garder une variance constante tout au long du réseau (1/n) pour garder des données normalisée (remplacer batch norm)
training:
val_accuracy 0.602666676044
validation:
Mean Accuracy: 24.747%
Mean Recall: 19.766%
Mean Precision: 29.996%

- Experiment 12
    - epoch = 42
    - batch_size = 32
    - 3 layers (512->256->128)
    - swish activation
    - AdamW Optimizer
    goal : reduce overfitting
```python
# Load architecture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Input, AveragePooling2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.losses import CategoricalFocalCrossentropy
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.activations import swish
import tensorflow as tf

model = Sequential()
model.add(Input(shape=(224, 224, 3)))
# Réduit le bruit et le nombre de paramètres drastiquement
model.add(MaxPooling2D(pool_size=(4, 4))) 
model.add(Flatten())

# Layer 1
model.add(Dense(512, kernel_initializer='lecun_normal')) 
#model.add(BatchNormalization())
model.add(Activation('swish')) 
#model.add(Dropout(0.4)) 

# Layer 2
model.add(Dense(256, kernel_initializer='lecun_normal')) 
#model.add(BatchNormalization())
model.add(Activation('swish'))
#model.add(Dropout(0.4))

# Layer 3
model.add(Dense(128, kernel_initializer='lecun_normal')) 
#model.add(BatchNormalization())
model.add(Activation('swish'))
#model.add(Dropout(0.4))

# Output
model.add(Dense(len(categories)))
model.add(Activation('softmax'))

# Learning rate légèrement plus bas car SELU est très dynamique
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.0005, 
    decay_steps=40 * (len(anns_train) // 32)
)

model.compile(
    optimizer=AdamW(learning_rate=lr_schedule, weight_decay=1e-4),
    # Gamma plus fort pour récupérer les "Fishing Vessels" la ou recall plus bas 
    loss=CategoricalFocalCrossentropy(alpha=0.25, gamma=3.0), 
    metrics=['accuracy']
)
model.summary()
```
POOLING 4x4
training:
0.5871999859809875
validation:

Mean Accuracy: 52.107%
Mean Recall: 44.971%
Mean Precision: 52.036%

POOLING 3x3
Mean Accuracy: 51.787%
Mean Recall: 44.783%
Mean Precision: 51.821%


- Exp 13
```python
model = Sequential()
model.add(Input(shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Flatten())

# Layer 1
model.add(Dense(384, kernel_initializer='he_normal')) 
#model.add(BatchNormalization())
model.add(Activation('swish')) 
#model.add(Dropout(0.4)) 

# Layer 2
model.add(Dense(192, kernel_initializer='he_normal')) 
#model.add(BatchNormalization())
model.add(Activation('swish'))
#model.add(Dropout(0.4))

# Layer 3
model.add(Dense(96, kernel_initializer='he_normal')) 
#model.add(BatchNormalization())
model.add(Activation('swish'))
#model.add(Dropout(0.4))

# Output
model.add(Dense(len(categories)))
model.add(Activation('softmax'))

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001, 
    decay_steps=50 * (len(anns_train) // 32)
)

model.compile(
    optimizer=AdamW(learning_rate=lr_schedule, weight_decay=0.001), 
    loss=CategoricalFocalCrossentropy(alpha=0.25, gamma=2.5), 
    metrics=['accuracy']
)
```
Mean Accuracy: 50.187%

Mean Recall: 42.737%

Mean Precision: 51.096%


- Exp 14
```python
model = Sequential()
model.add(Input(shape=(224, 224, 3)))

# On garde le Pooling 4x4 qui marche bien
model.add(MaxPooling2D(pool_size=(4, 4))) 
model.add(Flatten())

# Layer 1 : On réduit encore (256 suffisent largement après un pooling 4x4)
model.add(Dense(256, kernel_initializer='he_normal')) 
model.add(Activation('swish')) 

# Layer 2 : Le goulot d'étranglement (Bottleneck)
# Force le modèle à résumer l'image en seulement 64 concepts
model.add(Dense(64, kernel_initializer='he_normal'))
model.add(Activation('swish'))

# Output Directement après (Pas de 3ème couche cachée inutile)
model.add(Dense(len(categories)))
model.add(Activation('softmax'))

# Scheduler
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001, 
    decay_steps=50 * (len(anns_train) // 32)
)

model.compile(
    # Weight Decay agressif (0.005) pour tuer l'overfitting
    optimizer=AdamW(learning_rate=lr_schedule, weight_decay=0.005),
    loss=CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0), 
    metrics=['accuracy']
)
```
Mean Accuracy: 48.107%

Mean Recall: 44.672%

Mean Precision: 50.883%

- EXP 15  - ffnn-exp15-leo
With Melen improvement on the images 
Mean Accuracy: 52.373%
Mean Recall: 44.558%
Mean Precision: 54.660%

- EXO 15 - ffnn-exp15bis-leo 
Resizing instead of pooling 
Mean Accuracy: 52.907%
Mean Recall: 44.815%
Mean Precision: 52.024%

- EXP 16 - ffnn_exp16_leo
bicubic sizing 56x56 images 
```python
# Load architecture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Input
from tensorflow.keras.losses import CategoricalFocalCrossentropy
from tensorflow.keras.optimizers import Nadam 
import tensorflow as tf

model = Sequential()
model.add(Input(shape=(56, 56, 3)))
model.add(Flatten())

# Layer 1
model.add(Dense(1024, kernel_initializer='lecun_normal')) 
#model.add(BatchNormalization())
model.add(Activation('selu')) 
#model.add(Dropout(0.4)) 

# Layer 2
model.add(Dense(512, kernel_initializer='lecun_normal')) 
#model.add(BatchNormalization())
model.add(Activation('selu')) 
#model.add(Dropout(0.4)) 

# Layer 3
model.add(Dense(256, kernel_initializer='lecun_normal')) 
#model.add(BatchNormalization())
model.add(Activation('selu')) 
#model.add(Dropout(0.4)) 

# Output
model.add(Dense(len(categories)))
model.add(Activation('softmax'))

# Learning rate légèrement plus bas car SELU est très dynamique
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001,
    decay_steps=40 * (len(anns_train) // 32)
)

model.compile(
    optimizer=Nadam(learning_rate=lr_schedule), # Adam with nesterov momentum 
    loss=CategoricalFocalCrossentropy(), 
    metrics=['accuracy']
)
model.summary()
```
```python
    # Load image
                img = load_geoimage(filename)
                # Convertir en tenseur Tensorflow pour le traitement
                img_tensor = tf.convert_to_tensor(img)
                # Convertir en float32 et normaliser entre 0 et 1
                img_tensor = tf.image.convert_image_dtype(img_tensor, tf.float32)
                # Redimensionnement via bicubic (+ précis)
                img_resized = tf.image.resize(img_tensor, [56, 56], method='bicubic')
                # On repasse en Numpy pour construire le batch
                images.append(img_resized.numpy())                
                probabilities = np.zeros(len(categories))
                probabilities[list(categories.values()).index(obj.category)] = 1
                labels.append(probabilities)
```
Results : 
Mean Accuracy: 59.467%
Mean Recall: 53.411%
Mean Precision: 56.207%``

To address the curse of dimensionality inherent to dense architectures, we implemented a unified preprocessing pipeline that downsamples all inputs to $56 \times 56$ using bicubic interpolation, ensuring that essential spatial features are preserved while maintaining a manageable parameter count for both training and inference.

## Melen
### EXPERIMENT 1 - file: basic_ffnn_exp1_melen.ipynb
#### changes : default, without changing the template
#### Results : 
Mean Accuracy: 37.013%
Mean Recall: 24.446%
Mean Precision: 30.915%

### EXPERIMENT 2 - file: ffnn_exp2_melen.ipynb
#### changes : add Normalization for 255 in pixels to have between 0 and 1 to anticipate Explosion Gradients and a stratify(labels) in train_test_split to maintain class proportions in both sets, ensuring more reliable validation metrics given dataset imbalance.
#### Results : 
Mean Accuracy: 39.093%
Mean Recall: 27.869%
Mean Precision: 33.115%

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

