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

### RECAP FIRST 7 EXPERIMENTS
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

### EXPERIMENT 9 - ffnn_final1_leo.ipynb
#### changes - epoch 20->30 / batch_size : 16->32 / adding layers (1024->512->256 to generalize the model)
#### results : 
validation accuracy: 60.6% best one
mean recall: 26% -> fail on little objects (hard to see the shapes but normal for a ffNN not a CNN)

### ARCHITECTURE PROPOSITIONS (we have to provide 3 architectures) 
- EXPERIMENT 9
    - epoch = 60
    - batch_size = 32
    - 3 layers (1024->512->256)
```python
# Load architecture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Input, AveragePooling2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2

print('Load model')
model = Sequential()
model.add(Input(shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Flatten())
# adding hidden layers to improve model's complexity 
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(len(categories)))
model.add(Activation('softmax'))
model.summary()
```

### BEST MODEL FOR NOW


## Melen


## Adrian

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

