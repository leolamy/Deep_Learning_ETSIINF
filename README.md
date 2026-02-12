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
the mo
Solution -> reduce image size (pooling)

### EXPERIMENT 4 - file: ffnn_exp4_leo.ipynb
#### changes : adding pooling to reduce image size -> reduce amount of parameters
#### results
Mean Accuracy: 38.827%
Mean Recall: 39.409%
Mean Precision: 42.877%
#### comments : 
Best results so far


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

