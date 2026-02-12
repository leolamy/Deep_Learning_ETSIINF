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

