### Architecture

```
# Load architecture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Resizing, BatchNormalization, Dropout, RandAugment, LayerNormalization

print('Load model')
model = Sequential()
model.add(RandAugment(value_range=(0, 255), num_ops=2, factor=0.5, seed=42,input_shape=(224, 224, 3)))
# model.add(Resizing(height=200,width=200,interpolation="bicubic"))
model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('gelu'))
model.add(Dropout(rate=0.25))
model.add(Dense(512))
model.add(LayerNormalization())
model.add(Activation('gelu'))
model.add(Dropout(rate=0.25))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('gelu'))
model.add(Dropout(rate=0.25))
model.add(Dense(128))
model.add(LayerNormalization())
model.add(Activation('gelu'))
model.add(Dropout(rate=0.25))
model.add(Dense(len(categories)))
model.add(Activation('softmax'))
model.summary()
```

### hyperparameters

```
from tensorflow.keras.optimizers import Adam, LossScaleOptimizer, Nadam
from tensorflow.keras.losses import CategoricalFocalCrossentropy

# Learning rate is changed to 0.001
opt=Nadam(learning_rate=0.001, beta_1=0.9,beta_2=0.999,epsilon=1e-07,weight_decay=None,clipnorm=None,clipvalue=None,global_clipnorm=None,use_ema=False,ema_momentum=0.99,ema_overwrite_frequency=None,loss_scale_factor=None,gradient_accumulation_steps=None,name="nadam")
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

```

```
# Generate the list of objects from annotations
objs_train = [(ann.filename, obj) for ann in anns_train for obj in ann.objects]
objs_valid = [(ann.filename, obj) for ann in anns_valid for obj in ann.objects]
# Generators
batch_size = 32
train_generator = generator_images(objs_train, batch_size, do_shuffle=True)
valid_generator = generator_images(objs_valid, batch_size, do_shuffle=False)
```

```
import math
import numpy as np

print('Training model')
epochs = 80
train_steps = math.ceil(len(objs_train)/batch_size)
valid_steps = math.ceil(len(objs_valid)/batch_size)
h = model.fit(train_generator, steps_per_epoch=train_steps, validation_data=valid_generator, validation_steps=valid_steps, epochs=epochs, callbacks=callbacks, verbose=1)
# Best validation model
best_idx = int(np.argmax(h.history['val_accuracy']))
best_value = np.max(h.history['val_accuracy'])
print('Best validation model: epoch ' + str(best_idx+1), ' - val_accuracy ' + str(best_value))
```