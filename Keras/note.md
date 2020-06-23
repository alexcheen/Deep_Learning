# keras callbacks

  - model checkpointing
  - early stopping
  - dynamic modify parameters
  - record/verify results;
  - visualized

```python
keras.callbacks.ModelCheckpoint
keras.callbacks.EarlyStopping
keras.callbacks.LearningRateScheduler
keras.callbacks.ReduceLROnPlateau
keras.callbacks.CSVLogger
```

## ModelCheckpoint & EarlyStopping


```python
import keras
callbacks_list =  [
    keras.callbacks.EarlyStopping(monitor='acc', patience=1,),
    keras.callbacks.ModelCheckpoint(filepath='my_model.h5', monitor='val_loss', save_best_only=True,)
]
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.fit(x, y, epochs=10, batch_size=12, callbacks=callbacks_list, validation_data=(x_val, y_val))
```

## ReduceLROnPlateau


```python
callbacks_list = [
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.1, patience=10)
]
model.fit(x, y, epochs=10, batch_size=32, callbacks=callbacks=callbacks_list, validation_data=(x_val, y_val))
```

## custom callbacks

 - on_epoch_begin/on_epoch_end
 - on_batch_begin/on_batch_end
 - on_train_begin/on_train_end

```pytyon
#demo
import keras
import numpy as np
class ActivationLogger(keras.callbacks.Callback):

    def set_model(self, model):
        self.model = model
        layer_output = [layer.output for layer in model.layers]
        self.activations_model = keras.models.Model(model.input, layer_outpus)
    
    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is None:
            raise RuntimeError('Requires validation_data.')
        validation_sample = self.validation_data[0][0:1]
        f = open('activation_at_epoch_' + str(epoch) + '.npz', 'w')
        np.savez(f, activations)
        f.close()
```

# TensorBoard

```python
callbacks = [
    keras.callbacks.TensorBoard(log_dir='my_log_dir', histogram_freq=1, embeddings_freq=1)
]
history = model.fit(x_train, y_train, epochs=20, batch_size=128, validation_spilt=0.2, callbacks=callbacks)
```
# BatchNormalization

```python
conv_model.add(layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)))
conv_model.add(layers.BatchNormalization())
conv_model.summary()
```
```
Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_10 (Conv2D)           (None, 26, 26, 32)        320       
_________________________________________________________________
batch_normalization_10 (Batc (None, 26, 26, 32)        128       
=================================================================
Total params: 448
Trainable params: 384
Non-trainable params: 64
_________________________________________________________________
```
批归一化这里的参数为(32+32+32+32), 分别为
 1. 归一化的均值方差，为不可训练参数 
 2. 对归一化的数据重新做 均值方差变换，为可训练参数


# Depthwise separable convolution

```python
from keras.models import Sequential, Model
from keras import layers
height = 64
width = 64
channels = 3
num_classes = 10
model = Sequential()
model.add(layers.SeparableConv2D(32, 3, activation='relu', input_shape=(height, width, channels,)))
model.add(layers.SeparableConv2D(64, 3, activation='relu'))
model.add(layers.MaxPooling2D(2))
model.add(layers.SeparableConv2D(64, 3, activation='relu'))
model.add(layers.SeparableConv2D(128, 3, activation='relu'))
model.add(layers.MaxPooling2D(2))
model.add(layers.SeparableConv2D(64, 3, activation='relu'))
model.add(layers.SeparableConv2D(128, 3, activation='relu'))
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
```

这里注意参数数量
```
Y=W2(W1*X+B1)+B2=W2*W1*X+W2*B1+B2
```
通过以上公式可以看出在深度可分离卷积中的逐通道卷积是不用加偏置的，在生成特征图谱的时候再加入偏置即可，省去一些参数。
```
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
separable_conv2d_1 (Separabl (None, 62, 62, 32)        155       
_________________________________________________________________
separable_conv2d_2 (Separabl (None, 60, 60, 64)        2400      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 30, 30, 64)        0         
_________________________________________________________________
separable_conv2d_3 (Separabl (None, 28, 28, 64)        4736      
_________________________________________________________________
separable_conv2d_4 (Separabl (None, 26, 26, 128)       8896      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 13, 13, 128)       0         
_________________________________________________________________
separable_conv2d_5 (Separabl (None, 11, 11, 64)        9408      
_________________________________________________________________
separable_conv2d_6 (Separabl (None, 9, 9, 128)         8896      
_________________________________________________________________
global_average_pooling2d_1 ( (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 32)                4128      
_________________________________________________________________
dense_2 (Dense)              (None, 10)                330       
=================================================================
Total params: 38,949
Trainable params: 38,949
Non-trainable params: 0
_________________________________________________________________
```
例如 
```
155 = (3*3*3)+(1*1*3+1)*32
```
