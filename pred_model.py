import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from sklearn.preprocessing import PolynomialFeatures

class pred_block(tf.keras.Model):

    def __init__(self, input_shape):
        super().__init__(name='pred_block')

        self.batch, self.window=input_shape        
        self.min_window=2
        
        self.preprocess=[
        layers.RepeatVector(self.min_window),
        layers.Reshape((-1, 1)),
        layers.Reshape((self.window, -1)),
        layers.Conv1D(self.window//3, self.window//2),
        layers.LSTM(self.window*2),
        layers.Flatten()
        ]


        self.block=[
            layers.Dense(1024),
            layers.Dropout(0.2),
            layers.LeakyReLU(alpha=0.01),
            layers.Dense(1024),
            layers.Dropout(0.2),
            layers.LeakyReLU(alpha=0.01),
            layers.Dense(1)
            ]
        
        self.dummy_input=tf.keras.Input(shape=(10, ), dtype=tf.float32)        
        #dummy_state=tf.constant(np.zeros(shape=(1, 10), dtype=np.float32))
        self.dummy_output=self(self.dummy_input)
        
    def call(self, input_tensor):
        x=input_tensor # shape=(batch, size)
        
        for l in self.preprocess:
            x=l(x)
            
        x=layers.concatenate([x, input_tensor])
        
        for l in self.block:
            x=l(x)
            
        
        return x
        

class pred_model(object):

    def __init__(self, input_shape):
        self.input_shape=input_shape
        self.model=pred_block(self.input_shape)

    def compile(self):
        self.model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam())
            
    def train_model_with_batch(self, trains, labels, batch_size, epochs=10):
        trains=trains.repeat().batch(batch_size)
        labels=labels.repeat().batch(batch_size)
        
        z=zip(trains.take(epochs), labels.take(epochs))
        
        for train, label in z:
            self.model.fit(train, label)  
            
if __name__=='__main__':
    data_path=r'C:\Users\DELL\Desktop\data1.csv'
    data=pd.read_csv(data_path)

    f=lambda x, y, z: 2*np.sin(x)+y**2+10/z
    x=np.linspace(1, 200, 1000)
                                                           
    z=zip(x, x, x)
    x_test=[]
    y_test=[]
    
    for i, j, k in z:
        x_test.append([i, j, k])
        y_test.append(f(i, j, k))

    x_test=np.array(x_test).reshape(-1, 3)
    y_test=np.array(y_test).reshape(-1, 1)
                                                           
    features=data.copy()
    labels=features.pop('f')
    
    features=np.array(features)
    print('before poly', features[0])
    labels=np.array(labels)
    poly=PolynomialFeatures(2)
    features=poly.fit_transform(features)
    datum=features[0]
    
    
    trains=tf.data.Dataset.from_tensor_slices(features)
    labels=tf.data.Dataset.from_tensor_slices(labels)

    # model setting
    
    batch_size=64

    p_model=pred_model((batch_size, datum.shape[0]))

    p_model.compile()

    epochs=1000
    
    # train
    
    p_model.train_model_with_batch(trains, labels, batch_size, epochs=epochs)
    history=p_model.train_model_with_batch(trains, labels, batch_size)
    inputs=p_model.model.dummy_input
    outputs=p_model.model.dummy_output
    m=tf.keras.Model(inputs, outputs)
    tf.keras.utils.plot_model(m, to_file=r'C:\Users\DELL\Desktop\pred_model.png', rankdir='LR', dpi=72, show_shapes=True)
    # plot
    x_test=poly.fit_transform(x_test)    
    y_pred=p_model.model.predict(x_test)
    save_path=r'C:\Users\DELL\Desktop\pred_model'
    p_model.model.save(save_path)
    plt.plot(y_pred, 'ro', label='predicted one')
    plt.plot(y_test, 'bo', label='real one')
    plt.legend()
    plt.show()
        

        
