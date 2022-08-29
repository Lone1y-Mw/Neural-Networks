import pandas as pd

data = pd.read_csv('gpascore.csv')
data = data.dropna()

# data.isnull().sum()
# data.fillna()
# data[''].min() / .max() / .count()

xData = [ ]
yData = data['admit'].values

for i, rows in data.iterrows():
     xData.append([ rows['gre'], rows['gpa'], rows['rank'] ])


from pickletools import optimize
import numpy as np
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit( np.array(xData), np.array(yData), epochs=1000 )


predict = model.predict( [ [gre1, gpa1, rank1], [gre2, gpa2, rank2] ])
print(predict)