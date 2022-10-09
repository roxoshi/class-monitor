"""
Here we will train an ML model on our 
corpus which we have converted to document vectors 
"""
import numpy as np
import os
from preprocessing import TransformText


train_path = os.path.join('data','train.csv')
t = TransformText
t.readcsv = train_path
output_dataset = t.run()
labels = [x[0] for x in output_dataset]
word_vector = np.array([x[1] for x in output_dataset])

print(output_dataset[:10])
