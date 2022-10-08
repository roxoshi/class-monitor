from preprocessing import TransformText

import os
train_path = os.path.join('data','train.csv')
print(train_path)

with open(train_path, 'r') as f:
    raw_txt = f.readlines()



t = TransformText
t.readcsv = train_path
output_dataset = t.run()
print(output_dataset[:3])