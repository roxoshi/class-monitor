import os
train_path = os.path.join('data','train.csv')
print(train_path)

with open(train_path, 'r') as f:
    raw_txt = f.readlines()

# convert to format where 1st column is label
# and second column is text





