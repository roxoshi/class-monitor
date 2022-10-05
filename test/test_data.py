import os
train_path = os.path.join('data','train.csv')
print(train_path)

with open(train_path, 'r') as f:
    raw_txt = f.readlines()

print("text is: \n",raw_txt[0])

