# munge_data.py
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import json

# Load and preprocess pokemon data
df = pd.read_csv('data/pokemon.csv')

# Get unique labels
LABELS = df['Type1'].unique()
print(LABELS)


def merge_type_strings(row):
    t1 = row['Type1']
    t2 = row['Type2']

    if t2 is np.nan:
        return t1

    return t1 + ' ' + t2


df['Type'] = df.apply(lambda row: merge_type_strings(row), axis=1)

# Preliminary data exploration - Analyze label frequencies
label_freq = df['Type'].apply(lambda s: str(s).split(' ')).explode().value_counts().sort_values(ascending=False)

# Bar plot
mpl.style.use("fivethirtyeight")
plt.figure(figsize=(12, 10))
sns.barplot(y=label_freq.index.values, x=label_freq, order=label_freq.index)
plt.title("Label frequency", fontsize=14)
plt.xlabel("")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Data Preparation - Fix Type column so that it's a list of Types
df['Type'] = df['Type'].apply(lambda s: [l for l in str(s).split(' ')])
print(df.head())

# Train/Validation/Test split
X_temp, X_test, y_temp, y_test = train_test_split(df['Name'], df['Type'], test_size=0.05)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1)

print("Training set size: \t", len(X_train))
print("Validation set size: \t", len(X_val))
print("Test set size: \t\t", len(X_test))

# Add label for filepath
X_train = [os.path.join("./data/", str(f) + '.png') for f in X_train]
X_val = [os.path.join("./data/", str(f) + '.png') for f in X_val]
X_test = [os.path.join("./data/", str(f) + '.png') for f in X_test]

print(X_train[:3])

# Turn labels into lists
y_train = list(y_train)
y_val = list(y_val)
y_test = list(y_test)

print(y_train[:3])

# View data - display some images
fig, axes = plt.subplots(5, 5, figsize=(10, 10))

for i, ax in enumerate(axes.reshape(-1)):
    img_path = X_train[i]
    label = y_train[i]

    ax.set_title(label, size=10)
    ax.imshow(Image.open(str(img_path)))
    ax.axis(False)

plt.show()

# Label encoding using MultiLabelBinarizer
type_encoding = {}

mlb = MultiLabelBinarizer()
mlb.fit(y_train)

print("Labels: ")
# Loop over all labels and show
N_LABELS = len(mlb.classes_)
for i, label in enumerate(mlb.classes_):
    print("{}. {}".format(i, label))
    type_encoding[i] = label

# Transform the targets of the training and test sets
y_train_bin = mlb.transform(y_train)
y_val_bin = mlb.transform(y_val)
y_test_bin = mlb.transform(y_test)

# Print example of movie posters and their binary targets
for i in range(3):
    print(X_train[i], y_train_bin[i])

# Verify data shapes
print(y_train_bin.shape, y_val_bin.shape, y_test_bin.shape)
print(len(X_train), len(X_val), len(X_test))

# Save X to file
np.savetxt('data/munged/X_train.csv', np.array(X_train), fmt='%s')
np.savetxt('data/munged/X_val.csv', np.array(X_val), fmt='%s')
np.savetxt('data/munged/X_test.csv', np.array(X_test), fmt='%s')

# Save y to file
np.savetxt('data/munged/y_train.csv', y_train_bin, delimiter=',')
np.savetxt('data/munged/y_val.csv', y_val_bin, delimiter=',')
np.savetxt('data/munged/y_test.csv', y_test_bin, delimiter=',')

# Save type encoding
json_tmp = json.dumps(type_encoding)

with open("data/munged/labels.json", "w") as f:
    f.write(json_tmp)