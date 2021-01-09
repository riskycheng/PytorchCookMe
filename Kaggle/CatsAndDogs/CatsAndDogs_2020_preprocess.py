import os
import shutil
data_dir = '/home/jiache/dataset/cats_and_dogs_2020/cats_vs_dogs'
files = os.listdir(data_dir)
for file in files:
    if not file.endswith('.jpg'):
        continue
    if file.startswith('cat'):
        shutil.move(os.path.join(data_dir, file), os.path.join(data_dir, 'cats/' + file))
    elif file.startswith('dog'):
        shutil.move(os.path.join(data_dir, file), os.path.join(data_dir, 'dogs/' + file))