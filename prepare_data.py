from __future__ import print_function
import numpy as np
import os
import sys
from scipy import ndimage
from six.moves import cPickle as pickle



image_size = 28  # Pixel alr larg
pixel_depth = 255.0  # niveis por  pixel.

def load_letter(folder, min_num_images):
  """Carrega os dados."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size, 3),
                         dtype=np.float32)
  print(folder)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      image_data = ndimage.imread(image_file).astype(float)
      print(image_data.shape)
      image_data.resize(image_size, image_size, 3)
      dataset[num_images, :, :, :] = image_data
      num_images = num_images + 1
    except IOError as e:
      print('Nao pode ser lido:', image_file, ':', e, '- it\'s ok, saindo.')

  dataset = dataset[0:num_images, :, :, :]
  if num_images < min_num_images:
    raise Exception('Muito poucas imagens: %d < %d' %
                    (num_images, min_num_images))

  print('tensor do dataset completo:', dataset.shape)
  print('Media:', np.mean(dataset))
  print('Desvio padrao:', np.std(dataset))
  return dataset

def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []

  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s Ja existente - saindo.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Nao foi possivel salvar em ', set_filename, ':', e)

  return dataset_names

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  test_dataset, test_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes

  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # mistura as imagens para validacao e teste aleatorio
        np.random.shuffle(letter_set)
        if test_dataset is not None:
          test_letter = letter_set[:vsize_per_class, :, :, :]
          test_dataset[start_v:end_v, :, :, :] = test_letter
          test_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class

        train_letter = letter_set[vsize_per_class:end_l, :, :, :]
        train_dataset[start_t:end_t, :, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Nao foi possivel processar os dados de', pickle_file, ':', e)
      raise

  return test_dataset, test_labels, train_dataset, train_labels

def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size, 3), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels




if __name__ == '__main__' :

  cwd = os.getcwd()
  folders = os.listdir(cwd)
  train_folders = []
  for folder in folders:
      if os.path.isdir(folder) and folder[0] != ".":
          train_folders.append(folder)
  datasets = maybe_pickle(train_folders, 17)

  train_size = 84
  valid_size = 18

  test_dataset, test_labels, train_dataset, train_labels = merge_datasets(
    datasets, train_size, valid_size)

  print('Training:', train_dataset.shape, train_labels.shape)
  print('Testing:', test_dataset.shape, test_labels.shape)

  train_dataset, train_labels = randomize(train_dataset, train_labels)
  test_dataset, test_labels = randomize(test_dataset, test_labels)

  pickle_file = 'data.pickle'
  print(train_folders)
  try:
    f = open(pickle_file, 'wb')
    save = {
      'train_dataset': train_dataset,
      'train_labels': train_labels,
      'test_dataset': test_dataset,
      'test_labels': test_labels,
      'labels':train_folders,
      }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
  except Exception as e:
    print('Nao foi possivel salvar em ', pickle_file, ':', e)
    raise
