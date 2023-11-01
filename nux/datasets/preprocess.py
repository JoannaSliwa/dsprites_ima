import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import random

def filter_func(example, epsilon=0.1):
    '''
    A filter function to only include examples where the rotation is between 0 and pi/4, and the shape is a square
    :param example, all images:
    :return example, only  filtered ones:
    '''
    rotation = example["value_orientation"]
    shape = example["value_shape"]
    return (rotation >= 0+epsilon) and (rotation <= (np.pi/2)-epsilon) and (shape == 1)

def change_color(example):
    '''
    Function to change the color of the image
    :param example, with white objects:
    :return example that can be one from 18 colors defined in list_of_colors:
    '''

    ind = tf.random.uniform(shape=(), minval=0, maxval=len(list_of_colors)-1, dtype=tf.int32)
    color = list_of_colors[ind] # Randomly choose a color from the list
    color_label = list_of_colors_labels[ind]
    example["value_color"] = color
    example["label_color"] = color_label

    #image = tf.image.grayscale_to_rgb(example["image"]) # Convert the grayscale image to RGB
    image = tf.repeat(example["image"], repeats=3, axis=-1)*255 # Repeat the channel dimension to get a 3-channel image
    image = tf.where(tf.math.equal(image, 255), color, image) # Replace the grayscale value with the chosen color
    #image = tf.repeat(example["image"], 3, axis=-1)
    #image = tf.cast(image, tf.float32)
    example["image"] = image

    return example

def change_colors(example, ind):
    color = list_of_colors[ind] # Choose a color from the list
    color_label = list_of_colors_labels[ind]

    example["value_color"] = color
    example["label_color"] = color_label

    image = tf.repeat(example["image"], repeats=3, axis=-1)*255
    image = tf.where(tf.math.equal(image, 255), color, image)
    example["image"] = image

    return example


def modify_dsprites(ds):
    # Create a list of datasets with different color for each example
    colored_datasets = [
        ds.map(lambda x: change_colors(x, ind), num_parallel_calls=tf.data.AUTOTUNE)
        for ind in range(len(list_of_colors))
    ]

    # Interleave the datasets to create a single dataset
    modified_ds = tf.data.Dataset.from_tensor_slices(colored_datasets).interleave(
        lambda ds: ds,
        cycle_length=len(list_of_colors),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return modified_ds


# List of 20 random colors to choose from
list_of_colors_labels = ['red', 'maroon', 'orange', 'brown', 'yellow', 'olive', 'lime', 'green', 'teal', 'cyan', 'blue', 'navy', 'purple', 'magenta', 'pink', 'lavender', 'mint', 'beige', 'apricot', 'grey']

list_of_colors = [[230, 25, 75], [128, 0, 0], [245, 130, 48], [170, 110, 40], [255, 225, 25], [128, 128, 0], [210, 245, 60], [60, 180, 75], [0, 128, 128], [70, 240, 240], [0, 130, 200], [0, 0, 128], [145, 30, 180], [240, 50, 230], [250, 190, 212], [220, 190, 255], [170, 255, 195], [255, 250, 200], [255, 215, 180], [128, 128, 128]]



list_of_colors_labels =  np.arange(0,len(list_of_colors))
list_of_colors = tf.convert_to_tensor(list_of_colors, dtype=tf.uint8)
list_of_colors_labels = tf.convert_to_tensor(list_of_colors_labels)


# Apply the color change function to the dataset
def change_color_func(example):
    return change_color(example)

def subtract_mean(example):
    # Calculate the mean of the dataset
    mean = tf.reduce_mean(example["image"], axis=(0,1), keepdims=True)

    # Cast the uint8 tensor to int32
    image_int = tf.cast(example["image"], tf.int32)
    mean_int = tf.cast(mean, tf.int32)

    # Subtract the mean from each element in the image
    diff = tf.subtract(image_int, mean_int)
    tf.print(diff)
    # Get a mask for pixels that are not zero or negative
    #mask = tf.math.greater_equal(diff, 0)

    # Apply the mask to the difference tensor
    #masked_diff = tf.where(mask, diff, -diff)

    # Cast the result back to uint8
    #masked_diff_uint8 = tf.cast(masked_diff, tf.uint8)

    # Update the example dictionary with the new image
    example["image"] = diff#masked_diff_uint8

    return example


def add_noise(example):
    noise = tf.random.uniform(shape=tf.shape(example['image'])[:-1], minval=1, maxval=20)
    noise = tf.repeat(tf.expand_dims(noise, axis=-1), repeats=3, axis=-1)
    image_int = tf.cast(example["image"], tf.int32)
    noise_int = tf.cast(noise, tf.int32)
    #tf.print(noise_int)
    mask = tf.math.greater_equal(image_int, 230)
    mask_line = tf.cast(tf.reduce_sum(tf.cast(mask, tf.uint8), axis=2), tf.bool)
    mask_fin = tf.repeat(tf.expand_dims(mask_line, axis=-1), repeats=3, axis=-1)
    noisy_imagep = image_int + noise_int
    noisy_imagem = image_int - noise_int
    noisy_image = tf.where(mask_fin, noisy_imagem, noisy_imagep)
    example['image'] = tf.cast(noisy_image, tf.uint8)

    return example

def add_noise_func(example):
    return add_noise(example)


def separate_attributes_to_dict(x):

    image = x['image']
    label_x = x['value_x_position']
    label_y = x['value_y_position']
    label_color = tf.cast(x['label_color'], tf.float32)
    label_scale = x['value_scale']
    label_orientation = x['value_orientation']
    other_attributes = tf.stack([label_x, label_y, label_color, label_scale, label_orientation], axis=0)
    return {'image': image, 'latents': other_attributes}

# Apply the function to the dataset

#ds = ds.map(change_color_func)
#ds = ds.map(add_noise_func)
#print(ds)
#images = np.array([example["image"].numpy() for example in ds])
#print(images.shape)
#mean_image = np.mean(images, axis=0)
#images -= mean_image
#ds = tf.data.Dataset.from_tensor_slices(images)
#ds = ds.map(lambda image: {"image": image})
#ds = subtract_mean2(ds)
#for example in ds2.take(2):
    #plt.figure()
    #plt.imshow(example["image"])
    #plt.show()
    #print(example["latents"])
    #print(image["image"])
    #print(np.mean(image["image"]))
    #print(tf.reduce_mean(image["image"],axis=(0,1)))
    #print(tf.math.reduce_std(image["image"]))
    #plt.figure()
    #plt.imshow(example['image'])
    #print(example['image'])
    #plt.show()
'''


#lista = []
for example in ds.take(1):
#    if example["value_orientation"] not in lista:
#        print(example["value_orientation"])
#        lista.append(example["value_orientation"])
    plt.figure()
    plt.imshow(example['image'])
    print(example['image'])
    plt.show()



import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# Load the dsprites dataset
ds, ds_info = tfds.load("dsprites", split="train", with_info=True)

# Create a function to apply rotation and color change to an example
def change_color(example):
    #Function to change the color of the image
    #:param example, with white objects:
    #:return example that can be one from 18 colors defined in list_of_colors:

    ind = tf.random.uniform(shape=(), minval=0, maxval=len(list_of_colors), dtype=tf.int32)
    color = list_of_colors[ind] # Randomly choose a color from the list
    color_label = list_of_colors_labels[ind]
    example["value_color"] = color
    example["label_color"] = color_label

    image = tf.image.grayscale_to_rgb(example["image"]) # Convert the grayscale image to RGB
    image = tf.where(tf.math.equal(image, 1), color, image) # Replace the grayscale value with the chosen color
    example["image"] = image

    return example

# List of 18 random colors to choose from
list_of_colors = [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180],
                  [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212], [0, 128, 128], [220, 190, 255],
                  [170, 110, 40], [255, 250, 200], [128, 0, 0], [170, 255, 195], [128, 128, 0], [255, 215, 180],
                  [0, 0, 128], [128, 128, 128]]
list_of_colors_labels =  np.arange(0,len(list_of_colors))
list_of_colors = tf.convert_to_tensor(list_of_colors, dtype=tf.uint8)
list_of_colors_labels = tf.convert_to_tensor(list_of_colors_labels)




# Apply the change_color function to the dataset
ds = ds.map(change_color)
'''
