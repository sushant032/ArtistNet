import os
import sys
import image
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from utils import *
from functions import *
import numpy as np
import tensorflow as tf
from tkinter import *
# %matplotlib inline
print('Loading VGG-19 model')
model = load_vgg_model("imagenet-vgg-verydeep-19.mat")
name_of_image = input("Enter the name of the image with extension: ")
name_of_filter = input("Enter the name of the filter with extension: ")
content_image = load_and_resize_image("images/"+name_of_image)
type(content_image)

generated_image = generate_noise_image(content_image)
imshow(generated_image[0])

STYLE_LAYERS = [
    ('conv1_1', 0.5),
    ('conv2_1', 0.5),
    ('conv3_1', 0.5),
    ('conv4_1', 0.5),
    ('conv5_1', 0.5)]

tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(3)
    J_content = np.random.randn()    
    J_style = np.random.randn()
    J = total_cost(J_content, J_style)
    print("J = " + str(J))
    
# Reset the graph
tf.reset_default_graph()

# Start interactive session
sess = tf.InteractiveSession()

# content_image = scipy.misc.imread("images/test.jpg")
content_image = load_and_resize_image("images/"+name_of_image)
# content_image = scipy.misc.imread(content_image)
type(content_image)
content_image = reshape_and_normalize_image(content_image)
content_image.shape

style_image =load_and_resize_image("images/"+name_of_filter)
style_image.shape
imshow(style_image)
style_image = reshape_and_normalize_image(style_image)
style_image.shape
type(style_image)

model = load_vgg_model("imagenet-vgg-verydeep-19.mat")
sess.run(model['input'].assign(content_image))


out = model['conv4_2']

a_C = sess.run(out)
a_G = out


J_content = compute_content_cost(a_C, a_G)


sess.run(model['input'].assign(style_image))

J_style = compute_style_cost(model, STYLE_LAYERS,sess)

J = total_cost(J_content, J_style,  alpha = 10, beta = 40)

optimizer = tf.train.AdamOptimizer(2.0)

train_step = optimizer.minimize(J)

model_nn(sess, generated_image,model,J,J_content,J_style,train_step)

root = Tk()
root.title('Generated image')
canv = Canvas(root, width=300, height=400, bg='white')
canv.grid(row=2, column=3)

img = PhotoImage(file="D:\\DL Notes\\Project\\output\\generated_image.png")
canv.create_image(0,0, anchor=NW, image=img)

mainloop()