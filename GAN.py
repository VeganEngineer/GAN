import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


#!wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
#!wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
#!wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
#!wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz

#mnist = tf.keras.datasets.mnist

import input_data


mnist = input_data.read_data_sets("MNIST_Fashion/")


#Training parameters
learning_rate = 0.0002
batch_size = 128
epochs = 100000

#Network parameters
image_dim = 784      #image size is 28 x 28
gen_hidd_dim = 256
disc_hidd_dim = 256
z_noise_dim = 100      # input noise datapoint

def xavier_init(shape):
    return tf.random_normal(shape=shape,stddev=1./tf.sqrt(shape[0]/2.0))


# Define weights and bias dictionaries

weights = {"disc_H": tf.Variable(xavier_init([image_dim, disc_hidd_dim])),
           "disc_final": tf.Variable(xavier_init([disc_hidd_dim, 1])),
           "gen_H": tf.Variable(xavier_init([z_noise_dim, gen_hidd_dim])),
           "gen_final": tf.Variable(xavier_init([gen_hidd_dim, image_dim]))
           }

bias = {"disc_H": tf.Variable(xavier_init([disc_hidd_dim])),
        "disc_final": tf.Variable(xavier_init([1])),
        "gen_H": tf.Variable(xavier_init([gen_hidd_dim])),
        "gen_final": tf.Variable(xavier_init([image_dim]))
        }

# Creating the Computational Graph
# Define discriminator function

def Discriminator(x):
    hidden_layer = tf.nn.relu(tf.add(tf.matmul(x,weights["disc_H"]),bias["disc_H"]))
    final_layer= tf.add(tf.matmul(x,weights["disc_H"]),bias["disc_final"])
    disc_output = tf.nn.sigmoid(final_layer)
    return final_layer,disc_output

# Define generator network
def Generator(x):
    hidden_layer = tf.nn.relu(tf.add(tf.matmul(x, weights["gen_H"]), bias["gen_H"]))
    final_layer = tf.add(tf.matmul(x, weights["gen_final"]), bias["gen_final"])
    gen_output = tf.nn.sigmoid(final_layer)
    return gen_output

#Define the placeholders for external input

z_input = tf.placeholder(tf.float32, shape = [None,z_noise_dim], name="input_noise")
x_input = tf.placeholder(tf.float32, shape = [None, image_dim], name="real_input")

# Building the Generator Network
with tf.name_scope("Generator") as scope:
    output_Gen = Generator(z_input)

# Building the Discriminator Network
with tf.name_scope("Discriminator") as scope:
    real_output1_Disc = real_output_Disc = Discriminator(x_input)     #Implements D(x)
    fake_output1_Disc = fake_output_Disc = Discriminator(output_Gen)  #Implements D(G(x))



#First kind of loss
with tf.name_scope("Discriminator_Loss") as scope:
    Discriminator_Loss = -tf.reduce_mean(tf.log(real_output_Disc + 0.0001) + tf.log(1. - fake_output_Disc + 0.0001))

with tf.name_scope("Generator_Loss") as scope:
    Generator_Loss = -tf.reduce_mean(tf.log(fake_output_Disc + 0.0001)) #due to max log(D(G(z))

#TensorBoard Summary
Disc_loss_total = tf.summary.scalar("Disc_Total_loss",Discriminator_Loss)
Gen_loss_total = tf.summary.scalar("Gen_loss", Generator_Loss)

#Second type of loss
with tf.name_scope("Discriminator_Loss") as scope:
    Disc_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_output1_Disc,labels=tf.ones_like(real_output1_Disc)))
    Disc_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_output1_Disc,labels=tf.zeros_like(fake_output1_Disc)))
    Discriminator_Loss = Disc_real_loss + Disc_fake_loss


with tf.name_scope("Generator_Loss") as scope:
    Generator_Loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_output1_Disc,labels=tf.zeros_like(fake_output1_Disc)))

#TensorBoard Summary
Disc_loss_real_summary = tf.summary.scalar("Disc_loss_real",Disc_real_loss)
Disc_loss_fake_summary = tf.summary.scalar("Disc_loss_fake", Disc_fake_loss)
Disc_loss_summary = tf.summary.scalar("Disc_Total_real", Discriminator_loss)

Disc_loss_total = tf.summary.merge([Disc_loss_real_summary,Disc_loss_fake_summary,Disc_loss_summary])
Gen_loss_total = tf.summary.scalar("Gen_Loss",Generator_Loss)

#Define the variables

Generator_var = [weights["gen_H"],weights["gen_final"],bias["gen_H"],bias["gen_final"]]
Discriminator_var = [weights["disc_H"],weights["disc_final"],bias["disc_H"],bias["disc_final"]]

#Define the optimizer

with tf.name_scope("Optimizer_Discriminator") as scope:
    Discriminator_optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(Generator_Loss,var_list = Generator_var)


#Initialize the variables
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
writer = tf.summary.FileWriter("./log",sess.graph)

for epoch in range(epochs):
    X_batch, _ = mnist.train.next_batch(batch_size)

    #Generate noise to feed the discriminator
    Z_noise = np.random.uniform(-1.,1.,size=[batch_size,z_noise_dim])
    _, Disc_loss_epoch = sess.run([Discriminator_optimize,Discriminator_Loss],feed_dict = {X_input:X_batch,Z_input:Z_noise})
    _, Gen_loss_epoch = sess.run([Discriminator_optimize, Discriminator_Loss],
                                  feed_dict={X_input: X_batch, Z_input: Z_noise})

    #Running the Discriminator summary
    summary_Disc_Loss = sess.run(Disc_loss_total, feed_dict={X_input: X_batch, Z_input: Z_noise})

    #Adding the Discriminator summary
    writer.add_summary(summary_Disc_Loss, epoch)

    #Running the Generator summary
    summary_Gen_Loss = sess.run(Gen_loss_total, feed_dict = {Z_input:Z_noise})

    #Adding the Generator summary
    writer.add_summary(summary_Gen_Loss,epoch)

    if epoch % 2000 == 0:
        print("Steps:{0} : Generator Loss :{1}, Discriminator Loss: {2}".format(epoch,Gen_loss_epoch,Disc_loss_epoch))

#Testing
#Generate images from noise, using the generator network.
n = 6
canvas = np.empty((28*n,28*n))

for i in range(n):
    #Noise input
    Z_noise = np.random.uniform(-1.,1.,size=[batch_size,z_noise_dim])
    #Generate image from noise
    g = sess.run(output_Gen,feed_dict={Z_input:Z_noise})
    #Reverse colours for better display
    g = -1 * (g-1)

    for j in range(n):
        # Draw the generated digits
        canvas[i * 28:(i+1)*28, j*28:(j+1)*28] = g[j].reshape([28,28])


plt.figure(figsize=(n,n))
plt.imshow(canvas, origin="upper", cmap="gray")
plt.show()