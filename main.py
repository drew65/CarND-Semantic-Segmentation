import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

data_dir = '/home/drew/carND3/CarND-Semantic-Segmentation/data/'
helper.maybe_download_pretrained_vgg(data_dir)

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    w3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    w4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    w7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return w1, keep, w3, w4, w7
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # Final  Encoder Layer ; 1x1 convolutional layer taking the output from final convulutional layer of VGG,
    # The dense (fully connected) layer from the original VGG is replaced by 1-by-1 convolution to preserve the spatial information.
    conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), name='conv_1x1')

    # The decoder part of the network upsamples the input into the original image size.
    # The output will be a 4-dimensional tensor: batch size, original image size (height/width) and the number of classes.
    # 1st Decoder Layer ; Takes input from the conv_1x1 layer
    deconv1_output = tf.layers.conv2d_transpose(conv_1x1, num_classes, 4, 2, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), name='deconv1')

    deconv1_output = tf.nn.elu(deconv1_output)

    pool_4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),\
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),name='pool_4')

    deconv1_output = tf.add(deconv1_output, pool_4)

    # 2nd decoder layer;
    deconv2_output = tf.layers.conv2d_transpose(deconv1_output, num_classes, 4, 2, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), name='deconv2')

    deconv2_output = tf.nn.elu(deconv2_output)

    pool_3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),\
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),name='pool_3')

    deconv2_output = tf.add(deconv2_output, pool_3)

    # 3rd decoder layer;
    deconv3_output = tf.layers.conv2d_transpose(deconv2_output, num_classes, 16, 8, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), name='deconv3')

    deconv3_output = tf.nn.elu(deconv3_output)


    return deconv3_output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    training_operation = optimizer.minimize(cross_entropy_loss)

    return logits, training_operation, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    sess.run(tf.global_variables_initializer())
    #optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    #training_operation = optimizer.minimize(loss_operation)
    Learning_rate = 1e-4
    DROPOUT = 0.5
    print("hello1")
    for epoch in range(epochs):
      for batch, (image, label) in enumerate(get_batches_fn(batch_size)):
        feed_dict = {
          input_image: image,
          correct_label: label,
          keep_prob: DROPOUT,
          learning_rate: Learning_rate
        }

        _ , loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)
        print("EPOCH: {:5} | BATCH: {:5} | LOSS: {:10.5}".format(epoch, batch, loss))
    #pass
    #sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
print("hello2")
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    learning_rate = 1e-4
    epochs = 20
    batch_size = 5
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    print("hello3")
    #config = tf.ConfigProto()
    #config.gpu_options.allocator_type = 'BFC'
    #with tf.Session(config = config) as s:

    gpu_options = tf.GPUOptions(allow_growth=True)
    #session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    #with tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    #with session as sess:
        # Path to vgg model
        print("hello4")
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        print("hello5")
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        #setup placeholder tensors
        print("hello6")
        correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
        learning_rate = tf.placeholder(tf.float32)
        print("hello7")
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        print("hello8")
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        print("hello9")
        logits, training_operation, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)
        print("hello10")


        # TODO: Train NN using the train_nn function
        #sess.run(tf.global_variables_initializer())
        print("hello11")
        train_nn(sess, epochs, batch_size, get_batches_fn, training_operation, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)

        print("thats all folks")

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
