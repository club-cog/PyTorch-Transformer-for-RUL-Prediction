import tensorflow as tf
tf.random.set_seed(1)
from model import *
from loading_data import *
from testing import *
from visualize import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse




@tf.function
def train_step(X, y, t, num_steps):
    with tf.GradientTape() as tape:
        # forward pass
        outputs = model(X, t, training=True)

        # obtain the loss function
        loss = criterion(y, outputs)

        loss_normalized = loss / num_steps  # normalize the loss

    # backward pass
    grads = tape.gradient(loss_normalized, model.trainable_variables)
    return loss, grads


def training():
    for epoch in range(num_epochs):  # iteration of epoch
        i = 1
        epoch_loss = 0

        while i <= 100:  # iteration of unit

            # fetch the data of unit i
            x = group.get_group(i).to_numpy()
            total_loss = 0
            accum_grads = [tf.zeros_like(v) for v in model.trainable_variables]

            for t in range(x.shape[0] - 1):
                if t == 0:  # skip the first and last for convolution without padding
                    continue
                else:
                    X = x[t - 1:t + 2, 2:-1]  # fetch the 3 * 14 feature as input

                y = x[t, -1:]  # fetch the corresponding target rul as label

                X_train_tensors = tf.convert_to_tensor(X, dtype=tf.float32)
                y_train_tensors = tf.convert_to_tensor(y, dtype=tf.float32)

                X_train_tensors_final = tf.reshape(
                    X_train_tensors, (1, 1, X_train_tensors.shape[0], X_train_tensors.shape[1]))

                # forward and backward pass, loss normalized by the number of steps in the unit
                loss, grads = train_step(X_train_tensors_final, y_train_tensors, tf.constant(t),
                                         tf.constant(x.shape[0] - 2, dtype=tf.float32))

                # summarize the loss
                total_loss += float(loss)

                # accumulate gradients over the unit
                accum_grads = [a + g for a, g in zip(accum_grads, grads)]

            # only update after finishing one unit
            optim.apply_gradients(zip(accum_grads, model.trainable_variables))

            i += 1
            epoch_loss += total_loss / x.shape[0]

        # evaluate model
        rmse, result = testing(group_test, y_test, model)

        print("Epoch: %d, training loss: %1.5f, testing rmse: %1.5f" % (epoch, epoch_loss / 100, rmse))

    return result, rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FD001', help='which dataset to run')
    opt = parser.parse_args()
    
    num_epochs = 4 # Number of training epochs
    d_model = 128  # dimension in encoder
    heads = 4  # number of heads in multi-head attention
    N = 2  # number of encoder layers
    m = 14  # number of features


    
    if opt.dataset == 'FD001':
        # loading training and testing sets
        group, y_test, group_test = loading_FD001()
        
        #setting the drooput rate
        dropout = 0.1
        
        # define and load model
        model = Transformer(m, d_model, N, heads, dropout)
        model(tf.zeros((1, 1, 3, m)), 0, training=False)  # build the weights

        # initialization, equivalent to nn.init.xavier_uniform_ on every parameter with dim > 1
        for p in model.trainable_variables:
            if len(p.shape) > 1:
                p.assign(tf.keras.initializers.GlorotUniform()(p.shape, dtype=p.dtype))
                
        # initialize Adam optimizer (epsilon matches torch.optim.Adam's default)
        optim = tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-8)

        # mean-squared error for regression
        criterion = tf.keras.losses.MeanSquaredError()

        # training with evaluation
        result, rmse = training()

        # testing already done in training() for each epoch to see live testing rmse, or
        # can be done once after finish training
        # rmse, result = testing(group_test, y_test, model)

        # visualize the testing result
        visualize(result, rmse)
    else:
        print('Either dataset not implemented or not defined')

