"""
Title: Probabilistic Bayesian Neural Networks
Author: [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)
Date created: 2021/01/15
Last modified: 2021/01/15
Description: Building probabilistic Bayesian neural network models with TensorFlow Probability.
Code modified by Andrew Polar: 2022/09/28, the dataset is replaced by generated in get_output, 
the level of stochasticity is added by noise value, when 0.0 the model becomes deterministic.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_probability as tfp
import math

FEATURE_NAMES = ["x0", "x1", "x2", "x3", "x4"]
dataset_size = 10100
train_size = 10000
hidden_units = [8, 8]
learning_rate = 0.001
batch_size = 256
num_epochs = 1000
noise = 0.8

def pearson_correlation(x, y, N):
    xy = 0.0
    x2 = 0.0
    y2 = 0.0
    for i in range(N):
        xy += x[i] * y[i]
        x2 += x[i] * x[i]
        y2 += y[i] * y[i]
    xy /= N
    x2 /= N
    y2 /= N
    xav = 0.0
    for i in range(N):
        xav += x[i]
    xav /= N
    yav = 0.0
    for i in range(N):
        yav += y[i]
    yav /= N
    ro = xy - xav * yav
    if (ro < 0.0): 
        ro = -ro
    ro /= math.sqrt(x2 - xav * xav + 0.00001)
    ro /= math.sqrt(y2 - yav * yav + 0.00001)
    return ro

def get_output(z0, z1, z2, z3, z4, noise):
    z0 += noise * (np.random.rand(1) - 0.5)
    z1 += noise * (np.random.rand(1) - 0.5)
    z2 += noise * (np.random.rand(1) - 0.5)
    z3 += noise * (np.random.rand(1) - 0.5)
    z4 += noise * (np.random.rand(1) - 0.5)
    pi = 3.14159265359
    p = 1.0 / pi
    p *= 2.0 + 2.0 * z2
    p *= 1.0 / 3.0
    p *= math.atan(20.0 * math.exp(z4) * (z0 - 0.5 + z1 / 6.0)) + pi / 2.0
    q = 1.0 / pi
    q *= 2.0 + 2.0 * z3
    q *= 1.0 / 3.0
    q *= math.atan(20.0 * math.exp(z4) * (z0 - 0.5 - z1 / 6.0)) + pi / 2.0
    return p + q

def get_Sample(z0, z1, z2, z3, z4, N):
    sample = np.empty(N, dtype=float)
    for j in range(N):
        sample[j] = get_output(z0, z1, z2, z3, z4, noise)
    return sample

def get_MonteCarlo(examples, validation_size):
    x0 = examples['x0'].numpy()
    x1 = examples['x1'].numpy()
    x2 = examples['x2'].numpy()
    x3 = examples['x3'].numpy()
    x4 = examples['x4'].numpy()
    mean = np.empty(validation_size, dtype=float)
    stdv = np.empty(validation_size, dtype=float)
    sample_size = 32
    for i in range(validation_size):
        sample = get_Sample(x0[i], x1[i], x2[i], x3[i], x4[i], sample_size)
        mean[i] = sample.mean()
        stdv[i] = 0.0
        for j in range(sample_size):
            stdv[i] += (sample[j] - mean[i]) * (sample[j] - mean[i])
        stdv[i] /= (sample_size - 1)
        stdv[i] = math.sqrt(stdv[i])
    return mean, stdv

def get_train_and_test_splits(dataset_size, train_size, batch_size=1):
    x0 = np.random.rand(dataset_size,) 
    x1 = np.random.rand(dataset_size,)
    x2 = np.random.rand(dataset_size,)
    x3 = np.random.rand(dataset_size,)
    x4 = np.random.rand(dataset_size,)
    y = np.empty(dataset_size, dtype=float)

    for idx in range(dataset_size):
        y[idx] = get_output(x0[idx], x1[idx], x2[idx], x3[idx], x4[idx], noise)

    dataset = tf.data.Dataset.from_tensor_slices(({'x0':x0, 'x1':x1, 'x2':x2, 'x3':x3, 'x4':x4}, y))

    train_dataset = (
        dataset.take(train_size).shuffle(buffer_size=train_size).batch(batch_size)
    )
    test_dataset = dataset.skip(train_size).batch(batch_size)
    return train_dataset, test_dataset    

def create_model_inputs():
    inputs = {}
    for feature_name in FEATURE_NAMES:
        inputs[feature_name] = layers.Input(
            name=feature_name, shape=(1,), dtype=tf.float32
        )
    return inputs

def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model

def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model

def create_probablistic_bnn_model(train_size):
    inputs = create_model_inputs()
    features = keras.layers.concatenate(list(inputs.values()))
    features = layers.BatchNormalization()(features)
    for units in hidden_units:
        features = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1 / train_size,
            activation="sigmoid",
        )(features)
    distribution_params = layers.Dense(units=2)(features)
    outputs = tfp.layers.IndependentNormal(1)(distribution_params)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def run_experiment(model, loss, train_dataset, test_dataset):
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss=loss,
        metrics=[keras.metrics.RootMeanSquaredError()],
    )
    print("Start training the model...")
    model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)
    print("Model training finished.")
    _, rmse = model.evaluate(train_dataset, verbose=0)
    print(f"Train RMSE: {round(rmse, 3)}")

    print("Evaluating model performance...")
    _, rmse = model.evaluate(test_dataset, verbose=0)
    print(f"Test RMSE: {round(rmse, 3)}")

def negative_loglikelihood(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets)
 
#########################################################
# code execution
#########################################################

validation_size = dataset_size - train_size
train_dataset, test_dataset = get_train_and_test_splits(dataset_size, train_size, batch_size)
examples, targets = list(test_dataset.unbatch().shuffle(batch_size * 10).batch(validation_size))[0]
mean_monte_carlo, stdv_monte_carlo = get_MonteCarlo(examples, validation_size)
prob_bnn_model = create_probablistic_bnn_model(train_size)
run_experiment(prob_bnn_model, negative_loglikelihood, train_dataset, test_dataset)
prediction_distribution = prob_bnn_model(examples)
prediction_mean = prediction_distribution.mean().numpy().tolist()
prediction_stdv = prediction_distribution.stddev().numpy()

# The 95% CI is computed as mean ± (1.96 * stdv)
upper = (prediction_mean + (1.96 * prediction_stdv)).tolist()
lower = (prediction_mean - (1.96 * prediction_stdv)).tolist()
prediction_stdv = prediction_stdv.tolist()

mean = np.empty(validation_size, dtype=float)
stdv = np.empty(validation_size, dtype=float)
print("Index, prediction mean, prediction STDV, Monte Carlo mean, Monte Carlo STDV")
for idx in range(validation_size):
    print(
        f"{idx}, "
        f"{round(prediction_mean[idx][0], 4)}, "
        f"{round(prediction_stdv[idx][0], 4)}, "
        f"{round(mean_monte_carlo[idx], 4)}, "
        f"{round(stdv_monte_carlo[idx], 4)} "
    )
    mean[idx] = prediction_mean[idx][0]
    stdv[idx] = prediction_stdv[idx][0]

print("-------------------------------------------------")
print(f"Correlation for expectations = {pearson_correlation(mean, mean_monte_carlo, validation_size)}")
print(f"Correlation for standard deviations = {pearson_correlation(stdv, stdv_monte_carlo, validation_size)}")
print(f"Target limits = {min(targets)} - {max(targets)}")


