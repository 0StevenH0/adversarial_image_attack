import tensorflow as tf

def uniform_noise(images, low=-.005, high=.005, out=None):

    if tf.rank(images)==4:
        batch_size, height, width, channels = images.shape
        noise = tf.random.uniform(shape=(batch_size,height, width, channels), minval=low, maxval=high, dtype=tf.float32)

    elif tf.rank(images)==3:
        height, width, channels = images.shape
        noise = tf.random.uniform(shape=(height, width, channels), minval=low, maxval=high, dtype=tf.float32)

    if out == "img":
        # Add noise to the images and clip the values
        noisy_images = images + noise
        noisy_images = tf.clip_by_value(noisy_images, 0, 1)
        return noisy_images

    return noise


def gaussian_noise(images, mean=1, std=0.1, out=None):

    if tf.rank(images)==4:
        batch_size, height, width, channels = images.shape
        noise = tf.random.normal(shape=(batch_size, height, width, channels), mean=mean, stddev=std, dtype=tf.float32)

    elif tf.rank(images)==3:
        height, width, channels = images.shape
        noise = tf.random.normal(shape=(height, width, channels), mean=mean, stddev=std, dtype=tf.float32)

    if out == "img":
        # Add noise to the images and clip the values
        noisy_images = images + noise
        noisy_images = tf.clip_by_value(noisy_images, 0, 1)
        return noisy_images

    return noise



def fgsm_noise(model, images, true, epsilon=0.007, out=None):
    # Calculate gradients of the loss with respect to the input image
    with tf.GradientTape() as tape:
        tape.watch(images)
        prediction = model(images)
        loss = tf.keras.losses.sparse_categorical_crossentropy(true, prediction)

    gradient = tape.gradient(loss, images)

    noise = epsilon * tf.sign(gradient)

    if out == "img":
            images = images + noise
            images = tf.clip_by_value(images, 0, 1)
            return images
    return noise


# not tested
def pgd_noise(model, image, epsilon, num_iterations, step_size, out=None):
    adv_image = tf.identity(image)

    for _ in range(num_iterations):
        with tf.GradientTape() as tape:
            tape.watch(adv_image)
            prediction = model(adv_image)
            loss = tf.keras.losses.categorical_crossentropy(tf.one_hot([target_class], 10), prediction)

        gradient = tape.gradient(loss, adv_image)

        adv_image = adv_image + step_size * tf.sign(gradient)

        noise = adv_image - image
        noise = tf.clip_by_value(noise, -epsilon, epsilon)
        if out == "img":
            image = image + noise
            image = tf.clip_by_value(image, 0, 1)
    return noise

