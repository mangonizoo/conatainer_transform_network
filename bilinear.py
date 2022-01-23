import tensorflow as tf


def K_meshgrid(x, y):
    return tf.meshgrid(x, y)

def K_linspace(start, stop, num):
    return tf.linspace(start, stop, num)


class BilinearInterpolation(tf.keras.layers.Layer):
    """Performs bilinear interpolation as a keras layer
    References
    ----------
    [1]  Spatial Transformer Networks, Max Jaderberg, et al.
    [2]  https://github.com/skaae/transformer_network
    [3]  https://github.com/EderSantana/seya
    """

    def __init__(self, output_size, **kwargs):
        self.output_size = output_size
        super(BilinearInterpolation, self).__init__(**kwargs)

    def get_config(self):
        return {
            'output_size': self.output_size,
        }

    def compute_output_shape(self, input_shapes):
        height, width = self.output_size
        num_channels = input_shapes[0][-1]
        return (None, height, width, num_channels)

    def call(self, tensors, mask=None):
        X, transformation = tensors
        output = self._transform(X, transformation, self.output_size)
        return output

    def _interpolate(self, image, sampled_grids, output_size):

        batch_size = tf.keras.backend.shape(image)[0]
        height = tf.keras.backend.shape(image)[1]
        width = tf.keras.backend.shape(image)[2]
        num_channels = tf.keras.backend.shape(image)[3]
        #print(sampled_grids[:, 0:1, :],sampled_grids[:, 1:2, :],sampled_grids[:, 2:3, :])
        x = tf.keras.backend.flatten(sampled_grids[:, 0:1, :]) 
        y = tf.keras.backend.flatten(sampled_grids[:, 1:2, :])
        z = tf.keras.backend.flatten(sampled_grids[:, 2:3, :])
        z = tf.add(z, 0.000001)
        
        x = tf.cast(tf.math.divide(x,z) , dtype='float32')
        y = tf.cast(tf.math.divide(y,z) , dtype='float32')
        
        
        
        x = .5 * (x + 1.0) * tf.cast(width, dtype='float32')
        y = .5 * (y + 1.0) * tf.cast(height, dtype='float32')

        x0 = tf.cast(x, 'int32')
        x1 = x0 + 1
        y0 = tf.cast(y, 'int32')
        y1 = y0 + 1

        max_x = int(tf.keras.backend.int_shape(image)[2] - 1)
        max_y = int(tf.keras.backend.int_shape(image)[1] - 1)

        x0 = tf.keras.backend.clip(x0, 0, max_x)
        x1 = tf.keras.backend.clip(x1, 0, max_x)
        y0 = tf.keras.backend.clip(y0, 0, max_y)
        y1 = tf.keras.backend.clip(y1, 0, max_y)

        pixels_batch = tf.keras.backend.arange(0, batch_size) * (height * width)
        pixels_batch = tf.expand_dims(pixels_batch, axis=-1)
        flat_output_size = output_size[0] * output_size[1]
        base = tf.keras.backend.repeat_elements(pixels_batch, flat_output_size, axis=1)
        base = tf.keras.backend.flatten(base)

        # base_y0 = base + (y0 * width)
        base_y0 = y0 * width
        base_y0 = base + base_y0
        # base_y1 = base + (y1 * width)
        base_y1 = y1 * width
        base_y1 = base_y1 + base

        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        flat_image = tf.reshape(image, shape=(-1, num_channels))
        flat_image = tf.keras.backend.cast(flat_image, dtype='float32')
        pixel_values_a = tf.keras.backend.gather(flat_image, indices_a)
        pixel_values_b = tf.keras.backend.gather(flat_image, indices_b)
        pixel_values_c = tf.keras.backend.gather(flat_image, indices_c)
        pixel_values_d = tf.keras.backend.gather(flat_image, indices_d)

        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')

        area_a = tf.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = tf.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = tf.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = tf.expand_dims(((x - x0) * (y - y0)), 1)

        values_a = area_a * pixel_values_a
        values_b = area_b * pixel_values_b
        values_c = area_c * pixel_values_c
        values_d = area_d * pixel_values_d
        return values_a + values_b + values_c + values_d

    def _make_regular_grids(self, batch_size, height, width):
        # making a single regular grid
        x_linspace = K_linspace(-1., 1., width)
        y_linspace = K_linspace(-1., 1., height)
        x_coordinates, y_coordinates = K_meshgrid(x_linspace, y_linspace)
        x_coordinates = tf.keras.backend.flatten(x_coordinates)
        y_coordinates = tf.keras.backend.flatten(y_coordinates)
        ones = tf.ones_like(x_coordinates)
        grid = tf.keras.backend.concatenate([x_coordinates, y_coordinates, ones], 0)

        # repeating grids for each batch
        grid = tf.keras.backend.flatten(grid)
        grids = tf.keras.backend.tile(grid, tf.keras.backend.stack([batch_size]))
        return tf.reshape(grids, (batch_size, 3, height * width))

    def _transform(self, X, affine_transformation, output_size):
        batch_size, num_channels = tf.keras.backend.shape(X)[0], tf.keras.backend.shape(X)[3]
        transformations = tf.reshape(affine_transformation,
                                    shape=(batch_size, 3, 3))
        # transformations = K.cast(affine_transformation[:, 0:2, :], 'float32')
        regular_grids = self._make_regular_grids(batch_size, *output_size)
        sampled_grids = tf.keras.backend.batch_dot(transformations, regular_grids)
        interpolated_image = self._interpolate(X, sampled_grids, output_size)
        new_shape = (batch_size, output_size[0], output_size[1], num_channels)
        interpolated_image = tf.reshape(interpolated_image, new_shape)
        return interpolated_image
