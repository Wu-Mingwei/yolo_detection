#Feature Extraction: Darknet-53
def darknet53_residual_block(inputs, filters, training, data_format, strides=1):
    """ Darknet53 残差块 (Residual Block) """
    
    # **保存原始输入作为 Shortcut**
    shortcut = inputs

    # **第一层 1x1 卷积**
    inputs = conv2d_fixed_padding(inputs, filters=filters, kernel_size=1, data_format=data_format)
    
    # **第二层 3x3 卷积**
    inputs = conv2d_fixed_padding(inputs, filters=filters, kernel_size=3, data_format=data_format)
    
    # **如果 `shortcut` 形状不同，则添加 1x1 卷积匹配维度**
    if shortcut.shape[-1] != inputs.shape[-1]:
        shortcut = conv2d_fixed_padding(shortcut, filters=filters, kernel_size=1, data_format=data_format)

    # **执行残差连接**
    inputs += shortcut
    return inputs 

def darknet53(inputs, training, data_format):
    inputs = conv2d_fixed_padding(
        inputs, filters= 32, kernel_size= 3,
        data_format = data_format
    )
    inputs = batch_norm (inputs, training = training, data_format= data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha= _LEAKY_RELU)
    inputs = conv2d_fixed_padding(
        inputs, filters = 64, kernel_size= 3, strides= 2, data_format= data_format
    )
    inputs = batch_norm(inputs, training = training, data_format= data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha= _LEAKY_RELU)
    inputs = darknet53_residual_block(inputs, filters = 32, training= training,data_format=data_format)
    inputs = conv2d_fixed_padding(
        inputs, filters= 128, kernel_size= 3, strides= 2, data_format= data_format
    )
    inputs = batch_norm(inputs, training= training, data_format= data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha= _LEAKY_RELU)

    for _ in range(8):
        inputs = darknet53_residual_block(
            inputs, filters = 128,
            training = training,
            data_format = data_format
        )

        route1 = inputs 
        inputs = conv2d_fixed_padding(
            inputs, filters = 512, kernel_size=3,
            strides = 2, data_format = data_format
        )
        inputs = batch_norm(inputs, training = training, data_format = data_format)
        inputs = tf.nn.leaky_relu(inputs, alpha= _LEAKY_RELU)


    for _ in range(8):
        inputs = darknet53_residual_block(
            inputs, filters = 256, training= training, data_format=data_format
        )
        route2 = inputs 
        inputs = conv2d_fixed_padding(
            inputs, filters = 1024, kernel_size= 3,
            strides = 2, data_format = data_format
        )
        inputs = batch_norm(inputs, training= training, data_format=data_format)
        inputs = tf.nn.leaky_relu(inputs, alpha= _LEAKY_RELU)

    for _ in range(4):
        inputs = darknet53_residual_block(
            inputs, filters=512,
            training = training,
            data_format = data_format
        )

    return route1, route2, inputs


# #Convolution Layers

def yolo_convolution_block(inputs, filters, training, data_format):
    """ YOLOv3 Convolution Block with Residual Connections """
    
    # 第 1 次 1x1 卷积
    inputs = conv2d_fixed_padding(inputs, filters=filters, kernel_size=1, data_format=data_format)
    
    # 第 1 次 3x3 卷积
    inputs = conv2d_fixed_padding(inputs, filters=2*filters, kernel_size=3, data_format=data_format)

    # 第 2 次 1x1 卷积
    inputs = conv2d_fixed_padding(inputs, filters=filters, kernel_size=1, data_format=data_format)

    # 第 2 次 3x3 卷积
    inputs = conv2d_fixed_padding(inputs, filters=2*filters, kernel_size=3, data_format=data_format)

    # 第 3 次 1x1 卷积
    inputs = conv2d_fixed_padding(inputs, filters=filters, kernel_size=1, data_format=data_format)

    # 这个 `route` 作为 Residual 连接
    route = inputs  

    # 最后的 3x3 卷积
    inputs = conv2d_fixed_padding(inputs, filters=2*filters, kernel_size=3, data_format=data_format)

    return route, inputs



# Detection Layers



def yolo_layer(inputs, n_classes, anchors, img_size, data_format="channels_last"):
    """ YOLO 检测层: 预测边界框、类别概率和置信度 """

    n_anchors = len(anchors)

    # 替换 TensorFlow 1.x 代码，改为 tf.keras.layers.Conv2D()
    inputs = tf.keras.layers.Conv2D(
        filters=n_anchors * (5 + n_classes),
        kernel_size=1,
        strides=1,
        use_bias=True,
        data_format=data_format
    )(inputs)

    # 获取网格大小
    shape = tf.shape(inputs)
    grid_shape = shape[2:4] if data_format == 'channels_first' else shape[1:3]

    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])

    inputs = tf.reshape(inputs, [-1, n_anchors * grid_shape[0] * grid_shape[1], 5 + n_classes])

    # 计算步长
    strides = (img_size[0] // grid_shape[0], img_size[1] // grid_shape[1])

    # 拆分预测变量
    box_centers, box_shapes, confidence, classes = tf.split(inputs, [2, 2, 1, n_classes], axis=-1)

    # 计算网格偏移
    x = tf.range(grid_shape[0], dtype=tf.float32)
    y = tf.range(grid_shape[1], dtype=tf.float32)
    x_offset, y_offset = tf.meshgrid(x, y)
    x_offset = tf.reshape(x_offset, (-1, 1))
    y_offset = tf.reshape(y_offset, (-1, 1))
    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    x_y_offset = tf.tile(x_y_offset, [1, n_anchors])
    x_y_offset = tf.reshape(x_y_offset, [1, -1, 2])

    # 计算边界框中心点
    box_centers = tf.nn.sigmoid(box_centers)
    box_centers = (box_centers + x_y_offset) * strides

    # 计算边界框尺寸
    anchors = tf.tile(anchors, [grid_shape[0] * grid_shape[1], 1])
    box_shapes = tf.exp(box_shapes) * tf.cast(anchors, tf.float32)

    # 计算置信度和类别概率
    confidence = tf.nn.sigmoid(confidence)
    classes = tf.nn.sigmoid(classes)

    # 拼接最终输出
    inputs = tf.concat([box_centers, box_shapes, confidence, classes], axis=-1)

    return inputs
