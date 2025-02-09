# Upsample layer


def upsample(inputs, out_shape, data_format="channels_last"):
    """ 使用最近邻插值进行上采样 """
    if data_format == "channels_first":
        inputs = tf.transpose(inputs, [0, 2, 3, 1])  # 转换为 channels_last
        new_height = out_shape[3]
        new_width = out_shape[2]
    else:
        new_height = out_shape[1]
        new_width = out_shape[2]

    # **修正拼写错误 & 兼容 TensorFlow 2.x**
    inputs = tf.image.resize(inputs, (new_height, new_width), method="nearest")  # ✅ 使用 `resize()`

    if data_format == "channels_first":
        inputs = tf.transpose(inputs, [0, 3, 1, 2])  # 还原为 channels_first
    
    return inputs

# Non-max suppression

# 计算边界框的左上角和右下角
def build_boxes(inputs):
    center_x, center_y, width, height, confidence, classes = tf.split(inputs, [1,1,1,1,1,-1], axis=-1)

    top_left_x = center_x - width / 2
    top_left_y = center_y - height / 2
    bottom_right_x = center_x + width / 2
    bottom_right_y = center_y + height / 2

    boxes = tf.concat(
        [top_left_x, top_left_y, bottom_right_x, bottom_right_y, confidence, classes], axis=-1
    )

    return boxes 

# 非极大值抑制 (NMS)
def non_max_suppression(inputs, n_classes, max_output_size, iou_threshold, confidence_threshold):
    batch = tf.unstack(inputs)
    boxes_dicts = []  # 存储所有图片的检测框

    for boxes in batch:
        # 过滤掉低置信度的检测框
        boxes = tf.boolean_mask(boxes, boxes[:, 4] > confidence_threshold)

        # 获取类别索引，并转换为 float32
        classes = tf.argmax(boxes[:, 5:], axis=-1)
        classes = tf.expand_dims(tf.cast(classes, tf.float32), axis=-1)

        # 拼接类别信息
        boxes = tf.concat([boxes[:, :5], classes], axis=-1)

        # 存储每个类别的检测框
        boxes_dict = {}

        for cls in range(n_classes):
            # 获取属于当前类别的框
            mask = tf.equal(boxes[:, 5], tf.cast(cls, tf.float32))  # 确保 mask 是 bool 类型
            class_boxes = tf.boolean_mask(boxes, mask)

            if tf.shape(class_boxes)[0] > 0:  # 仅在类别检测框不为空时继续
                # 拆分坐标、置信度
                boxes_coords, boxes_conf_scores, _ = tf.split(class_boxes, [4,1,-1], axis=-1)
                boxes_conf_scores = tf.reshape(boxes_conf_scores, [-1])  # 转换形状

                # 执行 NMS
                indices = tf.image.non_max_suppression(boxes_coords, boxes_conf_scores,
                                                       max_output_size, iou_threshold)

                class_boxes = tf.gather(class_boxes, indices)
                boxes_dict[cls] = class_boxes[:, :5]  # 存储 [x1, y1, x2, y2, score]

        boxes_dicts.append(boxes_dict)  # 追加到 batch 结果中

    return boxes_dicts



# Final Model class

class Yolo_v3(tf.keras.Model):
    def __init__(self, n_classes, model_size, max_output_size, iou_threshold,
                 confidence_threshold, data_format=None):
        super(Yolo_v3, self).__init__()
        
        # 设置数据格式
        if not data_format:
            data_format = 'channels_first' if tf.test.is_built_with_cuda() else 'channels_last'

        self.n_classes = n_classes
        self.model_size = model_size
        self.max_output_size = max_output_size
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.data_format = 'channels_last'

    def call(self, inputs, training=False):
        if self.data_format == 'channels_first':
            inputs = tf.transpose(inputs, [0, 3, 1, 2])  # 修正拼写错误

        inputs = inputs / 255.0  # 归一化输入

        # Darknet53 主干网络
        route1, route2, inputs = darknet53(inputs, training=training, data_format=self.data_format)

        # 第一阶段 YOLO 头部
        route, inputs = yolo_convolution_block(inputs, filters=512, training=training, data_format=self.data_format)
        detect1 = yolo_layer(inputs, n_classes=self.n_classes, anchors=_ANCHORS[6:9],
                             img_size=self.model_size, data_format=self.data_format)

        # 第二阶段 YOLO 头部
        inputs = conv2d_fixed_padding(route, filters=256, kernel_size=1, data_format=self.data_format)
        inputs = upsample(inputs, out_shape=route2.get_shape().as_list(), data_format=self.data_format)
        axis = 1 if self.data_format == 'channels_first' else 3
        inputs = tf.concat([inputs, route2], axis=axis)
        
        route, inputs = yolo_convolution_block(inputs, filters=256, training=training, data_format=self.data_format)
        detect2 = yolo_layer(inputs, n_classes=self.n_classes, anchors=_ANCHORS[3:6],
                             img_size=self.model_size, data_format=self.data_format)

        # 第三阶段 YOLO 头部
        inputs = conv2d_fixed_padding(route, filters=128, kernel_size=1, data_format=self.data_format)
        upsample_size = route1.get_shape().as_list()  # 修正拼写错误
        inputs = upsample(inputs, out_shape=upsample_size, data_format=self.data_format)
        inputs = tf.concat([inputs, route1], axis=axis)

        route, inputs = yolo_convolution_block(inputs, filters=128, training=training, data_format=self.data_format)
        detect3 = yolo_layer(inputs, n_classes=self.n_classes, anchors=_ANCHORS[0:3],
                             img_size=self.model_size, data_format=self.data_format)

        # 合并所有检测层
        inputs = tf.concat([detect1, detect2, detect3], axis=1)
        inputs = build_boxes(inputs)

        # 执行非极大值抑制 (NMS)
        boxes_dicts = non_max_suppression(inputs, n_classes=self.n_classes,
                                          max_output_size=self.max_output_size,
                                          iou_threshold=self.iou_threshold,
                                          confidence_threshold=self.confidence_threshold)
        return boxes_dicts






