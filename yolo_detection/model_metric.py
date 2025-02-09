
# **模型参数**
_MODEL_SIZE = (416, 416)
batch_size = len(image_names)  # 计算 batch_size
batch = load_images(image_names, model_size=_MODEL_SIZE)  # 读取图片并预处理

# **加载 COCO 类别**
def load_class_names(class_file_path):
    with open(class_file_path, 'r') as f:
        class_names = f.read().strip().split('\n')
    return class_names

class_names = load_class_names(r'C:\Users\wming\OneDrive\Desktop\kaggle\yolo_detection\coco.names')
n_classes = len(class_names)

# **YOLO 模型参数**
max_output_size = 10
iou_threshold = 0.4
confidence_threshold = 0.6

# **加载 Darknet 预训练权重**
def load_darknet_weights(model, weights_path):
    with open(weights_path, 'rb') as f:
        _ = np.fromfile(f, dtype=np.int32, count=5)  # 读取文件头信息（5个 int）
        
        for layer in model.layers:
            if not layer.weights:
                continue  # 跳过无权重的层

            weights = []
            for w in layer.weights:
                shape = w.shape.as_list()
                size = np.prod(shape)
                weights.append(np.fromfile(f, dtype=np.float32, count=size).reshape(shape))

            # **使用 `tf.Variable.assign()` 代替 `set_weights()`**
            for var, w in zip(layer.weights, weights):
                var.assign(w)  # ✅ 兼容 TensorFlow 2.x

# **初始化 YOLOv3 模型**
model = Yolo_v3(n_classes=n_classes, model_size=_MODEL_SIZE,
                max_output_size=max_output_size,
                iou_threshold=iou_threshold,
                confidence_threshold=confidence_threshold)

# **加载预训练权重**
load_darknet_weights(model, 'yolov3.weights')

# **构建模型（正确的输入维度）**
model.build(input_shape=(None, 416, 416, 3))  # 修正 input_shape

# **测试模型（确保可以运行）**
dummy_input = tf.random.normal((1, 416, 416, 3))  # 生成随机输入
_ = model(dummy_input, training=False)  # 运行模型一次以初始化权重

# **保存 & 加载权重**
model.save_weights('yolov3.weights.h5')  # Keras 3 需要 `.weights.h5`
model.load_weights('yolov3.weights.h5')

# **运行检测**
detection_result = model(batch, training=False)

# **可视化检测结果**
draw_boxes(image_names, detection_result, class_names, _MODEL_SIZE)

