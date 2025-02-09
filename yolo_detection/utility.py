# Utility Functions

# 加载图片并调整大小
def load_images(img_names, model_size):
    imgs = []

    for img_name in img_names:
        img = Image.open(img_name).convert("RGB")  # 确保图片为 RGB 格式
        img = img.resize(size=model_size)
        img = np.array(img, dtype=np.float32) / 255.0  # 归一化 [0,1]
        img = np.expand_dims(img, axis=0)
        imgs.append(img)

    imgs = np.concatenate(imgs, axis=0)  # 确保批量维度正确

    return imgs 

# 加载类别名称
def load_class_name(file_name):
    with open(file_name, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

# 画出目标检测的边界框
def draw_boxes(img_names, boxes_dicts, class_names, model_size):
    colors = (np.array(color_palette('hls', len(class_names))) * 255).astype(np.uint8)

    for img_name, boxes_dict in zip(img_names, boxes_dicts):
        img = Image.open(img_name).convert("RGB")
        draw = ImageDraw.Draw(img)

        # 计算缩放比例
        resize_factor = (img.size[0] / model_size[0], img.size[1] / model_size[1])

        for cls, boxes in boxes_dict.items():
            for box in boxes:
                x_center, y_center, width, height, confidence = box[:5]

                # **确保类别索引 `cls` 在 `class_names` 里**
                class_id = int(cls)  # ✅ 确保 `cls` 是整数索引
                if class_id >= len(class_names):
                    continue  # **防止 `class_id` 超出 `class_names`**

                # **转换 YOLO 坐标到 `[x_min, y_min, x_max, y_max]`**
                x_min = (x_center - width / 2) * resize_factor[0]
                y_min = (y_center - height / 2) * resize_factor[1]
                x_max = (x_center + width / 2) * resize_factor[0]
                y_max = (y_center + height / 2) * resize_factor[1]

                # **确保 x_max > x_min, y_max > y_min**
                x_min, x_max = sorted([x_min, x_max])
                y_min, y_max = sorted([y_min, y_max])

                # **绘制边框**
                draw.rectangle([x_min, y_min, x_max, y_max], outline=tuple(colors[class_id]))

                # **绘制标签文本**
                text = "{} {:.1f}%".format(class_names[class_id], confidence * 100)  # ✅ 修正 `text`

                # **计算文本大小**
                text_size = draw.textbbox((0, 0), text)  # ✅ 修正 `draw.textsize()` 的计算方式
                text_width, text_height = text_size[2] - text_size[0], text_size[3] - text_size[1]

                # **绘制文本背景**
                draw.rectangle([x_min, y_min - text_height, x_min + text_width, y_min], fill=tuple(colors[class_id]))
                
                # **绘制文本**
                draw.text((x_min, y_min - text_height), text, fill="black")

        display(img)







# Converting weights to Tensorflow format




def load_weights(variables, file_name):
    """
    Args:
        variables: A list of tf.Variable to be assigned
        file_name: A name of a file containing weights.
    Returns:
        A list of assign operations.
    """
    with open(file_name, 'rb') as f:
        # 跳过文件头部（5个 int）
        np.fromfile(f, dtype=np.int32, count=5)
        weights = np.fromfile(f, dtype=np.float32)

        ptr = 0  # 权重指针
        assign_ops = []  # 存储 assign 操作

        # **Darknet53 部分**
        for i in range(52):
            conv_var = variables[5 * i]
            gamma, beta, mean, variance = variables[5 * i + 1: 5 * i + 5]
            batch_norm_vars = [beta, gamma, mean, variance]

            for var in batch_norm_vars:
                shape = var.shape.as_list()
                num_params = np.prod(shape)
                var_weights = weights[ptr: ptr + num_params].reshape(shape)
                ptr += num_params
                assign_ops.append(var.assign(var_weights))  # ✅ 使用 `assign()`

            shape = conv_var.shape.as_list()
            num_params = np.prod(shape)
            var_weights = weights[ptr: ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(conv_var.assign(var_weights))  # ✅ 使用 `assign()`

        # **YOLO 头部部分**
        ranges = [range(8), range(8, 16), range(16, 23)]
        unnormalized = [81, 93, 105]  # **YOLO 头部无 BatchNorm 层的索引**

        for j in range(3):
            for i in ranges[j]:
                current = 52 * 5 + 5 * i + j * 2
                conv_var = variables[current]
                gamma, beta, mean, variance = variables[current + 1: current + 5]
                batch_norm_vars = [beta, gamma, mean, variance]

                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr: ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(var.assign(var_weights))  # ✅ 使用 `assign()`

                shape = conv_var.shape.as_list()
                num_params = np.prod(shape)
                var_weights = weights[ptr: ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
                var_weights = np.transpose(var_weights, (2, 3, 1, 0))
                ptr += num_params
                assign_ops.append(conv_var.assign(var_weights))  # ✅ 使用 `assign()`

            # **没有 BatchNorm 的 YOLO 头部（有 Bias）**
            bias_var = variables[52 * 5 + unnormalized[j] * 5 + j * 2 + 1]
            shape = bias_var.shape.as_list()
            num_params = np.prod(shape)
            var_weights = weights[ptr: ptr + num_params].reshape(shape)
            ptr += num_params
            assign_ops.append(bias_var.assign(var_weights))  # ✅ 使用 `assign()`

    return assign_ops


#Loading images
image_names = [r'C:\Users\wming\OneDrive\Desktop\kaggle\yolo_detection\dog.jpg']
for img_path in image_names:
    if os.path.exists(img_path):  # 确保文件存在
        try:
            img = Image.open(img_path).convert("RGB")  # 确保为 RGB 格式
            display(img)  # 在 Jupyter Notebook 里显示图片
        except Exception as e:
            print(f"无法加载图片: {img_path}\n错误: {e}")
    else:
        print(f"图片文件不存在: {img_path}")