import tensorflow as tf

def read(image_path,height=0, width=0, image_type='png'):
    img = tf.io.read_file(image_path)
    # 解码图片
    if(image_type == 'png' or image_type == 'rgba'):
        img = tf.image.decode_png(img,channels=3) # RGBA,PNG
    elif image_type == 'jpg':
        img = tf.image.decode_jpeg(img, channels=3)  # RGBA,jpg
    if height != 0 and width != 0:
        img = tf.image.resize(img, [height, width])
    img = tf.cast(img, dtype=tf.uint8)
    return img

def resize(image_path, image_size, resize, image_type='png'):
    """
    图片预处理 1. 取中心部分 2. 缩放大小
    """
    img = tf.io.read_file(image_path)

    # 解码图片
    if(image_type == 'png' or image_type == 'rgba'):
        img = tf.image.decode_png(img,channels=3) # RGBA,PNG
    elif image_type == 'jpg':
        img = tf.image.decode_jpeg(img, channels=3)  # RGBA,jpg

    # img_data = tf.image.convert_image_dtype(img, dtype=tf.float32)
    # 此方法会填充和裁剪图片

    img = tf.image.resize_with_crop_or_pad(img, image_size, image_size)
    img = tf.image.resize(img, [resize, resize])
    img = tf.cast(img, dtype=tf.uint8)
    return img

def save(img, image_path):
    img = tf.image.encode_png(img)  # jpeg
    # 保存图片
    with tf.io.gfile.GFile(image_path, 'wb') as file:
        file.write(img.numpy())



if __name__ == '__main__':
    image = resize(r"C:\Users\echo\Desktop\bbb.png", 1080, 720)
    save(image,"./bb.png")

