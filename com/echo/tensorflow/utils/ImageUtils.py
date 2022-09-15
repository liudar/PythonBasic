import tensorflow as tf

def read(image_path,height=0, width=0):
    img = tf.io.read_file(image_path)
    # 解码图片
    img = tf.image.decode_png(img,channels=3) # RGBA,PNG
    # img = tf.image.decode_jpeg(img, channels=3)  # RGBA,jpg
    if height != 0 and width != 0:
        img = tf.image.resize(img, [height, width])
    img = tf.cast(img, dtype=tf.uint8)
    return img

def save(img, image_path):
    img = tf.image.encode_png(img)  # jpeg
    # 保存图片
    with tf.io.gfile.GFile(image_path, 'wb') as file:
        file.write(img.numpy())

if __name__ == '__main__':
    image = read(r"C:\Users\echo\Desktop\aaa.png",100,200)
    save(image,"./a.png")

