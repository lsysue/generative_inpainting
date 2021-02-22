"""
@parameters:
    --image_dir: 待inpaint的图片目录
    --mask_dir: 待inpaint图对应需使用的mask目录
    --checkpoint_dir: tensorflow的checkpoint目录
    --output_dir: inpaint图片保存的路径

@use command:
    # python inpaint_dir.py --image_dir data/input/ --mask_dir data/mask/
"""

import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel


def inpaint(arg_image_dir, arg_mask_dir, arg_checkpoint_dir, arg_output_dir):
    tf.reset_default_graph()
    FLAGS = ng.Config('inpaint.yml')
    # ng.get_gpus(1)

    model = InpaintCAModel()
    for arg_image in os.listdir(arg_image_dir):
        arg_mask = arg_image    # assume the mask has the same name as the image 
        if os.path.exists(arg_output_dir + arg_image):
            print("note |", arg_image, "already inpainted.")
            continue
        if os.path.exists(arg_image_dir + arg_image) and os.path.exists(arg_mask_dir + arg_mask):
            pass
        else:
            continue

        image = cv2.imread(os.path.join(arg_image_dir, arg_image))
        mask = cv2.imread(os.path.join(arg_mask_dir, arg_mask))
        name = arg_image

        assert image.shape == mask.shape 

        h, w, _ = image.shape
        grid = 8
        image = image[:h // grid * grid, :w // grid * grid, :]
        mask = mask[:h // grid * grid, :w // grid * grid, :]
        print('Shape of image: {}'.format(image.shape))

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)

        tf.reset_default_graph()
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            input_image = tf.constant(input_image, dtype=tf.float32)
            output = model.build_server_graph(FLAGS, input_image)
            output = (output + 1.) * 127.5
            output = tf.reverse(output, [-1])
            output = tf.saturate_cast(output, tf.uint8)
            # load pretrained model
            vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            assign_ops = []
            for var in vars_list:
                vname = var.name
                from_name = vname
                var_value = tf.contrib.framework.load_variable(arg_checkpoint_dir, from_name)
                assign_ops.append(tf.assign(var, var_value))
            sess.run(assign_ops)
            print('Model loaded.')
            result = sess.run(output)
            cv2.imwrite(arg_output_dir + name, result[0][:, :, ::-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_dir', default='', type=str,
                        help='The filename of image to be completed.')
    parser.add_argument('-m', '--mask_dir', default='', type=str,
                        help='The filename of mask, value 255 indicates mask.')
    parser.add_argument('-c', '--checkpoint_dir', default='model_logs/release_places2_256_deepfill_v2', type=str,
                        help='The directory of tensorflow checkpoint.')
    parser.add_argument('-o', '--output_dir', default='output/', type=str,
                        help='The directory of tensorflow checkpoint.')
    args, unknown = parser.parse_known_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    inpaint(args.image_dir, args.mask_dir, args.checkpoint_dir, args.output_dir)
