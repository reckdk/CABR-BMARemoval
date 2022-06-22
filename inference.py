import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import cv2

from keras.layers import (
        Concatenate, Dense, Flatten, AveragePooling2D, 
        BatchNormalization, Conv2D, Lambda, Activation, Multiply,
        UpSampling2D,Input, normalization,add, MaxPooling2D, 
        Conv2DTranspose,concatenate)
import tensorflow as tf

# Suppress messages.
tf.logging.set_verbosity(tf.logging.ERROR)
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(1)
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
config_tf.log_device_placement = True
sess = tf.Session(config=config_tf)
set_session(sess)


from utils import (
        Dice_loss, Dice_coef, 
        get_dice_wholefield, load_valdata_single)

if __name__ == '__main__':
    # Inference parameters.
    patch_h, patch_w = [64, 496]
    INVALID_H = 64 # Invalid for destripe.
    SEG_TH = 0.1
    start_w_list = [0, int((1000-patch_w)//2), 1000-patch_w]
            
    # Load image and mask.
    img_path = './dataset/AwakeOCA_orig.tif'
    print('Loading image from: {}.'.format(img_path))
    img = cv2.imread(img_path, 0)
    mask = cv2.imread(img_path.replace('_orig', '_mask_defective'), 0)
    mask_gt = cv2.imread(img_path.replace('_orig.tif', '_mask_gt.png'), 0)
    img_h, img_w = img.shape
    mask_output = mask.copy()

    # Load row labels. 
    stripe_label = np.load('./dataset/stripe_label.npy')
    assert len(stripe_label)>0, 'No stripe_label found. Please provide row labels.'
    stripe_label_valid = stripe_label
    stripe_label_valid[:int(INVALID_H/2)] = 0 # Not inpaint boundary.
    stripe_label_valid[(img_h - int(INVALID_H/2)):] = 0
    stripe_label_valid_idx = np.squeeze(np.argwhere(stripe_label_valid))

    # Prepare input patches.
    x_batch, y_batch = load_valdata_single(
        img, mask_gt, stripe_label_valid,
        patch_h, patch_w, start_w_list)
    img_mask_patch, loss_mask = x_batch
    mask_patch_gt = y_batch
    w_num = len(start_w_list)
    h_num = int(y_batch.shape[0]/w_num)

    # Load model.
    model_path = './model_weights/bestmodel.hdf5'
    print('Loading model from: {}.'.format(model_path))
    inpainting_model=load_model(model_path, custom_objects={
        "tf": tf, "Dice_loss":Dice_loss, "Dice_coef":Dice_coef})
    # Prediction.
    pred_batch = np.uint8(inpainting_model.predict_on_batch(x_batch)>SEG_TH)

    # Post-processing.
    # Recontruct wholefield image.
    pred_whole = np.zeros((len(stripe_label_valid_idx), img_w))
    pred_count_whole = np.zeros((len(stripe_label_valid_idx), img_w))
    for idx_w,start_w in enumerate(start_w_list):
        pred_tmp = np.zeros((0, patch_w))
        for idx_cur in range(h_num*idx_w, h_num*(idx_w+1)):
            centerline_idx = np.squeeze(np.argwhere(loss_mask[idx_cur,:,0,0]))
            pred_cur = pred_batch[idx_cur,centerline_idx,:,0]
            pred_cur = pred_cur.reshape((-1,patch_w))
            pred_tmp = np.concatenate((pred_tmp, pred_cur), axis=0)
        assert pred_tmp.shape[0] == len(stripe_label_valid_idx)
        pred_whole[:, start_w : start_w+patch_w] += pred_tmp
        pred_count_whole[:, start_w : start_w+patch_w] += 1
    pred_whole /= pred_count_whole

    # Stitch preds into original mask.
    mask_output[stripe_label_valid_idx] = pred_whole*255

    dice_wholefield = get_dice_wholefield(
        mask_output[stripe_label_valid_idx], mask_gt[stripe_label_valid_idx])

    cv2.imwrite(img_path.replace('_orig', '_mask_pred_Dice{:.03f}'.format(
            dice_wholefield)), mask_output)
    cv2.imwrite(img_path.replace('_orig', '_enhanced'), img*(mask_output/255 > 0))

    print('Prediction completed. Val_Dice: {:.03f}'.format(dice_wholefield))
