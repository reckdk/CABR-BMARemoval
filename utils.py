import os
import cv2
import keras.backend as K
import numpy as np
kernelx_Sobel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

def Dice_coef(y_true, y_pred, smooth = 1):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def Dice_loss(y_true, y_pred):
    return  1.0 - Dice_coef(y_true, y_pred)


def get_dice_wholefield(mask_pred, mask_gt, smooth=1, mask_th=0.1):
    mask_pred = (mask_pred>mask_th)
    mask_gt = (mask_gt>mask_th)

    intersection = np.sum(mask_pred*mask_gt)
    return (2. * intersection + smooth) / (np.sum(mask_pred) + np.sum(mask_gt) + smooth)


def locate_cont_stripe(label_bin, invert=False):
    '''
    Given an binary array, locate and count continuous '1's.
    '''
    stripe_start_list = []
    stripe_list = []

    last_label = -1
    noise_num = 0
    for label_idx, label in enumerate(label_bin):
        # Last label check
        if (label_idx == len(label_bin)-1) and label==1:
            noise_num = noise_num + 1
            stripe_list.append(noise_num)
            stripe_start_list.append(label_idx-noise_num)
            break

        if label == 0:
            # Count as one stripe. Then reset last_label and noise_num.
            if last_label == 1:
                stripe_list.append(noise_num)
                stripe_start_list.append(label_idx-noise_num)
                last_label = 0
                noise_num = 0
        elif label == 1:
            noise_num = noise_num + 1
            last_label = 1
        else:
            raise ValueError('label_bin should be {0, 1}.')

    stripe_list = np.array(stripe_list)
    hist = np.histogram(stripe_list, range=(min(stripe_list), max(stripe_list)+1), 
                 bins=(max(stripe_list)-min(stripe_list)) + 1)
    
    return stripe_list, stripe_start_list, hist


def crop_valpatch(img, mask, patch_h, patch_w, stripe_start_h, stripe_width, start_w):
    stripe_diameter_half = int((stripe_width-1) // 2)
    stripe_diameter_additional = (stripe_width+1) % 2
    stripe_center = stripe_start_h + stripe_diameter_additional + stripe_diameter_half
    start_h = int(stripe_center - patch_h/2)

    mask_gt_patch = mask[start_h : start_h+patch_h, start_w : start_w+patch_w].copy()

    loss_mask_full = np.zeros_like(mask)
    loss_mask_full[stripe_start_h : stripe_start_h+stripe_width,:] = 1
    loss_mask = loss_mask_full[
            start_h : start_h+patch_h, start_w : start_w+patch_w]

    img_fused = np.transpose(np.array([img, mask]), (1,2,0)) # (H,W,2)
    # Remove mask to be inpainted.
    img_fused[stripe_start_h : stripe_start_h+stripe_width, :, 1] = 0
    img_fused_patch = img_fused[
            start_h : start_h+patch_h, start_w : start_w+patch_w, :]

    # gradient_input
    edges_y = cv2.filter2D(img_fused_patch[...,0], cv2.CV_64F, kernelx_Sobel) 
    edges_y = abs(edges_y)
    edges_y = edges_y/np.max(edges_y)
    edges_y *= loss_mask # Stripe-like grad.
    # (img, mask, edges_y)
    img_fused_patch = np.concatenate([img_fused_patch, edges_y[...,np.newaxis]], axis=2)

    # Calculate loss only in stripe region.
    mask_gt_patch *= loss_mask

    return img_fused_patch, loss_mask.reshape([patch_h, patch_w, 1]), \
               mask_gt_patch.reshape([patch_h, patch_w, 1])


def load_valdata_single(
        img, mask, stripe_label_valid, 
        patch_h, patch_w, start_w_list,
        inpaint_stripe_label=None,
        mask_lastiter=None):
    '''
    Prepare model input from one sample.
    '''
    img_mask_patch = []
    mask_patch_gt = []
    loss_mask = []

    img_h, img_w = img.shape
    assert img_w==1000

    # Preprocess img and mask.
    img = img/255.
    mask[mask>0] = 1.

    # Extract the locations of stripes to be inpainted.
    stripe_width_list, stripe_start_h_list, _ = locate_cont_stripe(stripe_label_valid)

    for start_w in start_w_list:
        # Crop patches for validation.
        for stripe_width, stripe_start_h in zip(stripe_width_list, stripe_start_h_list):
            img_mask_patch_cur, loss_mask_cur, mask_patch_gt_cur = crop_valpatch(
                img, mask, patch_h, patch_w, stripe_start_h, stripe_width, start_w)
            img_mask_patch.append(img_mask_patch_cur)
            loss_mask.append(loss_mask_cur)
            mask_patch_gt.append(mask_patch_gt_cur)

    img_mask_patch = np.array(img_mask_patch) # (V,H,W,3)
    loss_mask = np.array(loss_mask) # (V,H,W,1)
    mask_patch_gt = np.array(mask_patch_gt) # (V,H,W,1)

    return [img_mask_patch, loss_mask], mask_patch_gt