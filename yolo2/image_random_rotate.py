# This python file is used to preprocess training images and the labels on them.
# rliu added a random angle rotation step in the data_augmentation function and fill_detectio function


#!/usr/bin/python
# encoding: utf-8
import random
import os
from PIL import Image
import numpy as np



def scale_image_channel(im, c, v):
    cs = list(im.split())
    cs[c] = cs[c].point(lambda i: i * v)
    out = Image.merge(im.mode, tuple(cs))
    return out

def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)
    
    def change_hue(x):
        x += hue*255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x
    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))

    im = im.convert('RGB')
    #constrain_image(im)
    return im

def rand_scale(s):
    scale = random.uniform(1, s)
    if(random.randint(1,10000)%2): 
        return scale
    return 1./scale

def random_distort_image(im, hue, saturation, exposure):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res = distort_image(im, dhue, dsat, dexp)
    return res

def data_augmentation(img, shape, jitter, hue, saturation, exposure):
    oh = img.height  
    ow = img.width
    
    dw =int(ow*jitter)
    dh =int(oh*jitter)

    pleft  = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop   = random.randint(-dh, dh)
    pbot   = random.randint(-dh, dh)

    swidth =  ow - pleft - pright
    sheight = oh - ptop - pbot

    sx = float(swidth)  / ow
    sy = float(sheight) / oh
    
    flip = 0
    cropped = img.crop( (pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))

    dx = (float(pleft)/ow)/sx
    dy = (float(ptop) /oh)/sy

    sized = cropped.resize(shape)

    if flip: 
        sized = sized.transpose(Image.FLIP_LEFT_RIGHT)
    img = random_distort_image(sized, hue, saturation, exposure)
    ang = random.randint(0,359);
    img = img.rotate(ang)    
    return img, flip, dx,dy,sx,sy,ang 

def fill_truth_detection(labpath, w, h, flip, dx, dy, sx, sy, ang):
    max_boxes = 50
    label = np.zeros((max_boxes,5))
    if os.path.getsize(labpath):
        bs = np.loadtxt(labpath)
        if bs is None:
            return label
        bs = np.reshape(bs, (-1, 5))
        cc = 0
        for i in range(bs.shape[0]):
            x1 = bs[i][1] - bs[i][3]/2
            y1 = bs[i][2] - bs[i][4]/2
            x2 = bs[i][1] + bs[i][3]/2
            y2 = bs[i][2] + bs[i][4]/2
            
            x1 = min(0.999, max(0, x1 * sx - dx)) 
            y1 = min(0.999, max(0, y1 * sy - dy)) 
            x2 = min(0.999, max(0, x2 * sx - dx))
            y2 = min(0.999, max(0, y2 * sy - dy))
            
            bs[i][1] = (x1 + x2)/2
            bs[i][2] = (y1 + y2)/2
            bs[i][3] = (x2 - x1)
            bs[i][4] = (y2 - y1)
            
            x_0 = bs[i][1]     # x coordinate of center of bbox before rotation
            y_0 = bs[i][2]     # y coordinate of center of bbox before rotation
            l_r = np.sqrt(np.square(x_0-0.5) + np.square(y_0-0.5))   # length from center of bbox before rotation to image center
            alpha = np.arctan2(y_0-0.5, x_0-0.5)    # original angle between center of bbox and positive x axis
            if alpha < 0:    # alpha belongs to [0, 360); alphs is integer
                alpha = alpha + 2*np.pi
            ang = ang / 180.0 * np.pi   # use radians
            x_r = 0.5 + l_r * np.cos(ang + alpha)   # position of center of bbox after rotation
            y_r = 0.5 + l_r * np.sin(ang + alpha)
            
            # x1_r: xmin; y1_r: ymin; x2_r: xmax; y2_r: ymax;
            x1_r = x_r - bs[i][3]/2
            y1_r = y_r - bs[i][4]/2
            x2_r = x_r + bs[i][3]/2
            y2_r = y_r + bs[i][4]/2
            
            # for the bbox to be inside the image
            x1_r = min(0.999, max(0, x1_r))
            y1_r = min(0.999, max(0, y1_r))
            x2_r = min(0.999, max(0, x2_r))
            y2_r = min(0.999, max(0, y2_r))

            # return xmin ymin width height
            bs[i][1] = (x1_r + x2_r)/2
            bs[i][2] = (y1_r + y2_r)/2
            bs[i][3] = (x2_r - x1_r)
            bs[i][4] = (y2_r - y1_r)
            
            if flip:
                bs[i][1] =  0.999 - bs[i][1] 
            
            if bs[i][3] < 0.001 or bs[i][4] < 0.001:
                continue
            label[cc] = bs[i]
            cc += 1
            if cc >= 50:
                break

    label = np.reshape(label, (-1))
    return label

def load_data_detection(imgpath, shape, jitter, hue, saturation, exposure):
    labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')

    ## data augmentation
    img = Image.open(imgpath).convert('RGB')
    img,flip,dx,dy,sx,sy,ang = data_augmentation(img, shape, jitter, hue, saturation, exposure)
    label = fill_truth_detection(labpath, img.width, img.height, flip, dx, dy, 1./sx, 1./sy, ang)
    return img,label
