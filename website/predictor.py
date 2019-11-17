import torch
import cv2
import numpy as np
from torch.autograd import Variable
from darknet import Darknet
from util import process_result, load_images, resize_image, cv_image2tensor, transform_result
import pickle as pkl
import argparse
import math
import random
import os.path as osp
import os
import sys
from datetime import datetime
from tqdm import tqdm

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")
    return names

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv3 object detection')
    parser.add_argument('-i', '--input', required=True, help='input image or directory or video')
    parser.add_argument('-t', '--obj-thresh', type=float, default=0.5, help='objectness threshold, DEFAULT: 0.5')
    parser.add_argument('-n', '--nms-thresh', type=float, default=0.4, help='non max suppression threshold, DEFAULT: 0.4')
    parser.add_argument('-o', '--outdir', default='detection', help='output directory, DEFAULT: detection/')

    args = parser.parse_args()

    return args

def create_batches(imgs, batch_size):
    num_batches = math.ceil(len(imgs) // batch_size)
    batches = [imgs[i*batch_size : (i+1)*batch_size] for i in range(num_batches)]

    return batches

def draw_bbox(imgs, bbox, colors, classes,read_frames,output_path):
    img = imgs[int(bbox[0])]

    label = classes[int(bbox[-1])]

    confidence = int(float(bbox[6])*100)

    label = label+' '+str(confidence)+'%'

    print(label)

    p1 = tuple(bbox[1:3].int())
    p2 = tuple(bbox[3:5].int())

    color = colors[int(bbox[-1])]
    cv2.rectangle(img, p1, p2, color, 4)
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
    p3 = (p1[0], p1[1] - text_size[1] - 4)
    p4 = (p1[0] + text_size[0] + 4, p1[1])
    cv2.rectangle(img, p3, p4, color, -1)

    cv2.putText(img, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 255, 255], 1)



def detect_image(model, fname, i):

    print('Loading input image(s)...')
    input_size = [int(model.net_info['height']), int(model.net_info['width'])]
    batch_size = int(model.net_info['batch'])

    imlist, imgs = load_images(fname)
    print('Input image(s) loaded')

    img_batches = create_batches(imgs, batch_size)

    # load colors and classes
    colors = pkl.load(open("pallete", "rb"))
    classes = load_classes("darknet/cfg/trash.names")


    start_time = datetime.now()
    print('Detecting...')

    for batchi, img_batch in zip((1,2), (3,4)):
        # img_tensors = [cv_image2tensor(img, input_size) for img in rangeimg_batch]
        # img_tensors = torch.stack(img_tensors)
        # img_tensors = Variable(img_tensors)
        return model.new_detect(batchi, i)
        detections = model(img_tensors, False).cpu()
        detections = process_result(detections, 100, 102)
        if len(detections) == 0:
            continue

        detections = transform_result(detections, img_batch, input_size)

        for detection in detections:
            draw_bbox(img_batch, detection, colors, classes,0,args.outdir)

        for i, img in enumerate(img_batch):
            save_path = osp.join(args.outdir, osp.basename(imlist[batchi*batch_size + i]))
            cv2.imwrite(save_path, img)
            print(save_path, 'saved')

    end_time = datetime.now()
    print('Detection finished in %s' % (end_time - start_time))

    return

def main():
    pass


if __name__ == "__main__":
    main()