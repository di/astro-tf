#!/usr/bin/python

import random
from PIL import Image
import colorsys
import glob
import sys
import math

import numpy
from scipy.cluster.vq import kmeans2
import warnings
warnings.filterwarnings('ignore')


def _distance(a, b):
    a_x, a_y = a
    b_x, b_y = b
    return math.sqrt((a_x - b_x)**2 + (a_y - b_y)**2)


def _to_hsv(r, g, b):
    return colorsys.rgb_to_hsv(float(r)/256, float(g)/256, float(b)/256)


def mean_distance(res, idx, xyz, _max=5):
    # Determine the average cluster distance
    dist_min = sys.maxint
    avg_dist = [[], [], []]
    for i, g in enumerate(idx):
        dist = _distance((xyz[i][0], xyz[i][1]), (res[g][0], res[g][1]))
        avg_dist[g].append(dist)
    for l in avg_dist:
        try:
            avg = sum(l)/len(l)
            dist_min = min(dist_min, avg)
        except ZeroDivisionError:
            pass
    return dist_min/_max


def collinearity(res, _max=125):
    # Determine the collinearity metric
    return abs(sum([
        (res[2][0] - res[0][0]) * (res[1][1] - res[0][1]),
        (res[2][1] - res[0][1]) * (res[0][0] - res[1][0]),
    ]))/_max


def hue_ratio(im, val_min=.25, val_max=.90):
    stds = 2
    a_avg = 0.713228289156627
    a_std = 0.03178173297698*stds
    b_avg = 0.192794843373494
    b_std = 0.03031368437293*stds
    a_c = b_c = c_c = 0

    xyz = []
    pix = im.load()
    wid, hei = im.size
    for x in range(10, wid-10):
        for y in range(10, hei-10):
            h, s, v = _to_hsv(*pix[x, y])

            if val_max > v > val_min:
                xyz.append([x, y, h])
                if a_avg - a_std < h < a_avg + a_std:
                    a_c += 1
                elif b_avg - b_std < h < b_avg + b_std:
                    b_c += 1
                else:
                    c_c += 1
    if (a_c < 10 or b_c < 10) and val_min > .075:
        return hue_ratio(im, val_min=val_min-.01)
    else:
        return float(a_c + b_c) / float(a_c + b_c + c_c), xyz


def features(filename, val_min=.25, val_max=.90):
    im = Image.open(filename)
    dist_min = sys.maxint
    collin_min = sys.maxint

    hue_rat, xyz = hue_ratio(im)

    # Repeated k-means-cluster stuff
    for t in range(20):
        # Do a k-means-squared clustering
        res, idx = kmeans2(numpy.array(xyz), 3)

        dist_min = min(dist_min, mean_distance(res, idx, xyz))
        collin_min = min(collin_min, collinearity(res))

    return [hue_rat, collin_min, dist_min]

if __name__ == "__main__":
    outs = []
    for i in glob.glob("./training_data/invalid/*.jpg"):
        outs.append(", ".join([str(x) for x in features(i) + [0.0]]))
    for i in glob.glob("./training_data/valid/*.jpg"):
        outs.append(", ".join([str(x) for x in features(i) + [1.0]]))
    random.shuffle(outs)
    training_data = open('astro.train', 'w')
    for line in outs:
        training_data.write("%s\n" % line)

    for test in range(1, 4):
        outs = []
        for i in glob.glob("./test_data/trial{}/invalid/*.jpg".format(test)):
            outs.append(", ".join([str(x) for x in features(i) + [0.0]]))
        for i in glob.glob("./test_data/trial{}/valid/*.jpg".format(test)):
            outs.append(", ".join([str(x) for x in features(i) + [1.0]]))
        test_data = open('astro.test.{}'.format(test), 'w')
        for line in outs:
            test_data.write("%s\n" % line)
