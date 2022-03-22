CLASSES = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
              'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
              'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
              'train', 'tvmonitor')

PASCAL_PALETTE = [0, 0, 0,
                128, 0, 0,
                0, 128, 0,
                128, 128, 0,
                0, 0, 128,
                128, 0, 128,
                0, 128, 128,
                128, 128, 128,
                64, 0, 0,
                192, 0, 0,
                64, 128, 0,
                192, 128, 0,
                64, 0, 128,
                192, 0, 128,
                64, 128, 128,
                192, 128, 128,
                0, 64, 0,
                128, 64, 0,
                0, 192, 0,
                128, 192, 0,
                0, 64, 128]

PASCAL_PALETTE_NOVEL = [0,0, 0,
                128, 0, 0,
                0, 128, 0,
                128, 128, 0,
                0, 0, 128,
                128, 0, 128,
                0, 128, 128,
                128, 128, 128,
                64, 0, 0,
                192, 0, 0,
                64, 128, 0,
                192, 128, 0,
                64, 0, 128,
                192, 0, 128,
                64, 128, 128,
                192, 128, 128,
                255, 127, 36, # Chocolate1
                255, 20, 147, # DeepPink
                25, 25, 112,  ## MidnightBlue
                119, 136, 153, ## LightSlateGray
                47, 79, 79] ## DarkSlateGray

def with_ignore_white(palette=PASCAL_PALETTE):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    palette[-3:] = [255,255,255]
    return palette
