import cv2
import os
import os.path as osp
import matplotlib.pyplot as plt
from torch.functional import Tensor


COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
    (255, 0, 128),
    (0, 128, 255),
    (0, 255, 128),
    (128, 255, 255),
    (255, 128, 255),
    (255, 255, 128),
    (60, 180, 0),
    (180, 60, 0),
    (0, 60, 180),
    (0, 180, 60),
    (60, 0, 180),
    (180, 0, 60),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
]

def display_image_in_actual_size(img, img_path):

  dpi = 80
  im_data = plt.imread(img_path)
  height, width, depth = im_data.shape

  # What size does the figure need to be in inches to fit the image?
  figsize = width / float(dpi), height / float(dpi)

  # Create a figure of the right size with one axes that takes up the full figure
  fig = plt.figure(figsize=figsize)
  # [left, bottom, width, height]
  ax = fig.add_axes([0, 0, 1, 1]) # spans the entire figure area

  # Hide spines, ticks, etc.
  # ax.axis('off')

  # Set title
  ax.set_title('/'.join(img_path.split('/')[-3:]), fontsize=12)

  # Display the image.
  ax.imshow(img.permute(1, 2, 0))

  plt.show()

def show_img(img: Tensor, img_path: str):

  dpi = 80
  # im_data = plt.imread(img_path)
  depth, height, width = img.shape

  # What size does the figure need to be in inches to fit the image?
  figsize = width / float(dpi), height / float(dpi)

  # Create a figure of the right size with one axes that takes up the full figure
  fig = plt.figure(figsize=figsize)
  # [left, bottom, width, height]
  ax = fig.add_axes([0, 0, 1, 1]) # spans the entire figure area

  # Hide spines, ticks, etc.
  # ax.axis('off')

  # Set title
  ax.set_title('/'.join(img_path.split('/')[-3:]), fontsize=12)

  # Display the image.
  ax.imshow(img.permute(1, 2, 0))

  plt.show()

def tb_img_with_lanes_(img, lanes, out_file=None, width=4):
  lanes_xys = []
  for _, lane in enumerate(lanes):
    xys = []
    for x, y in lane:
      if x <= 0 or y <= 0:
        continue
      x, y = int(x), int(y)
      xys.append((x, y))
    lanes_xys.append(xys)
  lanes_xys.sort(key=lambda xys : xys[0][0])


def imshow_lanes(img, lanes, show=False, out_file=None, width=4):
    lanes_xys = []
    for _, lane in enumerate(lanes):
        xys = []
        for x, y in lane:
            if x <= 0 or y <= 0:
                continue
            x, y = int(x), int(y)
            xys.append((x, y))
        lanes_xys.append(xys)
    lanes_xys.sort(key=lambda xys : xys[0][0])

    for idx, xys in enumerate(lanes_xys):
        for i in range(1, len(xys)):
            cv2.line(img, xys[i - 1], xys[i], COLORS[idx], thickness=width)


    if show:
        cv2.imshow(img)
        cv2.waitKey(0)

    if out_file:
        if not osp.exists(osp.dirname(out_file)):
            os.makedirs(osp.dirname(out_file))
        cv2.imwrite(out_file, img)