import math
import numpy as np
import numpy.ma as ma
import cv2
import imgaug.augmenters as iaa
from imgaug.augmentables.lines import LineString, LineStringsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from scipy.interpolate import InterpolatedUnivariateSpline
from clrnet.datasets.process.transforms import CLRTransforms

from ..registry import PROCESS

@PROCESS.register_module
class MyGenerateLaneLine(object):
    def __init__(self, transforms=None, cfg=None, training=True):
        print('Init Generate Lane Line')
        print('object: ', object)
        print('transforms: ', transforms)
        print('cfg: ', cfg)
        self.transforms = transforms
        self.img_w, self.img_h = cfg['img_w'], cfg['img_h']
        self.num_points = cfg['num_points']
        self.n_offsets = cfg['num_points']
        self.n_strips = cfg['num_points'] - 1
        self.strip_size = self.img_h / self.n_strips # 160 / (72 - 1) = 2.25 = 2
        self.max_lanes = cfg['max_lanes'] # 4
        self.offsets_ys = np.arange(self.img_h, -1, -self.strip_size) # [160, 158, 156, 154, ..., 2, 0]
        self.training = training

        print('transforms: ', transforms)
        print('transforms: ', type(transforms))
        print('Elotte: ', transforms is None)
        if transforms is None:
            print('transforms is: ', transforms)
            transforms = CLRTransforms(self.img_h, self.img_w)

        if transforms is not None:
            img_transforms = []
            for aug in transforms:
                # print('aug: ', aug)
                # a_key = list(aug.keys())[-1]
                # aug = aug[a_key]
                # print('aug: ', aug)
                p = aug['p']
                if aug['name'] != 'OneOf':
                    img_transforms.append(
                        iaa.Sometimes(p=p,
                                      then_list=getattr(
                                          iaa,
                                          aug['name'])(**aug['parameters'])))
                else:
                    img_transforms.append(
                        iaa.Sometimes(
                            p=p,
                            then_list=iaa.OneOf([
                                getattr(iaa,
                                        aug_['name'])(**aug_['parameters'])
                                for aug_ in aug['transforms']
                            ])))
        else:
            img_transforms = []
        self.transform = iaa.Sequential(img_transforms)

    def lane_to_linestrings(self, lanes):
        lines = []
        for lane in lanes:
            lines.append(LineString(lane)) # LineString class holds (x, y) coords

        return lines

    def sample_lane(self, points, sample_ys):
        # this function expects the points to be sorted
        # points = lane = [ [4.80999994 125.08872986] [ 11.45199966 120.10115051] ...]
        # sample_ys = offsets_ys = [160, 158, 156, 154, ..., 2, 0]
        points = np.array(points)
        print("points: ", points)
        if not np.all(points[1:, 1] < points[:-1, 1]):
            raise Exception('Annotaion points have to be sorted')
        x, y = points[:, 0], points[:, 1]

        # interpolate points inside domain minimum of 3rd degree polynomial
        assert len(points) > 1
        interp = InterpolatedUnivariateSpline(y[::-1],
                                              x[::-1],
                                              k=min(3,
                                                    len(points) - 1))
        domain_min_y = y.min()
        domain_max_y = y.max()
        sample_ys_inside_domain = sample_ys[(sample_ys >= domain_min_y)
                                            & (sample_ys <= domain_max_y)]
        assert len(sample_ys_inside_domain) > 0
        interp_xs = interp(sample_ys_inside_domain)

        # extending line segment ?
        # extrapolate lane to the bottom of the image with a straight line 
        # using the 2 points closest to the bottom
        two_closest_points = points[:2]

        # fit a straight line to the two closest data points
        extrap = np.polyfit(two_closest_points[:, 1],
                            two_closest_points[:, 0],
                            deg=1)

        # the Y-coordinates that fall outside the original data range
        extrap_ys = sample_ys[sample_ys > domain_max_y]

        # evaluate the extrapolated X-coordinates (extrap_xs) 
        # corresponding to the Y-coordinates in extrap_ys
        # The extrap represents the coefficients of the linear polynomial 
        # obtained from the np.polyfit() function
        extrap_xs = np.polyval(extrap, extrap_ys)

        # concat extrap_xs with interp_xs points
        all_xs = np.hstack((extrap_xs, interp_xs))

        # separate between inside and outside points

        # boolean mask, which filters the inside X points
        inside_mask = (all_xs >= 0) & (all_xs < self.img_w)

        # split into inside and outside X points
        xs_inside_image = all_xs[inside_mask]
        xs_outside_image = all_xs[~inside_mask]

        return xs_outside_image, xs_inside_image

    def filter_lane(self, lane):
        assert lane[-1][1] <= lane[0][1]
        filtered_lane = []
        used = set()
        for p in lane:
            if p[1] not in used:
                filtered_lane.append(p)
                used.add(p[1])

        return filtered_lane

    def transform_annotation(self, anno, img_wh=None):
        img_w, img_h = self.img_w, self.img_h

        old_lanes = anno['lanes']

        # removing lanes with less than 2 points
        old_lanes = filter(lambda x: len(x) > 1, old_lanes)
        # sort lane points by Y (bottom to top of the image)
        old_lanes = [sorted(lane, key=lambda x: -x[1]) for lane in old_lanes]
        # remove points with same Y (keep first occurrence)
        old_lanes = [self.filter_lane(lane) for lane in old_lanes]
        # normalize the annotation coordinates
        old_lanes = [[[
            x * self.img_w / float(img_w), y * self.img_h / float(img_h)
        ] for x, y in lane] for lane in old_lanes]

        eps = 1e5

        # create transformed annotations
        # holds the lane prediction values
        # 2 scores, 1 start_y, 1 start_x, 1 theta, 1 length, N+1 coordinates
        lanes = np.ones(
            (self.max_lanes, 2 + 1 + 1 + 1 + 1 + self.n_offsets), dtype=np.float32
        ) * -eps
        
        # lanes.shape [4, 78] 72 

        lanes_endpoints = np.ones((self.max_lanes, 2))

        # lanes are invalid by default, lanes.shape [4, 2],
        # The invalid state is column[0] = 1 and column[1] = 0
        lanes[:, 0] = 1
        lanes[:, 1] = 0
        for lane_idx, lane in enumerate(old_lanes):
            if lane_idx >= self.max_lanes:
                break

            try:
                # xs_outside_image, xs_inside_image ~ y coord
                xs_outside_image, xs_inside_image = self.sample_lane(
                    lane, self.offsets_ys)
            except AssertionError:
                continue
            if len(xs_inside_image) <= 1:
                continue

            # concat xs_outside_image with xs_inside_image
            all_xs = np.hstack((xs_outside_image, xs_inside_image))

            # activating the lanes from 1, 0 to 0, 1 state
            lanes[lane_idx, 0] = 0 
            lanes[lane_idx, 1] = 1 

            # print("len(xs_outside_image): ", len(xs_outside_image))
            # print("len(xs_inside_image): ", len(xs_inside_image)

            # start points coordinates
            # start_y
            lanes[lane_idx, 2] = len(xs_outside_image) / self.n_strips # (n_strips = 72 - 1)
            # start_x
            lanes[lane_idx, 3] = xs_inside_image[0]
            
            # stores the angles
            thetas = []

            # starts from i=1 to skip the first element 
            # since it will be used as a reference point for calculating angles 
            # with subsequent points
            for i in range(1, len(xs_inside_image)):

                # calculates the angle between the line segment 
                # fromed by the current point (xs_inside_image[i]) 
                # and the reference point (xs_inside_image[0]) 
                # with respect to the horizontal axis
                # reference: LaneATT (1) formula
                # arctg result interval (-pi/2, pi/2) / pi -> (-1/2, 1/2) ->
                # -> if (-1/2, 0) + 1 -> final interval (0, 1) where
                # (0, 1/2) is positive and (1/2, 1) is negative degrees
                theta = math.atan(
                    i * self.strip_size /
                    (xs_inside_image[i] - xs_inside_image[0] + eps)) / math.pi

                theta = theta if theta > 0 else 1 - abs(theta)
                thetas.append(theta)

            avg_theta = sum(thetas) / len(thetas)

            # lanes[lane_idx,
            #       4] = (theta_closest + theta_far) / 2  # averaged angle
            lanes[lane_idx, 4] = avg_theta
            lanes[lane_idx, 5] = len(xs_inside_image)
            lanes[lane_idx, 6:6 + len(all_xs)] = all_xs 
            lanes_endpoints[lane_idx, 0] = (len(all_xs) - 1) / self.n_strips
            lanes_endpoints[lane_idx, 1] = xs_inside_image[-1]


        new_anno = {
          'label': lanes,
          'old_anno': anno,
          'lane_endpoints': lanes_endpoints
        }
        return new_anno

    def linestrings_to_lanes(self, lines):
        lanes = []
        for line in lines:
            lanes.append(line.coords)

        return lanes

    def __call__(self, sample):
        img_org = sample['img']
        line_strings_org = self.lane_to_linestrings(sample['lanes'])
        """
          class LineStringsOnImage:
            Object that represents all line strings on a single image
          line_strings : list of imgaug.augmentables.lines.LineString
            List of line strings on the image.
          shape : tuple of int or ndarray
            The shape of the image on which the objects are placed.
        """
        line_strings_org = LineStringsOnImage(line_strings_org,
                                              shape=img_org.shape)

        # Trying to transfrom annotations multiple times
        for i in range(30):
            if self.training:
                mask_org = SegmentationMapsOnImage(sample['mask'],
                                                   shape=img_org.shape)
                img, line_strings, seg = self.transform(
                    image=img_org.copy().astype(np.uint8),
                    line_strings=line_strings_org,
                    segmentation_maps=mask_org)
            else:
                img, line_strings = self.transform(
                    image=img_org.copy().astype(np.uint8),
                    line_strings=line_strings_org)
            line_strings.clip_out_of_image_()
            new_anno = {'lanes': self.linestrings_to_lanes(line_strings)}
            try:
                annos = self.transform_annotation(new_anno,
                                                  img_wh=(self.img_w,
                                                          self.img_h))
                label = annos['label']
                lane_endpoints = annos['lane_endpoints']
                break
            except:
                if (i + 1) == 30:
                    self.logger.critical(
                        'Transform annotation failed 30 times :(')
                    exit()
        
        sample['img'] = img.astype(np.float32) / 255.
        sample['lane_line'] = label
        sample['lanes_endpoints'] = lane_endpoints
        sample['gt_points'] = new_anno['lanes']
        sample['seg'] = seg.get_arr() if self.training else np.zeros(
            img_org.shape)

        print("lane_line shape: ", ma.shape(sample['lane_line']))
        """
          sample = {
            'img_name': 'driver_161_90frame/06030946_0784.MP4/00090.jpg', 
            'img_path': 'data/CULane/driver_161_90frame/06030946_0784.MP4/00090.jpg', 
            'mask_path': 'data/CULane/laneseg_label_w16/driver_161_90frame/06030946_0784.MP4/00090.png', 
            'lane_exist': array([0, 1, 1, 1]), (list/train_gt.txt)
            'lanes': [[(166.811, 38.18600000000001), # the x cooridnates is from the 00090.lines.txt
                        (163.122, 40.897999999999996), # the y coordinates is the samples
                        (159.227, 43.61), 
                        (155.538, 46.322), 
                        (151.643, 49.034000000000006), 
                        (147.918, 51.745999999999995), 
                        (144.228, 54.458), 
                        (140.334, 57.16900000000001), 
                        (136.644, 59.881), 
                        (132.75, 62.59299999999999), 
                        (129.024, 65.305), 
                        (125.334, 68.017), 
                        (121.44, 70.72900000000001), 
                        (117.75, 73.441), 
                        (113.856, 76.15299999999999), 
                        (110.121, 78.864), 
                        (106.386, 81.576), 
                        (102.651, 84.28800000000001), 
                        (99.166, 87.0)], 
                        [(191.819, 38.18600000000001), 
                        (193.838, 40.897999999999996), 
                        (195.858, 43.61), 
                        (197.878, 46.322), 
                        (199.872, 49.034000000000006), 
                        (201.892, 51.745999999999995), 
                        (203.912, 54.458), 
                        (205.932, 57.16900000000001), 
                        (207.952, 59.881), 
                        (209.946, 62.59299999999999), 
                        (211.966, 65.305), 
                        (213.986, 68.017), 
                        (216.005, 70.72900000000001), 
                        (218.025, 73.441), 
                        (220.138, 76.15299999999999), 
                        (222.094, 78.864), 
                        (224.114, 81.576), 
                        (226.071, 84.28800000000001), 
                        (227.935, 87.0)], 
                        [(202.465, 38.18600000000001), 
                        (209.252, 40.897999999999996), 
                        (215.901, 43.61), 
                        (222.549, 46.322), 
                        (229.337, 49.034000000000006), 
                        (236.001, 51.745999999999995), 
                        (242.65, 54.458), 
                        (249.527, 57.16900000000001), 
                        (256.176, 59.881), 
                        (262.824, 62.59299999999999), 
                        (269.72, 65.305), 
                        (276.368, 68.017), 
                        (283.017, 70.72900000000001), 
                        (289.666, 73.441), 
                        (296.559, 76.15299999999999), 
                        (303.207, 78.864), 
                        (309.856, 81.576), 
                        (316.734, 84.28800000000001), 
                        (322.695, 87.0)]], 
            'img': array([[[0.        , 0.        , 0.        ], ... ,
            'mask': array([[1, 1, 1, ..., 1, 1, 1],
                    [1, 1, 1, ..., 1, 1, 1],
                    [1, 1, 1, ..., 1, 1, 1],
                    ...,
                    [1, 1, 1, ..., 1, 1, 1],
                    [0, 0, 0, ..., 0, 0, 0],
                    [0, 0, 0, ..., 0, 0, 0]], dtype=uint8),
            'lane_line': array([[ 0.0000000e+00,  1.0000000e+00,  0.0000000e+00,  1.5297635e+02, ...
            'lanes_endpoints': array([[  0.43661972, 191.17263591],
                                      [  0.45070423, 217.68424574],
                                      [  0.46478873, 229.69162085],
                                      [  1.        ,   1.        ]]),
            'gt_points': [array([[191.37253 ,  89.7261  ],
                                  [188.40715 ,  95.30216 ],
                                  [185.23438 , 100.90918 ],
                                  [182.269   , 106.48525 ],
                                  [179.0962  , 112.09225 ],
                                  [176.09459 , 117.67373 ],
                                  [173.1282  , 123.249954],
                                  [169.95615 , 128.85495 ],
                                  [166.98976 , 134.43117 ],
                                  [163.81798 , 140.03802 ],
                                  [160.81535 , 145.61966 ],
                                  [157.84898 , 151.19586 ],
                                  [154.6772  , 156.80272 ],
                                  [152.97688 , 159.999   ]], dtype=float32),
                          array([[216.55315 ,  85.970116],
                                  [219.33517 ,  90.68888 ],
                                  [222.11821 ,  95.40752 ],
                                  [224.90125 , 100.126144],
                                  [227.6581  , 104.84867 ],
                                  [230.44113 , 109.5673  ],
                                  [233.22417 , 114.28593 ],
                                  [236.00693 , 119.0027  ],
                                  [238.78996 , 123.72133 ],
                                  [241.5468  , 128.44386 ],
                                  [244.32988 , 133.16248 ],
                                  [247.11288 , 137.8811  ],
                                  [249.89494 , 142.59987 ],
                                  [252.67795 , 147.31853 ],
                                  [255.55463 , 152.02318 ],
                                  [258.27295 , 156.74956 ],
                                  [260.18945 , 159.999   ]], dtype=float32), 
                            array([[227.27261,  84.37118],
                                  [234.85556,  88.37383],
                                  [242.29956,  92.39723],
                                  [249.74251,  96.42078],
                                  [257.32648, 100.42328],
                                  [264.78558, 104.44442],
                                  [272.22952, 108.46782],
                                  [279.90286, 112.4551 ],
                                  [287.34683, 116.47849],
                                  [294.78983, 120.50204],
                                  [302.4825 , 124.48833],
                                  [309.92548, 128.51186],
                                  [317.36948, 132.53525],
                                  [324.81345, 136.55867],
                                  [332.5031 , 140.5454 ],
                                  [339.94583, 144.5671 ],
                                  [347.3898 , 148.59047],
                                  [355.0644 , 152.57947],
                                  [361.8156 , 156.70619]], dtype=float32)],
            'seg': array([[0, 0, 0, ..., 1, 1, 1],
                          [0, 0, 0, ..., 1, 1, 1],
                          [0, 0, 0, ..., 1, 1, 1],
                          ...,
                          [0, 0, 0, ..., 0, 0, 0],
                          [0, 0, 0, ..., 0, 0, 0],
                          [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)
            }                
        """
          
        # print("sample: ", sample)
        return sample