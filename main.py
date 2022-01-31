import cv2
import numpy as np
import os


class RoomSegmentation:
    def __init__(self, m=1, b=5, min_dist=1600, min_l=100, image_name=None, image_file=None):
        self.m_thresh = m
        self.b_thresh = b
        self.l_thresh = min_dist
        self.first_line_thresh = min_l
        if image_file is None:
            self.img = cv2.imread(image_name, 0)
            self.img = self.convert_to_bw(self.img, 200)
        else:
            self.img = image_file

        self.original_img = self.img.copy()
        if image_name is None:
            os.mkdir("Camera")
            self.dir = "Camera"
        else:
            if image_name.split('.')[1].split('/')[-1] not in os.listdir('./results'):
                os.mkdir('./results/' + image_name.split('.')[1].split('/')[-1])
            self.dir = './results/' + image_name.split('.')[1].split('/')[-1]

    @staticmethod
    def convert_to_sqr(img):
        row, col = img.shape
        if row > col:
            zero_side_image = np.zeros((row, (row - col) // 2), np.uint8)
            new_image = np.concatenate((zero_side_image, img, zero_side_image), axis=1)
        elif col > row:
            zero_side_image = np.zeros(((col - row) // 2, col), np.uint8)
            new_image = np.concatenate((zero_side_image, img, zero_side_image), axis=0)
        else:
            new_image = img
        return new_image

    def run(self, name):
        kernel = np.ones((2, 2), np.uint8)

        self.img = self.convert_to_sqr(self.img)
        cv2.imwrite(self.dir + "/0-SquareImage.png", self.img)

        detected_lines = self.line_segmentation(self.img, '/1-DetectedLines')
        extended_lines = self.extend_lines(detected_lines)
        extended_lined_img = cv2.createLineSegmentDetector(1).drawSegments(self.img, extended_lines)
        cv2.imwrite(self.dir + "/2-ExtendedLines.png", extended_lined_img)

        lined_img = cv2.cvtColor(extended_lined_img, cv2.COLOR_BGR2GRAY)

        rotated_img = self.rotate_img(lined_img, 90)
        detected_lines = self.line_segmentation(rotated_img, '/3-DetectedLines')
        extended2_lines = self.extend_lines(detected_lines)
        extended2_lined_img = cv2.createLineSegmentDetector(1).drawSegments(rotated_img, extended2_lines)
        cv2.imwrite(self.dir + "/4-ExtendedLines.png", extended2_lined_img)

        lined_img = cv2.cvtColor(extended2_lined_img, cv2.COLOR_BGR2GRAY)
        lined_img = self.convert_to_bw(lined_img, 200)
        lined_img = cv2.erode(lined_img, kernel, iterations=1)
        lined_img = cv2.morphologyEx(lined_img, cv2.MORPH_OPEN, kernel, iterations=5)
        cv2.imwrite(self.dir + "/5-MorphologyErodeOpen.png", lined_img)

        # circles = self.circle_detection(lined_img)

        second_rotated_img = self.rotate_img(lined_img, 270)
        components = self.connected_component(second_rotated_img)

        # original = self.transfer_to_original(components)
        # cv2.imshow('components' + name, components)
        # cv2.imwrite('output_%s.png' % name, components)
        cv2.imwrite(self.dir + '/6-output_%s.png' % name, components)

        # cv2.waitKey()

    @staticmethod
    def circle_detection(image):
        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=50, maxRadius=0)
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 0), 2)
        return image

    @staticmethod
    def convert_to_bw(image, thresh):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j] > thresh:
                    image[i][j] = 255
                else:
                    image[i][j] = 0
        return image

    def line_segmentation(self, img, name):
        lsd = cv2.createLineSegmentDetector(1)
        lines = lsd.detect(img)[0]
        cv2.imwrite(self.dir + name + '.png', lsd.drawSegments(img, lines))
        detected_lines = []
        for line in lines:
            x1 = line[0][0]
            y1 = line[0][1]
            x2 = line[0][2]
            y2 = line[0][3]
            if (x2 - x1) ** 2 + (y2 - y1) ** 2 < self.first_line_thresh:
                continue
            if x2 - x1 == 0:
                continue
            m = (y2 - y1) / (x2 - x1)
            b = y2 - m * x2
            detected_lines.append([m, b, (x1, y1), (x2, y2)])
        return detected_lines

    def extend_lines(self, lines):
        temp_lines = []
        final_lines = []
        for i in range(len(lines)):
            for j in range(i, len(lines)):
                l1 = lines[i]
                l2 = lines[j]
                if abs(l1[0] - l2[0]) < self.m_thresh and \
                        abs(l1[1] - l2[1]) < self.b_thresh:
                    temp_lines.append([[l1[2][0], l1[2][1], l2[2][0], l2[2][1]]])
                    temp_lines.append([[l1[2][0], l1[2][1], l2[3][0], l2[3][1]]])
                    temp_lines.append([[l1[3][0], l1[3][1], l2[2][0], l2[2][1]]])
                    temp_lines.append([[l1[3][0], l1[3][1], l2[3][0], l2[3][1]]])
                    temp_lines.sort(key=lambda k: (k[0][0] - k[0][2]) ** 2 + (k[0][1] - k[0][3]) ** 2)
                    k = temp_lines[0]
                    if (k[0][0] - k[0][2]) ** 2 + (k[0][1] - k[0][3]) ** 2 < self.l_thresh:
                        final_lines.append(temp_lines[0])
                    temp_lines = []
        return np.asarray(final_lines)

    @staticmethod
    def draw_lines(img, lines, name):
        tmp = cv2.createLineSegmentDetector(1).drawSegments(img, lines)
        cv2.imshow(name, tmp)

    @staticmethod
    def connected_component(img):
        ret, labels = cv2.connectedComponents(img, connectivity=8, ltype=4)

        # Map component labels to hue val
        label_hue = np.uint8(360 * labels / np.max(labels))
        blank_ch = 255 * np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

        # cvt to BGR for display
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

        # set bg label to black
        labeled_img[label_hue == 0] = 0
        return labeled_img

    @staticmethod
    def rotate_img(img, theta):
        rows, cols = img.shape
        m = cv2.getRotationMatrix2D((cols / 2, rows / 2), theta, 1)
        dst = cv2.warpAffine(img, m, (cols, rows))
        return dst

    def transfer_to_original(self, final_img):
        sqr_org_img = self.convert_to_sqr(self.original_img)
        w, h = sqr_org_img.shape
        blank_ch = np.zeros((w, h, 3))
        for i in range(w):
            for j in range(h):
                if np.sum(final_img[i][j]) != 0:
                    blank_ch[i][j] = final_img[i][j]
                else:
                    val = sqr_org_img[i][j]
                    blank_ch[i][j] = [val, val, val]
        return blank_ch


if __name__ == '__main__':
    office_a = RoomSegmentation(m=1, b=70, min_dist=1600, min_l=10,
                                image_file=None, image_name='./Data/office_a.png')
    office_a.run('office_a')

    # office_b = RoomSegmentation(m=1, b=5, min_dist=1600, min_l=100,
    #                             image_file=None, image_name='./Data/office_b.png')
    # office_b.run('office_b')

    # office_c = RoomSegmentation(m=1, b=5, min_dist=1600, min_l=100,
    #                             image_file=None, image_name='./Data/office_c.png')
    # office_c.run('office_c')

    # office_d = RoomSegmentation(m=1, b=15, min_dist=1600, min_l=100,
    #                             image_file=None, image_name='./Data/office_d.png')
    # office_d.run('office_d')

    # office_e = RoomSegmentation(m=1, b=20, min_dist=1600, min_l=100,
    #                             image_file=None, image_name='./Data/office_e.png')
    # office_e.run('office_e')

    # office_f = RoomSegmentation(m=1, b=15, min_dist=1600, min_l=100,
    #                             image_file=None, image_name='./Data/office_f.png')
    # office_f.run('office_f')

    # office_g = RoomSegmentation(m=1, b=20, min_dist=6400, min_l=100,
    #                             image_file=None, image_name='./Data/office_g.png')
    # office_g.run('office_g')

    # office_h = RoomSegmentation(m=1, b=100, min_dist=4900, min_l=1,
    #                             image_file=None, image_name='./Data/office_h.png')
    # office_h.run('office_h')

    # office_i = RoomSegmentation(m=1, b=15, min_dist=3600, min_l=100,
    #                             image_file=None, image_name='./Data/office_i.png')
    # office_i.run('office_i')

    # nlb = RoomSegmentation(m=0.04, b=15, min_dist=1600, min_l=60,
    #                        image_file=None, image_name='./Data/NLB.png')
    # nlb.run('NLB')

    # Freiburg101_scan = RoomSegmentation(m=1, b=6, min_dist=1600, min_l=100,
    #                                     image_file=None, image_name='./Data/Freiburg101_scan.png')
    # Freiburg101_scan.run('Freiburg101_scan')

    # Freiburg79_scan = RoomSegmentation(m=1, b=20, min_dist=1600, min_l=100,
    #                                    image_file=None, image_name='./Data/Freiburg79_scan.png')
    # Freiburg79_scan.run('Freiburg79_scan')
