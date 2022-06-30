import math
from matplotlib import image
from skimage.filters import sobel,gaussian
import cv2
import numpy as np
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from skimage.color import *
from PIL import Image
from skimage import segmentation
from skimage.feature import canny
from skimage.feature import peak_local_max
from skimage.transform import hough_line,hough_line_peaks




class ImageClass:
    def __init__(self, image):
        self.image = image

    # Binarisation

    def Seuillage(self, s):
        print(s)
        imageX = self.image.copy()
        for i in range(1, imageX.shape[0]):
            for j in range(1, imageX.shape[1]):
                if imageX[i, j] < s:
                    imageX[i, j] = 0
                else:
                    imageX[i, j] = 255
        return imageX

    def Otsu(self):
        pixel_number = self.image.shape[0] * self.image.shape[1]
        mean_weigth = 1.0 / pixel_number
        his, bins = np.histogram(self.image, np.arange(0, 257))
        final_thresh = -1
        final_value = -1
        intensity_arr = np.arange(256)
        # This goes from 1 to 254 uint8 range (Pretty sure wont be those values)
        for t in bins[1:-1]:
            pcb = np.sum(his[:t])
            pcf = np.sum(his[t:])
            Wb = pcb * mean_weigth
            Wf = pcf * mean_weigth

            mub = np.sum(intensity_arr[:t] * his[:t]) / float(pcb)
            muf = np.sum(intensity_arr[t:] * his[t:]) / float(pcf)
            # print mub, muf
            value = Wb * Wf * (mub - muf) ** 2

            if value > final_value:
                final_thresh = t
                final_value = value
        final_img = self.image.copy()
        final_img[self.image > final_thresh] = 255
        final_img[self.image < final_thresh] = 0
        return final_img

        # contour

    def grad(self):
        self.image = self.Otsu()

        imageX = self.image.copy()
        imageY = self.image.copy()
        for i in range(1, self.image.shape[0] - 1):
            for j in range(1, self.image.shape[1] - 1):
                imageX[i, j] = self.image[i, j+1] - self.image[i, j]
                imageY[i, j] = self.image[i+1, j] - self.image[i, j]
        imageXY = self.image.copy()
        for i in range(1, self.image.shape[0] - 1):
            for j in range(1, self.image.shape[1] - 1):
                imageXY[i, j] = math.sqrt(
                    imageX[i, j] ** 2 + imageY[i, j] ** 2)
                # if imageXY[i, j] < seuil:
                #     imageXY[i, j] = 0
                # else:
                #     imageXY[i, j] = 255
        self.image = imageXY
        imageXY = self.Otsu()
        return imageXY

    def Sobel(self):
        self.image = self.Otsu()
        imageX = self.image.copy()
        imageY = self.image.copy()
        for i in range(1, self.image.shape[0] - 1):
            for j in range(1, self.image.shape[1] - 1):
                imageY[i, j] = -self.image[i-1, j-1] - 2*self.image[i, j-1] - self.image[i+1, j-1] \
                    + self.image[i - 1, j + 1] + 2 * \
                    self.image[i, j + 1] + self.image[i + 1, j + 1]
                imageX[i, j] = self.image[i-1, j-1] + 2*self.image[i-1, j] + self.image[i - 1, j + 1]\
                    - self.image[i+1, j-1] - 2 * \
                    self.image[i+1, j] - self.image[i + 1, j + 1]
        imageXY = self.image.copy()
        for i in range(1, self.image.shape[0] - 1):
            for j in range(1, self.image.shape[1] - 1):
                imageXY[i, j] = math.sqrt(
                    imageX[i, j] ** 2 + imageY[i, j] ** 2)
               
        self.image = imageXY
        imageXY = self.Otsu()

        return imageXY

    def Laplacien(self):
        self.image = self.Otsu()

        imageXY = self.image.copy()
        for i in range(1, self.image.shape[0] - 1):
            for j in range(1, self.image.shape[1] - 1):
                imageXY[i, j] = -4*self.image[i, j] + self.image[i-1, j] + self.image[i+1, j] \
                    + self.image[i, j - 1] + self.image[i, j + 1]
                # if imageXY[i, j] < seuil:
                #     imageXY[i, j] = 0
                # else:
                #     imageXY[i, j] = 255
        self.image = imageXY
        imageXY = self.Otsu()
        

        return imageXY

    def Robert(self):
        self.image = self.Otsu()

        imageX = self.image.copy()
        imageY = self.image.copy()
        for i in range(1, self.image.shape[0] - 1):
                for j in range(1, self.image.shape[1] - 1):
                    imageX[i, j] = -self.image[i-1, j+1] + self.image[i, j]
                    imageY[i, j] = -self.image[i+1, j+1] + self.image[i, j]
        imageXY = self.image.copy()
        for i in range(1, self.image.shape[0] - 1):
                for j in range(1, self.image.shape[1] - 1):
                    imageXY[i, j] = math.sqrt(
                    imageX[i, j] ** 2 + imageY[i, j] ** 2)
        self.image = imageXY
        imageXY = self.Otsu()
        return imageXY
    
    # filtrage

    def Moyenneur(self, taille):
        imagefiltrage = self.image.copy()
        x = int((taille - 1)/2)
        for i in range(x, self.image.shape[0] - x):
            for j in range(x, self.image.shape[1] - x):
                s = 0
                for n in range(-x, x):
                    for m in range(-x, x):
                        s += self.image[i+n, j+m]/(taille*taille)
                imagefiltrage[i, j] = s
                s = 0
        imagefiltrage= self.image.copy()
        return  cv2.blur(imagefiltrage,(taille,taille))

    def Median(self, taille):
        imagefiltrage = self.image.copy()
        x = int((taille - 1) / 2)
        for i in range(x, self.image.shape[0] - x):
            for j in range(x, self.image.shape[1] - x):
                liste = []
                if imagefiltrage[i, j] == 0 or imagefiltrage[i, j] == 255:
                    for n in range(-x, x):
                        for m in range(-x, x):
                            liste.append(imagefiltrage[i + n, j + m])
                    liste.sort()
                    imagefiltrage[i, j] = liste[x + 1]
                    while len(liste) > 0:
                        liste.pop()
        imagefiltrage= self.image.copy()
        return  cv2.medianBlur(imagefiltrage, taille)

    def h(self, x, y, v):
        x = (1/(2*math.pi*math.pow(v, 2))) * \
            (math.exp(-(math.pow(x, 2)+math.pow(y, 2))/(2*math.pow(v, 2))))
        return x

    def Gaussien(self, v):
        imagefiltrage = self.image.copy()
        x = 1
        for i in range(x, self.image.shape[0] - x):
            for j in range(x, self.image.shape[1] - x):
                s = 0
                for a in range(-x, x):
                    for b in range(-x, x):
                        s = s + self.h(a, b, v)*self.image[i+a, j+b]
                imagefiltrage[i, j] = s
                s = 0
        imagefiltrage= self.image.copy()
        return  cv2.GaussianBlur(imagefiltrage,(3,3),v)

    # Morphologie

    def dilatation(self, H):
        imagecopy = self.image.copy()
        for i in range(1, self.image.shape[0] - 1):
            for j in range(1, self.image.shape[1] - 1):
                s = 0
                for k in range(i - 1, i + 2):
                    for l in range(j - 1, j + 2):
                        s = s + self.image[k, l] * H[k - i + 1][l - j + 1]
                if (s == 0):
                    imagecopy[i][j] = 0
                else:
                    imagecopy[i][j] = 255
        return imagecopy

    def Erosion(self, H):
        imagecopy = self.image.copy()

        for i in range(0, self.image.shape[0]):
            for j in range(0, self.image.shape[1]):
                if (self.image[i][j] > 128):
                    self.image[i][j] = 255
                else:
                    self.image[i][j] = 0

        for i in range(1, self.image.shape[0] - 1):
            for j in range(1, self.image.shape[1] - 1):
                s = 0
                for k in range(i - 1, i + 2):
                    for l in range(j - 1, j + 2):
                        s = s + self.image[k, l] * H[k - i + 1][l - j + 1]
                if (s == 2295):
                    imagecopy[i][j] = 255
                else:
                    imagecopy[i][j] = 0
        return imagecopy

    def Ouverture(self, H):
        img = self.Erosion(self.image, H)
        image1 = self.dilatation(img, H)
        return image1

    def Fermeture(self, H):
        img = self.dilatation(self.image, H)
        image1 = self.Erosion(img, H)
        return image1

    # Operations

    def rotate_image(self, angle):
        # Get the image size
        # No that's not an error - NumPy stores image matricies backwards
        image_size = (self.image.shape[1], self.image.shape[0])
        image_center = tuple(np.array(image_size) / 2)

        # Convert the OpenCV 3x2 rotation matrix to 3x3
        rot_mat = np.vstack(
            [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
        )

        rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

        # Shorthand for below calcs
        image_w2 = image_size[0] * 0.5
        image_h2 = image_size[1] * 0.5

        # Obtain the rotated coordinates of the image corners
        rotated_coords = [
            (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
            (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
            (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
            (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
        ]

        # Find the size of the new image
        x_coords = [pt[0] for pt in rotated_coords]
        x_pos = [x for x in x_coords if x > 0]
        x_neg = [x for x in x_coords if x < 0]

        y_coords = [pt[1] for pt in rotated_coords]
        y_pos = [y for y in y_coords if y > 0]
        y_neg = [y for y in y_coords if y < 0]

        right_bound = max(x_pos)
        left_bound = min(x_neg)
        top_bound = max(y_pos)
        bot_bound = min(y_neg)

        new_w = int(abs(right_bound - left_bound))
        new_h = int(abs(top_bound - bot_bound))

        # We require a translation matrix to keep the image centred
        trans_mat = np.matrix([
            [1, 0, int(new_w * 0.5 - image_w2)],
            [0, 1, int(new_h * 0.5 - image_h2)],
            [0, 0, 1]
        ])

        # Compute the tranform for the combined rotation and translation
        affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

        # Apply the transform
        result = cv2.warpAffine(
            self.image,
            affine_mat,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR
        )
        img = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return result

    def hist(self):

        k = 0
        try:
            test = self.image.shape[2]
        except IndexError:
            k = 1
        if k == 1:
            h = ImageClass.histo(self.image)
            plt.subplot(1, 1, 1)
            plt.plot(h)
            plt.show()

        else:
            for i in range(0, 3):
                h = ImageClass.histo(self.image[:, :, i])
                plt.subplot(1, 3, i + 1)
                plt.plot(h)
            plt.show()

    def histo(image):
        h = np.zeros(256)
        s = image.shape
        for j in range(s[0]):
            for i in range(s[1]):
                valeur = image[j, i]
                h[valeur] += 1
        return h

    def imhist(im):
        m, n = im.shape
        h = [0.0] * 256
        for i in range(m):
            for j in range(n):
                h[im[i, j]] += 1
        return np.array(h) / (m * n)

    def cumsum(h):
        return [sum(h[:i + 1]) for i in range(len(h))]

    def histeq(self):
        h = ImageClass.imhist(self.image)

        cdf = np.array(ImageClass.cumsum(h))
        sk = np.uint8(255 * cdf)
        s1, s2 = self.image.shape
        Y = np.zeros_like(self.image)
        for i in range(0, s1):
            for j in range(0, s2):
                Y[i, j] = sk[self.image[i, j]]
        plt.figure()
        plt.imshow(Y, cmap="gray")
        plt.show()
        return Y

    def etire(self):
        MaxV = np.max(self.image)
        MinV = np.min(self.image)
        Y = np.zeros_like(self.image)
        m = self.image.shape
        for i in range(m[0]):
            for j in range(m[1]):
                Y[i, j] = (255 / (MaxV - MinV) * self.image[i, j] - MinV)
        return Y

        # segmentation

    def segmKmeans(self , cluster):
        try:
            img = self.image
            h, w = img.shape
            image_2d = img.reshape(h * w, 1)
            pixel_vals = np.float32(image_2d)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
            retval, labels, centers = cv2.kmeans(pixel_vals, cluster, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            segmented_data = centers[labels.flatten()]
            segmented_image = segmented_data.reshape((img.shape))
            self.image= segmented_image
            result = self.histeq()
        

        except Exception as e:
            print("kMeansSegmentation ERROR : ", e)
        return result


    def Markers(self):
        img = self.image
        x , y , z = img.shape
        im_ = gaussian(img, sigma=4)
        if z==3:
            br = sobel(im_[:,:,0])
            bg = sobel(im_[:,:,1])
            bb = sobel(im_[:,:,2])
            brgb = br+bg+bb
        else : 
            brgb= sobel(im_[:,:])

        markers = peak_local_max(brgb.max()-brgb)
        markers = peak_local_max(brgb.max()-brgb, threshold_rel=0.99, min_distance=50)
        return markers 

    def Growing(self):
        markers = self.Markers()
        img = self.image
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        (thresh, bin_img) = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

        h = img.shape[0]
        w = img.shape[1]

        out_img = np.zeros(shape=(gray_img.shape), dtype=np.uint8)

        seeds = markers.tolist()
        for seed in seeds:
            x = seed[0]
            y = seed[1]
            out_img[x][y] = 255
        directs = [(-1,-1), (0,-1), (1,-1), (1,0), (1,1), (0,1),(-1,1),(-1,0)]
        visited = np.zeros(shape=(gray_img.shape), dtype=np.uint8)
        while len(seeds):
            seed = seeds.pop(0)
            x = seed[0]
            y = seed[1]
            visited[x][y] = 1
            
            for direct in directs:
                cur_x = x + direct[0]
                cur_y = y + direct[1]
                if cur_x <0 or cur_y<0 or cur_x >= h or cur_y >=w :
                    continue
                if (not visited[cur_x][cur_y]) and (bin_img[cur_x][cur_y]==bin_img[x][y]) :
                    out_img[cur_x][cur_y] = 255
                    visited[cur_x][cur_y] = 1
                    seeds.append((cur_x,cur_y))
        bake_img = img.copy()
        h = bake_img.shape[0]
        w = bake_img.shape[1]
        for i in range(h):
            for j in range(w):
                if out_img[i][j] != 255:
                    bake_img[i][j][0] = 255
                    bake_img[i][j][1] = 255
                    bake_img[i][j][2] = 255

                    
        
        return bake_img


    
    def Hough(self): 
        image1 = self.image
        gray=cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
        dst = cv2.Canny(gray, 50, 200)
        response = "success"


        try:
            lines= cv2.HoughLines(dst, 1, math.pi/180.0, 100, np.array([]), 0, 0)
        

            a,b,c = lines.shape
            for i in range(a):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0, y0 = a*rho, b*rho
                pt1 = ( int(x0+1000*(-b)), int(y0+1000*(a)) )
                pt2 = ( int(x0-1000*(-b)), int(y0-1000*(a)) )
                cv2.line(image1, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)
        except:
            response = "No lines is detected!! Try with another image, thank you "



        return image1,response

    def HoughCir(self):
        img = self.image
        img = cv2.medianBlur(img,5)
        cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        resp = "success"
        try:
            circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                                        param1=50,param2=30,minRadius=0,maxRadius=0)
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                # draw the outer circle
                cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
            # cv2.imshow('detected circles',cimg)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        except :
            resp = "No circle is detected !! please try with another image"

        return cimg,resp

        # segmentation


    def Division_Judge(self, h0, w0, h, w):
        area = self.image[h0: h0 + h, w0: w0 + w]
        mean = np.mean(area)
        std = np.std(area, ddof=1)

        total_points = 0
        operated_points = 0

        for row in range(area.shape[0]):
            for col in range(area.shape[1]):
                if (area[row][col] - mean) < 2 * std:
                    operated_points += 1
                total_points += 1

        if operated_points / total_points >= 0.95:
            return True
        else:
            return False

    def Merge(self, h0, w0, h, w):
        img = self.image
        for row in range(h0, h0 + h):
            for col in range(w0, w0 + w):
                if img[row, col] > 100 and img[row, col] < 200:
                    img[row, col] = 0
                else:
                    img[row, col] = 255

    def Recursion(self, h0, w0, h, w):

        # If the splitting conditions are met, continue to split
        if not self.Division_Judge( h0, w0, h, w) and min(h, w) > 5:
            # Recursion continues to determine whether it can continue to split
            # Top left square
            self.Division_Judge( h0, w0, int(h0 / 2), int(w0 / 2))
            # Upper right square
            self.Division_Judge( h0, w0 + int(w0 / 2), int(h0 / 2), int(w0 / 2))
            # Lower left square
            self.Division_Judge(h0 + int(h0 / 2), w0, int(h0 / 2), int(w0 / 2))
            # Lower right square
            self.Division_Judge( h0 + int(h0 / 2), w0 + int(w0 / 2), int(h0 / 2), int(w0 / 2))
        else:
            # Merge
            self.Merge( h0, w0, h, w)

    def segmenation_Part_region(self):
        img = self.image
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        segemented_img = img_gray.copy()
        self.image = segemented_img
        self.Recursion(0, 0, segemented_img.shape[0], segemented_img.shape[1])
        return segemented_img




