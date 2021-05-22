# -*- coding: utf-8 -*-
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


#OBJEKTNI RAZREDI ZA RAZLIČNE NAČINE DOLOČANJA RAZLIK MED SLIKAMA

class GrayDiff:
    def __init__(self, frame0, threshold, ksize):
        '''
        Konstruktor objekta za sivinsko razliko
        frame0 -> začetna slika
        threshold -> prag za upragovljanje
        ksize -> velikost jedra za Gaussov filter
        return: ničesar ne vrne
        '''
        self.img0 = cv.GaussianBlur(frame0, (ksize, ksize), sigmaX = 0) #sigma determined from ksize
        self.img0 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
        self.img0 = np.array(self.img0).astype('int32')
        
        self.threshold = threshold
        self.ksize = ksize

    def subtract(self, frame):
        '''
        Metoda za odštevanje sivinskih slik
        frame -> trenutna slika (lahko je barvna)
        return -> sivinska razlika med trenutno in začetno sliko
        '''
        gray = cv.GaussianBlur(frame, (self.ksize, self.ksize), sigmaX = 0) #sigma determined from ksize
        gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
        gray = np.array(gray).astype('int32')
        diff = np.abs(gray - self.img0)
        
        diff[diff<0] = 0
        diff[diff>255] = 255
        diff = diff.astype('uint8')
        
        return diff


class HSDiff:
    def __init__(self, frame0, threshold, ksize):
        '''
        Konstruktor objekta za evklidsko razdaljo v Hue Saturation polarnih koordinatah
        frame0 -> začetna slika
        threshold -> prag za upragovljanje
        ksize -> velikost jedra za Gaussov filter
        return: ničesar ne vrne
        '''
        self.img0 = cv.GaussianBlur(frame0, (ksize, ksize), sigmaX = 0) #sigma determined from ksize
        img0_HSV = cv.cvtColor(self.img0, cv.COLOR_BGR2HSV)
        img0_HSV = np.array(img0_HSV).astype('int32')
        
        self.h0 = np.array(img0_HSV[:,:,0]).astype('float64')
        self.s0 = np.array(img0_HSV[:,:,1]).astype('float64')
        
        #self.x0 = s0 * np.cos(2*h0*np.pi/180)
        #self.y0 = s0 * np.sin(2*h0*np.pi/180)
        
        self.threshold = threshold
        self.ksize = ksize
    
    def subtract(self, frame):
        '''
        Metoda za odštevanje sivinskih slik
        frame -> trenutna barvna slika
        return -> evklidska razdalja med piksli trenutne in začetne slike v HS polarnih koordinatah
        '''
        frame_gauss = cv.GaussianBlur(frame, (self.ksize, self.ksize), sigmaX = 0) #sigma determined from ksize
    
        frame_HSV = cv.cvtColor(frame_gauss, cv.COLOR_BGR2HSV)
        frame_HSV = np.array(frame_HSV).astype('int32')
        
        h = np.array(frame_HSV[:,:,0]).astype('float64')
        s = np.array(frame_HSV[:,:,1]).astype('float64')
        
        h_diff = np.abs(self.h0 - h).astype('uint8')
        diff = self.s0**2 + s**2 - 2 * self.s0 * s * cos_2deg(h_diff)
        diff = 255.0 * diff / (510.0**2)
        
        diff[diff<0] = 0
        diff[diff>255] = 255
        diff = diff.astype('uint8')
        
        return diff

#FUNKCIJE

def set_points(iImage, n):
    '''
    Funkcija za določanje točk s klikanjem
    iImage -> slika na kateri določamo točke
    n -> št. točk, ki jih želimo določiti
    return: array točk
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(iImage, cmap='gray')
    
    points = []
    def onclick(event):
        if event.key == 'shift':
            x, y = event.xdata, event.ydata
            points.append((x, y))
            ax.plot(x, y, 'or')
            fig.canvas.draw()
        
    ka = fig.canvas.mpl_connect('button_press_event', onclick)
    
    while len(points) < 9:
        cv.waitKey(200)
    
    return points

def determine_color(value, thresholds):
    if value > thresholds[0]:
        color = (0,255,0) #BGR
    elif value > thresholds[1]:
        color = (255,255,0) #BGR
    elif value > thresholds[2]:
        color = (255,0,0) #BGR
    elif value > thresholds[3]:
        color = (255,0,255) #BGR
    else:
        color = (0,0,255) #BGR
        
    return color

def circle_mask(imSize, center, r):
    #izračuna masko za krog
    #https://stackoverflow.com/questions/49330080/numpy-2d-array-selecting-indices-in-a-circle
    x = np.arange(0, imSize[0])
    y = np.arange(0, imSize[1])
    #arr = np.zeros((y.size, x.size))
    
    cx = float(center[0])
    cy = float(center[1])
    
    #izračun maske
    mask = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 < r**2
    return mask
    
def ring_mask(imSize, center, r_min, r_max):
    #izračuna masko za kolobar
    #https://stackoverflow.com/questions/49330080/numpy-2d-array-selecting-indices-in-a-circle
    x = np.arange(0, imSize[0])
    y = np.arange(0, imSize[1])
    #arr = np.zeros((y.size, x.size))
    
    cx = float(center[0])
    cy = float(center[1])
    
    #izračun maske
    mask_outer = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 < r_max**2
    mask_inner = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 > r_min**2
    return np.logical_and(mask_outer, mask_inner)

peg_prev_states_in = np.zeros((9,4), dtype='bool') #last 4 states for each peg - used for change empty->full
peg_prev_states_out = np.zeros((9,4), dtype='bool') #last 4 states for each peg - used for change full->empty
peg_states = np.zeros(9, dtype='bool') #are pegs in holes?
peg_change_iter = np.zeros(9) #num. of iteration when last change occured
def peg_states_changes(inside_circle, circle_th, inside_ring, ring_th, iteration):
    '''
    Funkcija določi status zatičev (vstavljen / ni vstavljen).
    inside_circle -> array povp. vrednosti na krogih okoli lukenj
    circle_th -> prag za zaznavo vstavljenega zatiča
    inside_ring -> array povp. vrednosti na kolobarjih okoli lukenj
    ring_th -> prag za zaznavo roke, ki prekriva območje ob luknji
    ring_th -> trenutna iteracija oz. zaporadna številka slike videoposnetka
    return: peg_states -> stanje zatičev
            peg_change_iter -> iteracija zadnje spremembe
    '''
    global peg_prev_states_in
    global peg_prev_states_out
    global peg_states
    global peg_change_iter
    
    inside_circle = np.array(inside_circle)
    inside_ring = np.array(inside_ring)
    
    if(inside_circle.shape != inside_ring.shape):
        raise Exception('lengths of arrays inside_circle and inside_ring must be equal')
    
    
    peg_inside = inside_circle > circle_th
    ring_empty = inside_ring < ring_th
    peg_inside_ring_empty = np.logical_and(peg_inside, ring_empty)
    #print('peg_inside:', peg_inside)
    peg_prev_states_in = np.concatenate((np.vstack(peg_inside_ring_empty), peg_prev_states_in[:,0:3]), axis=1)
    
    peg_prev_states_out = np.concatenate((np.vstack(peg_inside), peg_prev_states_out[:,0:3]), axis=1)
    #print('peg_prev_states_out\n', peg_prev_states_out.astype('int32'))
    
    for i in range(9):
        if peg_states[i]: #if current status: peg in hole
            if np.sum(peg_prev_states_out[i,0:3]) == 0:
                peg_states[i] = False
                peg_change_iter[i] = iteration - 3
            elif np.sum(peg_prev_states_out[i]) == 1:
                peg_states[i] = False
                peg_change_iter[i] = iteration - 4
        else: # if current status: hole empty
            if np.sum(peg_prev_states_in[i,0:3]) == 3:
                peg_states[i] = True
                peg_change_iter[i] = iteration - 3
            elif np.sum(peg_prev_states_in[i]) == 3:
                peg_states[i] = True
                peg_change_iter[i] = iteration - 4
                
    return peg_states, peg_change_iter

def points_near(point1, point2, maxDist):
    #preveri ali sta točki blizu
    #v trenutnem programu ni uporabljena
    point1 = np.array(point1)
    point2 = np.array(point2)
    
    dist = np.linalg.norm(point1 - point2)
    return dist <= maxDist


def hough_find_holes(frame0):
    '''
    Funkcija poišče lokacije lukenj za zatiče.
    frame0 -> slika na kateri iščemo luknje
    returns: correct_circles -> najdene luknje
    '''
    frame0_gauss = cv.GaussianBlur(cv.cvtColor(frame0, cv.COLOR_BGR2GRAY), (3,3), 0)
    
    circles = cv.HoughCircles(frame0_gauss, cv.HOUGH_GRADIENT, 1, 30, param1=100, param2=50, maxRadius=300, minRadius=100)
    frame_draw = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
    circles = circles[0]
    for circle in circles:
        cv.circle(frame_draw, (circle[0], circle[1]), int(circle[2]), (0,0,255), thickness = 2)
    cv.circle(frame_draw, (circles[0,0], circles[0,1]), int(circles[0,2]), (0,255,0), thickness = 2)
    big_circle = circles[0]
    
    circles = cv.HoughCircles(frame0_gauss, cv.HOUGH_GRADIENT, 1, 30, param1=120, param2=12, maxRadius=20, minRadius=5)
    circles = circles[0]
    
    dist_to_big_circle = np.linalg.norm(circles[:,0:2] - big_circle[0:2], axis=1)
    are_circles_correct = np.logical_and((dist_to_big_circle > 1.5*big_circle[2]), (dist_to_big_circle < 3.5*big_circle[2]))
    correct_circles = circles[are_circles_correct]
    
    return correct_circles
               

#vnaprej izračunane vrednosti kosinusa
cos_2deg_arr = np.array([ 1.00000000e+00,  9.99390827e-01,  9.97564050e-01,  9.94521895e-01,
        9.90268069e-01,  9.84807753e-01,  9.78147601e-01,  9.70295726e-01,
        9.61261696e-01,  9.51056516e-01,  9.39692621e-01,  9.27183855e-01,
        9.13545458e-01,  8.98794046e-01,  8.82947593e-01,  8.66025404e-01,
        8.48048096e-01,  8.29037573e-01,  8.09016994e-01,  7.88010754e-01,
        7.66044443e-01,  7.43144825e-01,  7.19339800e-01,  6.94658370e-01,
        6.69130606e-01,  6.42787610e-01,  6.15661475e-01,  5.87785252e-01,
        5.59192903e-01,  5.29919264e-01,  5.00000000e-01,  4.69471563e-01,
        4.38371147e-01,  4.06736643e-01,  3.74606593e-01,  3.42020143e-01,
        3.09016994e-01,  2.75637356e-01,  2.41921896e-01,  2.07911691e-01,
        1.73648178e-01,  1.39173101e-01,  1.04528463e-01,  6.97564737e-02,
        3.48994967e-02,  6.12323400e-17, -3.48994967e-02, -6.97564737e-02,
       -1.04528463e-01, -1.39173101e-01, -1.73648178e-01, -2.07911691e-01,
       -2.41921896e-01, -2.75637356e-01, -3.09016994e-01, -3.42020143e-01,
       -3.74606593e-01, -4.06736643e-01, -4.38371147e-01, -4.69471563e-01,
       -5.00000000e-01, -5.29919264e-01, -5.59192903e-01, -5.87785252e-01,
       -6.15661475e-01, -6.42787610e-01, -6.69130606e-01, -6.94658370e-01,
       -7.19339800e-01, -7.43144825e-01, -7.66044443e-01, -7.88010754e-01,
       -8.09016994e-01, -8.29037573e-01, -8.48048096e-01, -8.66025404e-01,
       -8.82947593e-01, -8.98794046e-01, -9.13545458e-01, -9.27183855e-01,
       -9.39692621e-01, -9.51056516e-01, -9.61261696e-01, -9.70295726e-01,
       -9.78147601e-01, -9.84807753e-01, -9.90268069e-01, -9.94521895e-01,
       -9.97564050e-01, -9.99390827e-01, -1.00000000e+00, -9.99390827e-01,
       -9.97564050e-01, -9.94521895e-01, -9.90268069e-01, -9.84807753e-01,
       -9.78147601e-01, -9.70295726e-01, -9.61261696e-01, -9.51056516e-01,
       -9.39692621e-01, -9.27183855e-01, -9.13545458e-01, -8.98794046e-01,
       -8.82947593e-01, -8.66025404e-01, -8.48048096e-01, -8.29037573e-01,
       -8.09016994e-01, -7.88010754e-01, -7.66044443e-01, -7.43144825e-01,
       -7.19339800e-01, -6.94658370e-01, -6.69130606e-01, -6.42787610e-01,
       -6.15661475e-01, -5.87785252e-01, -5.59192903e-01, -5.29919264e-01,
       -5.00000000e-01, -4.69471563e-01, -4.38371147e-01, -4.06736643e-01,
       -3.74606593e-01, -3.42020143e-01, -3.09016994e-01, -2.75637356e-01,
       -2.41921896e-01, -2.07911691e-01, -1.73648178e-01, -1.39173101e-01,
       -1.04528463e-01, -6.97564737e-02, -3.48994967e-02, -1.83697020e-16,
        3.48994967e-02,  6.97564737e-02,  1.04528463e-01,  1.39173101e-01,
        1.73648178e-01,  2.07911691e-01,  2.41921896e-01,  2.75637356e-01,
        3.09016994e-01,  3.42020143e-01,  3.74606593e-01,  4.06736643e-01,
        4.38371147e-01,  4.69471563e-01,  5.00000000e-01,  5.29919264e-01,
        5.59192903e-01,  5.87785252e-01,  6.15661475e-01,  6.42787610e-01,
        6.69130606e-01,  6.94658370e-01,  7.19339800e-01,  7.43144825e-01,
        7.66044443e-01,  7.88010754e-01,  8.09016994e-01,  8.29037573e-01,
        8.48048096e-01,  8.66025404e-01,  8.82947593e-01,  8.98794046e-01,
        9.13545458e-01,  9.27183855e-01,  9.39692621e-01,  9.51056516e-01,
        9.61261696e-01,  9.70295726e-01,  9.78147601e-01,  9.84807753e-01,
        9.90268069e-01,  9.94521895e-01,  9.97564050e-01,  9.99390827e-01])

def cos_2deg(iArr):
    return cos_2deg_arr[iArr]

    
    
#TOČKE ZATIČEV
peg_points_L = np.array([[352, 280],
       [460, 279],
       [568, 280],
       [351, 388],
       [460, 388],
       [568, 390],
       [353, 497],
       [461, 497],
       [567, 499]])
    
peg_points_R = np.array([[757, 267],
       [867, 273],
       [981, 281],
       [748, 376],
       [860, 382],
       [971, 392],
       [741, 485],
       [850, 495],
       [959, 502]])
    
peg_points = (peg_points_L, peg_points_R)
    
peg_points_6322 = np.array([[727, 265],
       [835, 267],
       [948, 266],
       [726, 376],
       [835, 376],
       [946, 378],
       [725, 484],
       [833, 486],
       [944, 487]])