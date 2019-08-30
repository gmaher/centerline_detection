import numpy as np
from scipy import ndimage

#TODO centerline by interpolating top and bottom and then rot+shift

class Vessel:
    def __init__(self, pos, length, width, rot):
        """
        pos - (x,y) coordinate
        rot - rotation radians
        """
        self.pos    = pos
        self.length = length
        self.width  = width
        self.rot    = rot

        self.points          = np.zeros((2,4))
        self.points[0,[0,3]] -= self.width/2
        self.points[0,[1,2]] += self.width/2
        self.points[1,[0,1]] -= self.length/2
        self.points[1,[2,3]] += self.length/2

        self.rot_mat      = np.zeros((2,2))
        self.rot_mat[0,0] = np.cos(self.rot)
        self.rot_mat[0,1] = -np.sin(self.rot)
        self.rot_mat[1,0] = np.sin(self.rot)
        self.rot_mat[1,1] = np.cos(self.rot)

        self.points = self.rot_mat.dot(self.points)

        self.points[0] += self.pos[0]
        self.points[1] += self.pos[1]

        self.box = np.zeros((4))
        self.box[0] = np.amin(self.points[0])
        self.box[1] = np.amin(self.points[1])
        self.box[2] = np.amax(self.points[0])-self.box[0]
        self.box[3] = np.amax(self.points[1])-self.box[1]

    def paint(self,H,W):
        self.im = np.zeros((H,W))

        vH = self.length*H
        vW = self.width*W

        start_y = int(H/2-vH/2)
        end_y   = int(H/2+vH/2)
        start_x = int(W/2-vW/2)
        end_x   = int(W/2+vW/2)

        self.im[start_y:end_y, start_x:end_x] = 1

        rot_deg = self.rot*180.0/np.pi

        self.im = ndimage.rotate(self.im, rot_deg, reshape=False)

        shift_x = W*(self.pos[0]-0.5)
        shift_y = H*(self.pos[1]-0.5)

        self.im = ndimage.shift(self.im, [shift_y,shift_x])
