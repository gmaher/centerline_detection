import numpy as np

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

    def paint(self,x):
        H = x.shape[0]
        W = x.shape[1]
