import numpy as np

def iou(bboxes1, bboxes2):
        x1, y1, w1, h1 = np.split(bboxes1, 4, axis=1)
        x2, y2, w2, h2 = np.split(bboxes2, 4, axis=1)

        x11 = x1-w1/2
        y11 = y1-h1/2
        x12 = x1+w1/2
        y12 = y1+h1/2

        x21 = x2-w2/2
        y21 = y2-h2/2
        x22 = x2+w2/2
        y22 = y2+h2/2

        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))

        interArea = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0)

        boxAArea = (x12 - x11) * (y12 - y11)
        boxBArea = (x22 - x21) * (y22 - y21)

        iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)

        return iou

def bbox_transform(anchors, boxes):
    '''
    note must be same scale
    '''
    N = boxes.shape[0]
    t = np.zeros((N,4))

    t[:,0] = (boxes[:,0]-anchors[:,0])/anchors[:,2]
    t[:,1] = (boxes[:,1]-anchors[:,1])/anchors[:,3]
    t[:,2] = np.log(boxes[:,2]/anchors[:,2])
    t[:,3] = np.log(boxes[:,3]/anchors[:,3])

    return t

def inv_bbox_transform(anchors, boxes):
    '''
    note must be same scale
    '''
    N = boxes.shape[0]
    t = np.zeros((N,4))

    t[:,0] = anchors[:,2]*boxes[:,0]+anchors[:,0]
    t[:,1] = anchors[:,3]*boxes[:,1]+anchors[:,1]
    t[:,2] = np.exp(boxes[:,2])*anchors[:,2]
    t[:,3] = np.exp(boxes[:,3])*anchors[:,3]

    return t

def create_box_gt(anchors, gt):
    '''
    anchors - Nanchors x 4
    gt - Ntruth x 4

    compute iou
     > 0.7 iou then assign class label 1 and weight 1
     < 0.3 iou then assign class label 0 and weight 1
     otherwise weight 0

     returns matched_boxes, labels, weights (Nanchors x 4)

     label - true if any anchor > 0.7
    '''

    #First calculate IOUs, then set weights of outside boxes to 0
    # or with IOU too small to 0
    n_anchors = anchors.shape[0]
    weights   = np.zeros((n_anchors))
    labels    = np.zeros((n_anchors))

    ious = iou(anchors, gt)

    max_inds = np.argmax(ious, axis=0)
    max_ious = np.amax(ious,axis=1)


    labels[max_inds] = 1
    labels[max_ious>0.7] = 1
    weights += 1
    weights[(labels != 1) & (max_ious > 0.3) & (max_ious < 0.7)] = 0

    max_inds = np.argmax(ious, axis=1)
    matched_boxes = gt[max_inds]

    t = bbox_transform(anchors,matched_boxes)

    return t, matched_boxes, labels, weights
