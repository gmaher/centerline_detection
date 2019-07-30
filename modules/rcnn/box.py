import tensorflow as tf

def denormalize_boxes_tf(boxes, anchors):
    bx = boxes[:,:,:,:,0]*anchors[:,:,:,:,2]+anchors[:,:,:,:,0]
    by = boxes[:,:,:,:,1]*anchors[:,:,:,:,3]+anchors[:,:,:,:,1]

    bw = tf.exp(boxes[:,:,:,:,2])*anchors[:,:,:,:,2]
    bh = tf.exp(boxes[:,:,:,:,3])*anchors[:,:,:,:,3]

    out_boxes = tf.stack([bx,by,bw,bh], axis=4)

    return out_boxes

def bbox_iou_center_xy(bboxes1, bboxes2):
    """ same as `bbox_iou_corner_xy', except that we have
        center_x, center_y, w, h instead of x1, y1, x2, y2 """

    x11, y11, w11, h11 = tf.split(bboxes1, 4, axis=1)
    x21, y21, w21, h21 = tf.split(bboxes2, 4, axis=1)

    xi1 = tf.maximum(x11, tf.transpose(x21))
    xi2 = tf.minimum(x11, tf.transpose(x21))

    yi1 = tf.maximum(y11, tf.transpose(y21))
    yi2 = tf.minimum(y11, tf.transpose(y21))

    wi = w11/2.0 + tf.transpose(w21/2.0)
    hi = h11/2.0 + tf.transpose(h21/2.0)

    inter_area = tf.maximum(wi - (xi1 - xi2 + 1), 0) \
                  * tf.maximum(hi - (yi1 - yi2 + 1), 0)

    bboxes1_area = w11 * h11
    bboxes2_area = w21 * h21

    union = (bboxes1_area + tf.transpose(bboxes2_area)) - inter_area

    #return inter_area / (union+0.0001)
    return xi1
