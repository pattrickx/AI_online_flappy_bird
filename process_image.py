import cv2
import numpy as np
bird = {"img":cv2.imread("./template/bird.png",cv2.IMREAD_UNCHANGED),
        "threshold":0.50}
restart_buttom = {"img":cv2.imread("./template/restart_buttom.png",cv2.IMREAD_UNCHANGED),
        "threshold":0.80}
pipe_up = {"img":cv2.imread("./template/pipe_up.png",cv2.IMREAD_UNCHANGED),
        "threshold":0.80}
pipe_down = {"img":cv2.imread("./template/pipe_down.png",cv2.IMREAD_UNCHANGED),
        "threshold":0.80}


def update_frame(frame,p1,template,color=(0,255,255)):
    w = template.shape[1]
    h = template.shape[0]
    cv2.rectangle(frame,p1,(p1[0]+w,p1[1]+h),color,2)

def AI_inputs(img,x_bird=None,y_center_bird=None,x_end_pipe=None , y_center_hole=None):
    frame = img.copy()
    result = cv2.matchTemplate(img,restart_buttom["img"], cv2.TM_CCOEFF_NORMED)
    _,max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val>restart_buttom["threshold"]:
        update_frame(frame,max_loc,restart_buttom["img"])
        return True,frame
    
    result = cv2.matchTemplate(img,bird["img"], cv2.TM_CCOEFF_NORMED)
    _,max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val>bird["threshold"]:
        update_frame(frame,max_loc,bird["img"],(0,255,0))
        x_bird= max_loc[0]
        y_center_bird = max_loc[1]+(bird["img"].shape[0]/2)
    else:
        return [],frame

    ## find pipe up
    result = cv2.matchTemplate(img,pipe_up["img"], cv2.TM_CCOEFF_NORMED)
    yloc,xloc = np.where(result>=pipe_up["threshold"])

    if len(xloc)>0:
        filter = xloc>x_bird
        nexts_pipe_x = xloc[filter]
        nexts_pipe_y = yloc[filter]
        nexts_pipe_xinds = nexts_pipe_x.argsort()
        nexts_pipe_x = nexts_pipe_x[nexts_pipe_xinds[::-1]]
        nexts_pipe_y = nexts_pipe_y[nexts_pipe_xinds[::-1]]
        if len(nexts_pipe_y)>0:
            update_frame(frame,(nexts_pipe_x[0],nexts_pipe_y[0]),pipe_up["img"],(0,0,255))
            x_end_pipe= nexts_pipe_x[0]+pipe_up["img"].shape[1]
            up_end_y =nexts_pipe_y[0]+pipe_up["img"].shape[0]
        else:
            return [],frame
    else:
        return [],frame


    ## find pipe_down
    result = cv2.matchTemplate(img,pipe_down["img"], cv2.TM_CCOEFF_NORMED)
    yloc,xloc = np.where(result>=pipe_down["threshold"])

    if len(xloc)>0:
        filter = xloc>x_bird
        nexts_pipe_x = xloc[filter]
        nexts_pipe_y = yloc[filter]
        nexts_pipe_xinds = nexts_pipe_x.argsort()
        nexts_pipe_x = nexts_pipe_x[nexts_pipe_xinds[::-1]]
        nexts_pipe_y = nexts_pipe_y[nexts_pipe_xinds[::-1]]
        if len(nexts_pipe_y)>0:
            update_frame(frame,(nexts_pipe_x[0],nexts_pipe_y[0]),pipe_down["img"],(0,0,255))
            y_center_hole = up_end_y+((nexts_pipe_y[0]-up_end_y)/2)
        else:
            return [],frame
    else:
        return [],frame
    
    return [x_end_pipe-x_bird,y_center_hole-y_center_bird],frame





