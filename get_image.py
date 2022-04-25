from process_image import AI_inputs
import numpy as np
import cv2
from mss import mss
def save_video(img_array,size):
    out = cv2.VideoWriter('project.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 1, size)
 
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print("finalizado")

with mss() as sct:
    monitor_number = 1
    mon = sct.monitors[monitor_number]
    monitor = {"top": mon["top"]+176, "left": mon["left"] +719, "width": 480, "height": 640}
    # img_array = []
    while True:
        screenshot = np.array(sct.grab(monitor))
        # screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        inputs,frame = AI_inputs(screenshot)
        print(inputs)
        cv2.imshow('Frame',frame)
        # img_array.append(frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # save_video(img_array,(480,640))
