
import cv2
import numpy as np
import math
import matplotlib.patches as mpatches
from tqdm import tqdm

def getBBs(row , th=0.0):
    x = row.PredictionString
    z = x.split()
    bbs = []
    
    for i in range(0,len(z),6):
        r = []
        
        if float(z[i+1]) < th :
            continue
            
        r.append(int(z[i]))  # Class ID
        r.append(float(z[i+1]))   # Confidence
        r.extend([int(i) for i in z[i+2:i+6]])  # BB boundaries
        
        bbs.append(r)
        
    return bbs




def draw_bboxes(img, tl, br, rgb, label="", conf=-1, label_location="tl", opacity=0.1, line_thickness=0):

    rect = np.uint8(np.ones((br[1]-tl[1], br[0]-tl[0], 3))*rgb)
    sub_combo = cv2.addWeighted(img[tl[1]:br[1],tl[0]:br[0],:], 1-opacity, rect, opacity, 1.0)    
    img[tl[1]:br[1],tl[0]:br[0],:] = sub_combo

    if line_thickness>0:
        img = cv2.rectangle(img, tuple(tl), tuple(br), rgb, line_thickness)
        
    
    
    if label:
        # DEFAULTS
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE = 1.666
        FONT_THICKNESS = 3
        FONT_LINE_TYPE = cv2.LINE_AA
        
        if type(label)==str:
            LABEL = label.upper().replace(" ", "_")
        else:
            LABEL = f"CLASS_{label:02}"
            
        if conf != -1:
            conf = conf*100
            conf = math.trunc(conf)
            LABEL = LABEL + f", {str(conf)}%"
        
        text_width, text_height = cv2.getTextSize(LABEL, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        
        label_origin = {"tl":tl, "br":br, "tr":(br[0],tl[1]), "bl":(tl[0],br[1])}[label_location]
        label_offset = {
            "tl":np.array([0, -10]), "br":np.array([-text_width, text_height+10]), 
            "tr":np.array([-text_width, -10]), "bl":np.array([0, text_height+10])
        }[label_location]
        img = cv2.putText(img, LABEL, tuple(label_origin+label_offset), 
                          FONT, FONT_SCALE, rgb, FONT_THICKNESS, FONT_LINE_TYPE)
        
    
    return img



def draw_bboxes_only(img, tl, br, rgb, opacity=0.1, line_thickness=0):
    
    rect = np.uint8(np.ones((br[1]-tl[1], br[0]-tl[0], 3))*rgb)
    sub_combo = cv2.addWeighted(img[tl[1]:br[1],tl[0]:br[0],:], 1-opacity, rect, opacity, 1.0)    
    img[tl[1]:br[1],tl[0]:br[0],:] = sub_combo

    if line_thickness>0:
        img = cv2.rectangle(img, tuple(tl), tuple(br), rgb, line_thickness)
        
    return img


