import xml.etree.ElementTree as ET  
import cv2
import os
import numpy as np
import time
import copy
#fast loops
from numba import njit 

def NPparseLocXML(imageXMLPath: str) -> dict:
    tree = ET.parse(imageXMLPath)
    root = tree.getroot()
    marker_data = root[1][1:]
    
    type_loc = {}
    for marker_type in marker_data:
        type = marker_type[0].text
        locs = np.array([[int(marker[0].text), int(marker[1].text)] for marker in marker_type.findall('Marker')], dtype = int)
        if locs.size:
            type_loc[type] = locs
        
    return type_loc


def NPpointBGR(img, boundaryPoints: np.ndarray, totalPoints: int) -> np.ndarray:
    return np.array([(img[boundaryPoints[(i*2)+1], boundaryPoints[i*2], 0], 
                      img[boundaryPoints[(i*2)+1], boundaryPoints[i*2], 1],
                      img[boundaryPoints[(i*2)+1], boundaryPoints[i*2], 2]) for i in range(totalPoints)], dtype=np.int16)


def NPsignifBGRChange(img, boundaryPoints: np.ndarray, startBGR: np.ndarray, BGRthresh: int, totalPoints: int) -> np.ndarray:
    '''
        Current method of detecting significant change is by seeing if difference of 
        current BGR and the start BGR is above a given BGR threshhold
    '''
    currBGR = NPpointBGR(img, boundaryPoints, totalPoints)
    diffBGR = np.abs(currBGR - startBGR)

    aboveThresh = np.greater(diffBGR, BGRthresh)
    return np.repeat(np.any(aboveThresh, axis=1), 2)
    
    # NOTE: Trying to see if hsv could be better than BGR
    # currHSV = np.apply_along_axis(BGRtoHSV, 1, currBGR)
    # startHSV = np.apply_along_axis(BGRtoHSV, 1, startBGR)
    # diffHSV = np.abs(currHSV - startHSV)
    
    # aboveThresh = np.greater(diffHSV, BGRthresh)
    # return np.repeat(np.any(aboveThresh, axis=1), 2)


def NPpointWalk(points: np.ndarray, img, totalPoints: int, step: int, iters: int, BGRthresh: int, maxB: np.ndarray) -> np.ndarray:
    minB = 0
    startBGR = NPpointBGR(img, points, totalPoints)
    # NOTE: this is what defines the points to test (total/2 must equal totalPoints)
    stepdirs = np.array([0, -step, 0, step, step, 0, -step, 0])
    
    for i in range(iters):
        # take a step
        points += stepdirs
        
        # determine if any (x,y) in points goes below or above the image bounds
        low = np.less(points, minB)
        high = np.greater(points, maxB)
        
        # check bgr of current step
        abvBGRthresh = NPsignifBGRChange(img, points, startBGR, BGRthresh, totalPoints)
        
        # bool mask to check if any (x,y) goes over the min or max image bounds
        o = (low | high | abvBGRthresh)
        
        # if any point is invalid
        if np.any(o):
            # repeat g twice (x, y), if x or y is true then they are both true
            # r is a bool mask that makes each x, y true if either one of them is true
            g = np.any(o.reshape(4, 2), axis=1)
            r = np.repeat(g, 2) 
            
            # m is a mask that determines which step no longer results in a valid point
            m = stepdirs * r
            
            # step backward if point is invalid
            points -= m 
            stepdirs -= m
            
        # Return points because no more steps can be taken
        if not np.any(stepdirs):
            return points
    
    # Delete after adding the BGR check
    return points



# Assuming 4 points
def NPdrawBoxes(points: np.ndarray, drawImg, totalPoints: int, boxColor: tuple = (0, 0, 255), boxThickness: int = 1):
    # if 4 points then order is [[down], [up], [right], [left]]
    pointCords = np.array([[points[i*2], points[(i*2)+1]] for i in range(totalPoints)])
    
    # Draw box
    topLeft = (pointCords[3][0], pointCords[1][1])
    bottomRight = (pointCords[2][0], pointCords[0][1])
    cv2.rectangle(drawImg, topLeft, bottomRight, color = boxColor, thickness = boxThickness)



# NOTE: simply changing totalPoints will not increase all points in the algorithm
def NPboundingBoxes(specLocs: dict, imagePath: str, BGRdev: int =70, iters: int = 300, step: int = 3, totalPoints: int = 4) -> None:
    orig = cv2.imread(imagePath)
    # copiedImg = copy.deepcopy(orig)
    h, w, _ = orig.shape
    
    # largest values for x and y boundary points
    maxB = np.tile([w, h], totalPoints)
    walkArgs = [orig, totalPoints, step, iters, BGRdev, maxB]
    
    """
        Below creates bounding edges for all species on the image
        Having a smaller BGRdev seems to work better for darker creatures and a larger BGRdev seems better for lighter creatures
        You can set "iters" to be as large as possible because it can stop early if there are no changes from a step
        dark:   dev = 25, step = 2
        light:  dev = 70, step = 3
        
    """
    
    for type, locs in specLocs.items():
        # Copying image so separate images can be created for each of the species in the image
        drawImg = copy.deepcopy(orig)
        startPoints = np.tile(locs, totalPoints)
        allPoints = np.apply_along_axis(NPpointWalk, 1, startPoints, *walkArgs)
        
        # Draw bounding boxes
        boxArgs = [drawImg, totalPoints]
        np.apply_along_axis(NPdrawBoxes, 1, allPoints, *boxArgs)
        
        # show testing points that were used to create bounding box
        for points in allPoints:
            cv2.circle(drawImg, (points[0], points[1]), 3, [0, 255, 0], -1)
            cv2.circle(drawImg, (points[2], points[3]), 3, [0, 255, 0], -1)
            cv2.circle(drawImg, (points[4], points[5]), 3, [0, 255, 0], -1)
            cv2.circle(drawImg, (points[6], points[7]), 3, [0, 255, 0], -1)

        file =  type + 'BOUNDS.jpeg'
        cv2.imwrite(file, drawImg)
        

if __name__ == "__main__":

    imagedir = 'ACq1/'
    image_locsPath = 'ACq1/2013-08-19_YB-ACq1.xml'
    imagePath = 'ACq1/2013-08-19_YB-ACq1.jpeg'
    origImage = cv2.imread(imagePath)
  
    
    # parse xml
    locs_start = time.time()
    spec_locs = NPparseLocXML(image_locsPath)
    locs_end = time.time()
    
    # ==================================================
    # create points for each species
    # createImgLocs(imagePath, image_locsPath, locs = spec_locs, size = 4, locBGR = [200, 200, 0])
    
    # ==================================================
    # bounding boxes
    box_start = time.time()
    NPboundingBoxes(spec_locs, imagePath)
    box_end = time.time()
    
    
    # # ===================================================
    print(f"NUMP Locs time: {locs_end - locs_start}")
    print(f"NUMP Box time: {box_end - box_start}")