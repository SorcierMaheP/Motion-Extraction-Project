#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv
import platform


# In[2]:


# list to store previous frames, needed to compute temporal changes
# using it as a buffer here, with length = prevFrameNumber
frameList = []

# number of previous frames to process:
prevFrameNumber = 3

currentFrameNumber = 0

# treshold for the final output:
treshold = 150

# detects os of user:
userPlatform = platform.system()

# selecting the videocapture backend in Linux is necessary in some cases
# change 0 to whatever webcam id is required
if userPlatform == "Linux":
    cam = (0, cv.CAP_V4L2)
elif userPlatform in ["Windows", "Darwin"]:
    cam = 0


# Posy method:
# 1) Duplicate video
# 2) Invert video, and decrease opacity to about 50%
# 3) "Shift the time of the video"?
#
# <!-- trying a simple differnce between frames for now -->

# In[12]:


# create a video capture object,
# sourcing it from a webcam with id 0 (the default one)
# replace the 0 with a filepath if using a video file instead
cap = cv.VideoCapture(*cam)

# set video fps to 60, because why not?
cap.set(cv.CAP_PROP_FPS, 60)

_, previousFrame = cap.read()
previousFrame = cv.cvtColor(previousFrame, cv.COLOR_BGR2GRAY)

while True:

    # capture each frame from the video feed -----------------------------------------------------------------------------------------------------------
    _, frame = cap.read()

    if frame is None:
        break

    # todo for visualising only, remove later
    cv.namedWindow("live input", cv.WINDOW_NORMAL)
    cv.imshow("live input", frame)

    #  process each frame here:   ----------------------------------------------------------------------------------------------------------------------

    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame = cv.GaussianBlur(frame, (5, 5), 0)

    currentFrameNumber += 1

    frameList.append(frame)

    if currentFrameNumber >= prevFrameNumber:

        currentFrameNumber = 0
        previousFrame = frameList.pop(0)
        previousFrame = cv.bitwise_not(previousFrame)
        frameList.clear()

    frame = cv.addWeighted(frame, 0.5, previousFrame, 0.5, 0)

    # use the treshold for using it as some sort of special effect by addign it back to the original
    # _, frame = cv.threshold(frame, treshold, 255, cv.THRESH_BINARY)

    # display the video in a window, per frame:

    cv.namedWindow("video", cv.WINDOW_NORMAL)
    cv.imshow("video", frame)

    # quit the loop if 'q' is pressed
    #! close the window by typing "q", closing it normally will cause the kernel to crash,
    #! among other things

    if cv.waitKey(1) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
