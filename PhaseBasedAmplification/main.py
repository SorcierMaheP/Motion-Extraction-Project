"""
PyTorch implementation of Phase Based Motion Magnification

Based on code by: https://github.com/itberrios/phase_based

Current approach loads all frames into memory so this won't work for large videos

For the Batch size selection, we try to avoid the need to zero pad batchsize, since
it leads to undesireable behavior in the processing. It is easier to taylor both the 
batchsize and scale factor such that we don't need zero padding.


 Sources:
    Papers: 
        - http://people.csail.mit.edu/nwadhwa/phase-video/phase-video.pdf
        - https://www.cns.nyu.edu/pub/eero/simoncelli95b.pdf
        - http://www.cns.nyu.edu/pub/eero/portilla99-reprint.pdf
    Code: 
        - http://people.csail.mit.edu/nwadhwa/phase-video/
        - https://github.com/LabForComputationalVision/matlabPyrTools
        - https://github.com/LabForComputationalVision/pyrtools
    Misc:
        - https://rafat.github.io/sites/wavebook/advanced/steer.html
        - http://www.cns.nyu.edu/~eero/steerpyr/
        - https://www.cns.nyu.edu/pub/lcv/simoncelli90.pdf
        - http://www.cns.nyu.edu/~eero/imrep-course/Slides/07-multiScale.pdf
"""

import os
import sys
import datetime
import re
import numpy as np
from PIL import Image
import cv2 as cv
import torch

from steerablePyramid import SteerablePyramid, SuboctaveSP
from phaseBasedProcessing import PhaseBased
from phaseUtils import *

# ==========================================================================================
# constants
EPS = 1e-6  # factor to avoid division by 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================================================================
# start main program

def main(colorsp):

    # Add parameters here:
    videoPath = "./SampleVideos/baby.avi"
    phaseMagnification = 10.0
    freqLowerbound = 0.04
    freqHigherBound = 0.5

    # options:  "luma1", "luma3", "gray", "yiq", "rgb"
    colorspace = colorsp
    # options: "full_octave", "half_octave", "smooth_half_octave", "smooth_quarter_octave"
    pyramidType = "smooth_quarter_octave"

    #  phase processing parameters
    sigma = 5
    attenuate = True
    # setting to -1 defaults to video default sample frequency
    sampleFrequency = -1
    ref_idx = 0
    # reduce the scale factor to reduce memory consumption
    scaleFactor = 1
    # batch size for CUDA parallelization
    batchSize = 2
    saveDirectory = "./../Temp"
    # save the result as a gif or not
    saveGif = False

    # ======================================================================================
    # start the clock once the args are received
    tic = cv.getTickCount()

    # ======================================================================================
    # Process input filepaths
    if not os.path.exists(videoPath):
        print(f"\nInput video path: {videoPath} not found! exiting \n")
        sys.exit()

    if not saveDirectory:
        saveDirectory = os.path.dirname(videoPath)
    elif not os.path.exists(saveDirectory):
        saveDirectory = os.path.dirname(videoPath)
        print(f"\nSave Directory not found, "
              "using default input video directory instead \n")

    video_name = re.search("\w*(?=\.\w*)", videoPath).group()
    video_output = f"{video_name}_{colorspace}_{int(phaseMagnification)}x.mp4"
    videoSavePath = os.path.join(saveDirectory, video_output)

    print(f"\nProcessing {video_name} "
          f"and saving results to {videoSavePath} \n")
    print(f"Device found: {DEVICE} \n")

    # ======================================================================================
    # Get frames and sample rate (frameRate) from input video

    # get forward and inverse colorspace functions
    # inverse colorspace obtains frames back in BGR representation
    if colorspace == "luma1":
        def colorspaceFunction(x): return bgr2yiq(x)[:, :, 0]

        def invColorspaceFunction(x): return cv.cvtColor(
            cv.normalize(x, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1),
            cv.COLOR_GRAY2BGR)

    elif colorspace == "luma3" or colorspace == "yiq":
        colorspaceFunction = bgr2yiq

        def invColorspaceFunction(x): return cv.cvtColor(
            cv.normalize(yiq2rgb(x), None, 0, 255,
                         cv.NORM_MINMAX, cv.CV_8UC3),
            cv.COLOR_RGB2BGR)

    elif colorspace == "gray":
        def colorspaceFunction(x): return cv.cvtColor(x, cv.COLOR_BGR2GRAY)

        def invColorspaceFunction(x): return cv.cvtColor(
            cv.normalize(x, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1),
            cv.COLOR_GRAY2BGR)

    elif colorspace == "rgb":
        def colorspaceFunction(x): return cv.cvtColor(x, cv.COLOR_BGR2RGB)

        def invColorspaceFunction(x): return cv.cvtColor(
            cv.normalize(x, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC3),
            cv.COLOR_RGB2BGR)

    # get scaled video frames in proper colorspace and sample frequency frameRate
    frames, videoSampleFreq = get_video(videoPath,
                                        scaleFactor,
                                        colorspaceFunction)

    # ======================================================================================
    # Prepare for processing

    # get reference frame info
    refFrame = frames[ref_idx]
    refHeight, refWidth = refFrame.shape[:2]

    # video length
    nFrames = len(frames)

    # get sample frequency frameRate, use input sample frequency if valid
    if sampleFrequency > 0.0:
        frameRate = sampleFrequency
        print(f"Detected Sample Frequency: {videoSampleFreq} \n")
        print(
            f"Sample Frequency overriden with input!: frameRate = {frameRate} \n")
    else:
        frameRate = videoSampleFreq
        print(f"Detected Sample Frequency: frameRate = {frameRate} \n")

    # Get Bandpass Filter Transfer function
    transferFunction = bandpass_filter(freqLowerbound,
                                       freqHigherBound,
                                       frameRate,
                                       nFrames,
                                       DEVICE)

    # Get Complex Steerable Pyramid Object
    max_depth = int(
        np.floor(np.log2(np.min(np.array((refWidth, refHeight))))) - 2)

    if pyramidType == "full_octave":
        csp = SteerablePyramid(depth=max_depth,
                               orientations=4,
                               filters_per_octave=1,
                               twidth=1.0,
                               complex_pyr=True)

    elif pyramidType == "half_octave":
        csp = SteerablePyramid(depth=max_depth,
                               orientations=8,
                               filters_per_octave=2,
                               twidth=0.75,
                               complex_pyr=True)

    elif pyramidType == "smooth_half_octave":
        csp = SuboctaveSP(depth=max_depth,
                          orientations=8,
                          filters_per_octave=2,
                          cos_order=6,
                          complex_pyr=True)

    elif pyramidType == "smooth_quarter_octave":
        csp = SuboctaveSP(depth=max_depth,
                          orientations=8,
                          filters_per_octave=4,
                          cos_order=6,
                          complex_pyr=True)

    # get Complex Steerable Pyramid Filters
    filters, crops = csp.get_filters(refHeight, refWidth, cropped=False)
    filtersTensor = torch.tensor(
        np.array(filters)).type(torch.float32).to(DEVICE)

    # TODO: ensure selected Batch Size is compatible with the number of filters
    # we don't want to rely on zero padding
    if (filtersTensor.shape[0] % batchSize) != 0:
        print(f"WARNING! Selected Batch size: {batchSize} might "
              f"not be compatible with the number of "
              f"Filters: {filtersTensor.shape[0]}! \n")

    # Compute DFT for all Video Frames
    framesTensor = torch.tensor(np.array(frames)).type(torch.float32) \
        .to(DEVICE)

    # ======================================================================================
    # Begin Motion Magnification processing

    print(f"Performing Phase Based Motion Magnification \n")

    phaseAmplified = PhaseBased(sigma, transferFunction, phaseMagnification, attenuate,
                                ref_idx, batchSize, DEVICE, EPS)

    if colorspace == "yiq" or colorspace == "rgb":
        # process each channel individually
        resultVideo = torch.zeros_like(framesTensor).to(DEVICE)
        for c in range(framesTensor.shape[-1]):
            video_dft = get_fft2_batch(framesTensor[:, :, :, c]).to(DEVICE)
            resultVideo[:, :, :, c] = phaseAmplified.process_single_channel(framesTensor[:, :, :, c],
                                                                            filtersTensor,
                                                                            video_dft)

    elif colorspace == "luma3":
        # process single Luma channel and add back to full color image
        resultVideo = framesTensor.clone()
        video_dft = get_fft2_batch(framesTensor[:, :, :, 0]).to(DEVICE)
        resultVideo[:, :, :, 0] = \
            phaseAmplified.process_single_channel(framesTensor[:, :, :, 0],
                                                  filtersTensor,
                                                  video_dft)

    else:
        # process single channel
        video_dft = get_fft2_batch(framesTensor).to(DEVICE)
        resultVideo = phaseAmplified.process_single_channel(framesTensor,
                                                            filtersTensor,
                                                            video_dft)

    # remove from CUDA and convert to numpy
    resultVideo = resultVideo.cpu().numpy()

    # ======================================================================================
    # Process results

    # get stacked side-by-side comparison frames
    originalHeight = int(refHeight/scaleFactor)
    originalWidth = int(refWidth/scaleFactor)
    middle = np.zeros((originalHeight, 3, 3)).astype(np.uint8)

    stackedFrames = []

    for vid_idx in range(nFrames):

        # get BGR frames
        bgrFrame = invColorspaceFunction(frames[vid_idx])
        bgrProcessed = invColorspaceFunction(resultVideo[vid_idx])

        # resize to original shape
        bgrFrame = cv.resize(bgrFrame, (originalWidth, originalHeight))
        bgrProcessed = cv.resize(bgrProcessed, (originalWidth, originalHeight))

        # stack frames
        stacked = np.hstack((bgrFrame,
                             middle,
                             bgrProcessed))

        stackedFrames.append(stacked)

    # ======================================================================================
    # make video
    # get width and height for stacked video frames
    stackedHeight, stackedWidth, _ = stackedFrames[-1].shape

    # save to mp4
    out = cv.VideoWriter(videoSavePath,
                         cv.VideoWriter_fourcc(*'MP4V'),
                         int(np.round(videoSampleFreq)),
                         (stackedWidth, stackedHeight))

    for frame in stackedFrames:
        out.write(frame)

    out.release()
    del out

    print(f"Result video saved to: {videoSavePath} \n")

    # ======================================================================================
    # make GIF if desired
    if saveGif:

        # replace video extension with ".gif"
        gif_save_path = re.sub("\.\w+(?<=\w)", ".gif", videoSavePath)

        print(f"Saving GIF to: {gif_save_path} \n")

        # size back down for GIF
        stackedHeight = int(stackedHeight*scaleFactor)
        stackedWidth = int(stackedWidth*scaleFactor)

        # accumulate PIL image objects
        pil_images = []
        for img in stackedFrames:
            img = cv.cvtColor(
                cv.resize(img, (stackedWidth, stackedHeight)), cv.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(img))

        # create GIF
        pil_images[0].save(gif_save_path,
                           format="GIF",
                           append_images=pil_images,
                           save_all=True,
                           duration=50,  # duration that each frame is displayed
                           loop=0)

    # ======================================================================================
    # end of processing

    # get time elapsed in Hours : Minutes : Seconds
    toc = cv.getTickCount()
    timeElapsed = (toc - tic) / cv.getTickFrequency()
    timeElapsed = str(datetime.timedelta(seconds=timeElapsed))

    print("Motion Magnification processing complete! \n")
    print(f"Time Elapsed (HH:MM:SS): {timeElapsed} \n")


col = ["gray", "yiq", "rgb"]

for colour in col:
    main(colour)
