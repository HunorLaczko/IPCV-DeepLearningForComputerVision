#https://docs.opencv.org/4.4.0/d6/d00/tutorial_py_root.html
#https://docs.opencv.org/4.4.0/dd/d43/tutorial_py_video_display.html "Playing Video from file"

from __future__ import print_function

import argparse
import numpy as np
import cv2

import matplotlib.pyplot as plt


def computeMSE(prev, curr):
    mse = np.sum((prev - curr) ** 2) / (prev.shape[0] * prev.shape[1])
    return mse
    
def computePSNR(mse):
    psnr = 10 * np.log10(255 ** 2 / mse + 1e-10)
    return psnr

def computeEntropy(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]) / (img.shape[0] * img.shape[1])
    ent = -np.sum(hist * np.log2(hist + 1e-10))
    return ent
    
def computeErrorImage(im1, im2):
    res = np.minimum(255, np.maximum(0, im1 - im2 + 128))
    return res

def computeOpticalFlow1(prev, curr):
    flow = cv2.calcOpticalFlowFarneback(curr, prev, flow=None, pyr_scale=0.5, levels=3, winsize=20, iterations=15, poly_n=5, poly_sigma=1.2, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    
    # optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
    # flow = optical_flow.calc(curr, prev, None)
    
    # flow = cv2.optflow.calcOpticalFlowDenseRLOF(curr, prev, None)

    return flow

def computeCompensatedFrame(prev, flow):
    h, w = flow.shape[:2]
    map = flow.copy()
    map[:,:,0] += np.arange(w)
    map[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(prev, map, None, cv2.INTER_LINEAR)
    return res
    
def computeGME(flow):
    src = np.zeros_like(flow)
    h, w = flow.shape[:2]
    c = np.array([w/2, h/2])
    src[:,:,0] += np.arange(w)
    src[:,:,1] += np.arange(h)[:,np.newaxis]
    src -= c
    srcPts = src.reshape((h*w, 2)) 

    dst = src + flow
    dstPts = dst.reshape((h*w, 2)) 

    homography, mask = cv2.findHomography(srcPts, dstPts, method=cv2.RANSAC, ransacReprojThreshold=3)
    gme = cv2.perspectiveTransform(src, homography) - src

    return gme

def computeGMEError(flow, gme):
    err = np.sqrt(np.sum((flow - gme) ** 2, axis=2))
    return err

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Read video file')
    parser.add_argument('video', help='input video filename')
    parser.add_argument('deltaT', help='input deltaT between frames', type=int)

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)

    if (cap.isOpened() == False):
        print("ERROR: unable to open video: " + args.video)
        quit()

    deltaT = args.deltaT
    video = args.video

    previousFrames=[]
    frameNumbers = []
    mses = []
    psnrs = []
    mse0s = []
    psnr0s = []
    ents = []
    ent0s = []
    entEs = []

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # flowVideo = cv2.VideoWriter(video + '_flow.avi', fourcc , fps, (width, height))
    # gmeVideo = cv2.VideoWriter(video + '_gme.avi',fourcc, fps, (width, height))
    # gmeErrorVideo = cv2.VideoWriter(video + '_gmError.avi',fourcc, fps, (width, height), isColor = False)
    # compensatedVideo = cv2.VideoWriter(video + '_compensated.avi',fourcc, fps, (width, height), isColor = False)
    # imErr0Video = cv2.VideoWriter(video + '_imErr0.avi',fourcc, fps, (width, height), isColor = False)
    # imErrVideo = cv2.VideoWriter(video + '_imErr.avi',fourcc, fps, (width, height), isColor = False)


    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        print(video + ": " + str(i) + " / " + str(totalFrames))
        
        if (ret==False):
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if (len(previousFrames) >= deltaT):
            prev = previousFrames.pop(0)

            flow = computeOpticalFlow1(prev, gray)
        
        
            compensatedFrame = computeCompensatedFrame(prev, flow)

            #cv2.imshow('compensated', compensatedFrame)
        
            imErr0 = computeErrorImage(prev, gray)
            imErr = computeErrorImage(compensatedFrame, gray)
            
            #cv2.imshow('imErr0', imErr0)
            #cv2.imshow('imErr', imErr)

            
            mse0 = computeMSE(prev, gray)
            psnr0 = computePSNR(mse0)
            mse = computeMSE(compensatedFrame, gray)
            psnr = computePSNR(mse)
            ent = computeEntropy(gray)
            ent0 = computeEntropy(imErr0)
            entE = computeEntropy(imErr)
        
            frameNumbers.append(i)
            mses.append(mse)
            psnrs.append(psnr)
            mse0s.append(mse0)
            psnr0s.append(psnr0)
            ents.append(ent)
            ent0s.append(ent0)
            entEs.append(entE)
        
            
            gme = computeGME(flow)
            
            gmeError = computeGMEError(flow, gme)
            
            cv2.imshow('flow', draw_flow(gray, flow))
            cv2.imshow('gme', draw_flow(gray, gme))
            cv2.imshow('gmeError', gmeError)
            # flowVideo.write(draw_flow(gray, flow))
            # gmeVideo.write(draw_flow(gray, gme))
            # gmeErrorVideo.write(cv2.normalize(gmeError.astype('uint8'), gmeError.astype('uint8'), 0, 255, cv2.NORM_MINMAX))
            # compensatedVideo.write(compensatedFrame)
            # imErr0Video.write(imErr0)
            # imErrVideo.write(imErr)
        
        previousFrames.append(gray.copy())
        i+=1

        cv2.imshow('frame', gray)
        
        cv2.waitKey(1)

    # flowVideo.release()
    # gmeVideo.release()
    # gmeErrorVideo.release()
    # compensatedVideo.release()
    # imErr0Video.release()
    # imErrVideo.release()

    plt.plot(frameNumbers, mse0s, label='MSE0')
    plt.plot(frameNumbers, mses, label='MSE')
    plt.xlabel('frames')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('MSE0 vs MSE')
    plt.savefig("mse.png")
    plt.show()

    plt.figure()
    plt.plot(frameNumbers, ents, label='Entropy')
    plt.plot(frameNumbers, ent0s, label='Entropy0')
    plt.plot(frameNumbers, entEs, label='EntropyE')
    plt.xlabel('frames')
    plt.ylabel('Entropy')
    plt.legend()
    plt.title('Entropy vs Entropy0 vs EntropyE')
    plt.savefig(video + "_entropy_all.png")
    plt.show()     
    
    plt.plot(frameNumbers, psnr0s, label='PSNR0')
    plt.plot(frameNumbers, psnrs, label='PSNR')
    plt.xlabel('frames')
    plt.ylabel('PSNR')
    plt.legend()
    plt.title('PSNR0 vs PSNR')
    plt.savefig("psnr.png")
    plt.show()
    
    
    cap.release()
    cv2.destroyAllWindows()
