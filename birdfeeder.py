import numpy as np
import cv2
import random
import os
import logging
from twython import Twython
from twython import TwythonError

from auth import (
        consumer_key,
        consumer_secret,
        access_token,
        access_token_secret
)
twitter = Twython(
        consumer_key,
        consumer_secret,
        access_token,
        access_token_secret
)


confidence_thr = 0.5
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
birdfeeder_dir=os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(filename=birdfeeder_dir+'/log/birdfeederAI.log', level=logging.DEBUG, format='%(asctime)s %(message)s')
mobilenet_dir=birdfeeder_dir+'/MobileNet-SSD/'
net = cv2.dnn.readNetFromCaffe(mobilenet_dir+ 'deploy.prototxt' , mobilenet_dir+ 'mobilenet_iter_73000.caffemodel')
blob=None

def applySSD(image):
    logging.debug('Started applySSD function')
    global blob
    mybird = bool(False)
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_thr:
            idx = int(detections[0,0,i,1])
            if CLASSES[idx]=="bird":
                mybird=bool(True)
    return mybird

def birdRatio(videoName):
    logging.debug('Started birdRatio function')
    totalBirdFrames = 1
    totalFrames = 1
    vc2 = cv2.VideoCapture(videoName)
    if vc2.isOpened():
        rval2,frame2 = vc2.read()
    else:
        rval2 = False
        logging.error('Can\'t open video for checking frames for birds')
        exit()

    while rval2:
        logging.debug('birdRatio video frames loop '+str(totalBirdFrames)+"/"+str(totalFrames))
        birdinFrame = applySSD(frame2)
        rval2, frame2 = vc2.read()
        if (birdinFrame):
            totalBirdFrames = totalBirdFrames + 1
        totalFrames = totalFrames + 1

    vc2.release()
    return totalBirdFrames/totalFrames

videoLength=3*60*1000
randomsec=random.randint(0,videoLength)


#vc = cv2.VideoCapture(birdfeeder_dir+"/birds_video.mp4")
# If you want to record birds using your camera comment the above line and uncomment the below line. If you want to find birds in a video uncomment the line above and comment the line below :)
vc = cv2.VideoCapture(0)
vc.set(cv2.CAP_PROP_POS_MSEC, randomsec)
if vc.isOpened():
    width = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = vc.get(cv2.CAP_PROP_FPS)
    fcount = vc.get(cv2.CAP_PROP_FRAME_COUNT)
else:
    logging.error('Can\'t open video')
    exit()

recording= False
framerecorded = 0
framecounter = 0
birdinFrame=False
fourcc = cv2.VideoWriter_fourcc(*'h264')
#out = cv2.VideoWriter('output.mp4',fourcc,20.0,(640,480))
out = cv2.VideoWriter(birdfeeder_dir+'/output.mp4',fourcc,fps,(int(width),int(height)))

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
    (h, w) = frame.shape[0] , frame.shape[1]
else:
    rval = False

logging.debug('Started main loop')
while rval:
    #You enter this loop once per frame
    rval, frame = vc.read()
    #uncomment the below line if you need to flip the camera upside down.
    frame = cv2.flip(frame,-1)
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
    framecounter = framecounter + 1
    if (framecounter > 60):
    # Write frame to disk every 60 frames so we can see what the camera is seeing    
        framecounter = 0
        cv2.imwrite(birdfeeder_dir+"/webserver/currentframe.jpg",frame)
    if (birdinFrame==False):
        #Check if this frame has a bird in it
        birdinFrame= applySSD(frame)
    if (birdinFrame== True and recording== False):
        #You have detected the first bird in a frame, start recording
        logging.info('Saw a bird, started recording video')
        recording=True
    if (recording == True):
        #write the frame to file keep track of how many frames you have saved.
        framerecorded = framerecorded + 1 
        out.write(frame)
    if (framerecorded > 200):
        #after 200 frames stop recording
        logging.info('Checking recorded video')
        recording = False
        birdinFrame=False
        framerecorded = 0
        out.release()
        filename = birdfeeder_dir+"/output.mp4"
        logging.debug('About to check frame for birds')
        birdsinvideo= birdRatio(filename)
        logging.debug('percentage of bird in video: '+str(birdsinvideo))
        if (birdsinvideo> 0.50):
            # if the recorded video has more than 50% of frames with a bird in it then tweet it
            logging.info('Tweeting bird video')
            video = open(filename,'rb')
            try:
                response = twitter.upload_video(media=video, media_type='video/mp4', media_category='tweet_video', check_progress=True)
                twitter.update_status(status='birdfeeder 0.5', media_ids=[response['media_id']])
            except TwythonError as e:
                logging.error('Twitter error:'+str(e))
            birdsinvideo=0
            video.close()
        randomsec=random.randint(0,videoLength)
        vc.set(cv2.CAP_PROP_POS_MSEC, randomsec)
        os.remove(birdfeeder_dir+'/output.mp4')
        out = cv2.VideoWriter(birdfeeder_dir+'/output.mp4',fourcc,fps,(int(width),int(height)))



vc.release()
