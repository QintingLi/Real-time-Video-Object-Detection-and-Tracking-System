import sys

import cv2
import numpy as np

# Tracking: since pedestrian move at a relatively low speed, the box of last frame and next frame should be similar in both shape and position
# Most close: According to perspective principle, more closer, it seems bigger in 2D section, so we could calculate size of the box to estimate distance

# Resize to 640*480
def resize_vga(frame):
    k=min(480/frame.shape[0],640/frame.shape[1])
    return cv2.resize(frame,(0,0),fx=k,fy=k)

# Own cc algorithm based of bfs, 8 connectivity
def my_connected_components(data):
    n,m=data.shape[0],data.shape[1]
    flag=0 #start from 0
    label=[[-1 for _ in range(m)] for _ in range(n)]
    record=[]
    aa=2
    for i in range(0,n,aa+2):
        for j in range(0,m,aa+2):
            # white and not been labeled, then start bfs
            if data[i][j]>0 and label[i][j]==-1:
                # 1 for left up  2 for right bottom
                x1,y1,x2,y2=i,j,i,j
                label[i][j]=flag
                p=[[i,j]]
                while p:
                    i,j=p.pop(0)
                    x1,y1,x2,y2=min(x1,i),min(y1,j),max(x2,i),max(y2,j)
                    for nx in range(i-aa,i+aa+1):
                        for ny in range(j-aa,j+aa+1):
                            if nx!=i and ny!=j and nx>=0 and nx<n and ny>=0 and ny<m and data[nx][ny]>0 and label[nx][ny]==-1:
                                label[nx][ny]=flag
                                p.append([nx,ny])
                flag+=1
                record.append([x1,y1,x2,y2])
    return label,record

def classifier(cnt,record):
    human,car,other=0,0,0
    for x1,y1,x2,y2 in record:
        h,w=x2-x1,y2-y1
        if h>50 or w>50:
            car+=1
        elif h>30 or w>30:
            human+=1
        elif h>15 and w>15:
            other+=1
    a1,a2,a3='s'*(human>1),'s'*(human>1),'s'*(human>1)
    print('Frame '+str(cnt).zfill(4)+": "+str(human+car+other)+" objects  ("+str(human)+" person"+a1+", "+str(car)+" car"+a2+" and "+str(other)+" other"+a3+")")

def estimated_background(cap):
    ids = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)
    frames = []
    for fid in ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        _, frame = cap.read()
        frames.append(frame)
    return resize_vga(np.median(frames, axis=0).astype(dtype=np.uint8))

# use Hanhattan distance to track box
def similar_box(pos1,pos2,delta):
    x1,y1,x2,y2=pos1
    p1,q1,p2,q2=pos2
    if abs(x1-p1)+abs(y1-q1)<delta and abs(x2-p2)+abs(y2-q2)<delta:
        return True
    return False

def getarg():
    args=sys.argv
    if(len(args)!=3):
        print('Invalid command!')
        sys.exit()
    if args[1]=='-b':
        return 1,args[2]
    elif args[1]=='-d':
        return 2,args[2]
    else:
        print('Invalid command!')
        sys.exit()

#------------------------------------------------------
flag,filename=getarg()
if flag==1:
    # Read video
    cap = cv2.VideoCapture(filename)
    # Get estimated background
    bg = estimated_background(cap)
    # Morphological kernal
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    kernal2=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8))
    # Gaussian Mixture background
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    cnt=0
    if not cap.isOpened():
        print("File open Error!")
        sys.exit()
    while cap.isOpened():
        cnt+=1
        ret, frame = cap.read()
        if ret:
            frame=resize_vga(frame)
            # Got estimated background img
            raw_mask = fgbg.apply(frame)
            # Remove noise by morphological open operation
            mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
            # Get connected components
            label,record=my_connected_components(mask)
            # Classify components
            classifier(cnt,record)
            # Get detected object
            mask2=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal2)
            dect_obj=cv2.bitwise_and(cv2.merge([mask2,mask2,mask2]),frame)
            # m1=np.hstack([frame,bg])
            # m2=np.hstack([cv2.merge([raw_mask,raw_mask,raw_mask]),dect_obj])
            ans=np.vstack([np.hstack([frame,bg]),np.hstack([cv2.merge([raw_mask,raw_mask,raw_mask]),dect_obj])])
            cv2.imshow('Task 1',ans)
            k = cv2.waitKey(100) & 0xff
            if k == 27:
                break
        else:
            print('Runtime Error!')
            sys.exit()
    cap.release()
    cv2.destroyAllWindows()
elif flag==2:
    # load the COCO class names
    with open('object_detection_classes_coco.txt', 'r') as f:
        class_names = f.read().split('\n')

    # From orage to red, means closer
    COLORS = [(0,0,255),(0,140,255),(0,165,255)]

    # load the DNN model
    model = cv2.dnn.readNet(model='frozen_inference_graph.pb',
                            config='ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt', 
                            framework='TensorFlow')



    # capture the video
    cap = cv2.VideoCapture(filename)
    # frame counter
    cnt=0;
    # id for pedestrian
    id=0
    # save position of each box in last frame
    record=[]
    if not cap.isOpened():
        print("File open Error!")
        sys.exit()
    # detect objects in each frame of the video
    while cap.isOpened():
        ret, frame = cap.read()
        frame=resize_vga(frame)
        f1,f2,f3=frame.copy(),frame.copy(),frame.copy()
        if ret:
            image = frame
            image_height, image_width, _ = image.shape
            # create blob from image
            blob = cv2.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123), 
                                        swapRB=True)
            model.setInput(blob)
            output = model.forward()
            # tmp cache for positions in present frame
            tmp=[]
            for detection in output[0, 0, :, :]:
                # extract the confidence of the detection
                confidence = detection[2]
                # if the result is possibly a person
                if confidence > .4 and class_names[int(detection[1])-1]=='person':
                    # get the bounding box coordinates
                    x1,y1 = detection[3] * image_width, detection[4] * image_height
                    # get the bounding box width and height
                    x2,y2 = detection[5] * image_width, detection[6] * image_height

                    # Mark dectected objects with boxes
                    cv2.rectangle(f1, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), thickness=2)

                    # Track moving objects by comparing to positions in last frame
                    label=0
                    for pos in record:
                        if similar_box([x1,y1,x2,y2],pos[:4],30):
                            label=pos[4]
                            break
                    if label==0:
                        id+=1
                        label=id
                    cv2.rectangle(f2, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), thickness=2)
                    cv2.putText(f2,str(label), (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    tmp.append([x1,y1,x2,y2,label])

            record=tmp[:]

            # get nearest 3 boxes
            record.sort(key=lambda x:((x[2]-x[0])*(x[3]-x[1])),reverse=True)
            for i in range(min(3,len(record))):
                cv2.rectangle(f3, (int(record[i][0]),int(record[i][1])),(int(record[i][2]),int(record[i][3])), COLORS[i], thickness=2)

            ans=np.vstack([np.hstack([image,f1]),np.hstack([f2,f3])])
            cv2.imshow('Task 2',ans)
            k = cv2.waitKey(100) & 0xff
            if k == 27:
                break
        else:
            print('Runtime Error!')
            sys.exit()
        cnt+=1
    cap.release()
    cv2.destroyAllWindows()
