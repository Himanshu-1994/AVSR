import cv2

def detect(path):
  #  img = cv2.imread(path)
    img = frame   
    cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")
    #cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    rects = cascade.detectMultiScale(img, 1.3, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (20,20))

    if len(rects) == 0:
        return [], img
    rects[:, 2:] += rects[:, :2]
    return rects, img

def box(rects, img,i):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), (127, 255, 0), 2)
    cv2.imwrite('./face2/detected'+str(i)+'.jpg', img);

cap = cv2.VideoCapture('v1.mpg')

for i in range(50):
    print i
    ret, frame = cap.read()
    print ret
    cv2.imwrite('./face_datanew/detected'+str(i)+'.jpg', frame);

   # rects, img = detect(frame)
   # box(rects, img,i)
