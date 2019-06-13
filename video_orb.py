import cv2

training_image = cv2.imread('./data/training_image.jpg')
training_image = cv2.cvtColor(training_image, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(200, 2.0)
kp_train, des_train = orb.detectAndCompute(training_image, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

cam = cv2.VideoCapture('./data/test_video.mp4')
print(cam.isOpened())

while cam.isOpened():
    ret_val, query_image = cam.read()
    if ret_val:
        query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
        kp_query, des_query = orb.detectAndCompute(query_image, None)

        matches = bf.match(des_train, des_query)
        matches = sorted(matches, key = lambda x: x.distance)
        result = cv2.drawMatches(training_image, kp_train, query_image, kp_query, matches[:200], query_image, flags = 2)

        cv2.imshow('Video',result)
    else:
        print('no frames available')

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
