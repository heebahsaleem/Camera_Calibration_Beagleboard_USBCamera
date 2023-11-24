import numpy as np
import cv2
import glob
import yaml

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

objpoints = []
imgpoints = [] 

images = glob.glob(r'/home/debian/person-tracker/images/*.jpg')
path = '/home/debian/person-tracker/images/'


found = 0
for fname in images:  
    img = cv2.imread(fname)
    #print(images[im_i])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (7,7), None)

    if ret == True:
        objpoints.append(objp)  
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        img = cv2.drawChessboardCorners(img, (7,7), corners2, ret)
        found += 1
        cv2.imshow('img', img)
        cv2.waitKey(500)
        image_name = path + '/calibresult' + str(found) + '.png' #saving the output image
        cv2.imwrite(image_name, img)


print("Number of images used for calibration: ", found)


cv2.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

data = {'Camera Matrix [camera_matrix]': np.asarray(mtx).tolist(),
        'Distortion Coefficient [dist_coeff]': np.asarray(dist).tolist(),
        'Rotation Vectors [rvecs]': np.asarray(rvecs).tolist(),
        'Translation [tvecs]': np.asarray(tvecs).tolist(),
        }
with open("/home/debian/person-tracker/images/calibration_matrix.yaml", "w") as f:
    yaml.dump(data, f)

# h, w = images.shape[:2]
# newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h),1,(w,h))

# dst = cv2.undistort(images, mtx, dist, None, newCameraMatrix)

# x,y,w,h=roi
# dst=dst[y:y+h, x:x+w]
# cv2.imwrite('/home/debian/Downloads/sonnet-team-person-tracker-22a2e11a2941/images/result1.jpg', dst)

#mapx,mapy