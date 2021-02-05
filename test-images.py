import cv2
from skimage.metrics import structural_similarity
import imutils
import argparse
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
    help="first input image")
ap.add_argument("-s", "--second", required=True,
    help="second image")
ap.add_argument("-of", "--output-folder", required=True, 
    help="Folder to store processed images in")
args = vars(ap.parse_args())

# load the two input images
imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])
# convert the images to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# compute blur for each image
blurA = cv2.Laplacian(grayA, cv2.CV_64F).var()
blurB = cv2.Laplacian(grayB, cv2.CV_64F).var()
# show the image
cv2.putText(imageA, "{}: {:.2f}".format("Blur", blurA), (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
cv2.putText(imageB, "{}: {:.2f}".format("Blur", blurB), (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = structural_similarity(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8") #diff image
print("SSIM: {}".format(score)) #prints similarity score


# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 100, 255,
    cv2.THRESH_BINARY_INV)[1]
cv2.putText(thresh, "{}: {:.2f}".format("Similarity Score", score), (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX, 0.8, 127, 3)

# write out processed images to folder
os.chdir(args["output_folder"])
cv2.imwrite(args["first"], imageA)
cv2.imwrite(args["second"], imageB)
firstFile = args["first"]
diffFilename = firstFile[:firstFile.find("no")-1] + firstFile[firstFile.find("cover")+5:-4] + "_diff.png"
print(diffFilename)
cv2.imwrite(diffFilename, thresh)

# show the output images
cv2.imshow("Original", imageA)
cv2.imshow("Modified", imageB)
cv2.imshow("Diff", diff)
cv2.imshow("Thresh",thresh)
cv2.waitKey(0)
