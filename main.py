'''
THIS ASSIGNMENT WAS DONE ON PyCharm WITH PYTHON VERSION 3.6.6 AND THE PACKAGE
opencv-contrib-python VERSION 3.4.2.16
'''
import cv2


# Function to show image
def cv2_imshow(text, image):
    cv2.imshow(text, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Function to convert image to grayscale
def grayScale(image):
    output = image.copy()
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    return gray


# Feature Extraction with SURF
def SURF(image, gray):
    # Create SURF Feature Detector object
    hessianThreshold = 500
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold)

    # Only features with hessian larger than hessianThreshold are retained by the detector
    keypoints, descriptors = surf.detectAndCompute(gray, None)
    print("Number of key points Detected: ", len(keypoints))

    # Draw key points on input image
    image = cv2.drawKeypoints(gray, keypoints, image)

    cv2_imshow('Feature Method - SURF', image)


# Feature Extraction with ORB
def ORB(image, gray):
    # Create ORB Feature Detector object
    orb = cv2.ORB_create()

    # Determine key points
    keypoints = orb.detect(gray, None)

    # Get the descriptors
    keypoints, descriptors = orb.compute(gray, keypoints)
    print("Number of keypoints Detected: ", len(keypoints))

    # Draw key points on input image
    image = cv2.drawKeypoints(gray, keypoints, image)

    cv2_imshow('Feature Method - ORB', image)


def main():
    # Read images
    image1 = cv2.imread('./victoria.jpg')
    image2 = cv2.imread('./victoria2.jpg')

    # Convert images to grayscale
    gray1 = grayScale(image1)
    gray2 = grayScale(image2)

    # Get user input on which Feature Detector to use
    featureOptions = ["A", "B"]
    imageOptions = ["A", "B"]
    userInputFeature = ""
    userInputImage = ""

    while userInputFeature.upper() not in featureOptions:
        print("\nFeature Detectors available: A: SURF, B: ORB")
        userInputFeature = input("Please select a Feature Detector: (A/B) ")
    while userInputImage.upper() not in imageOptions:
        print("\nImages available: A: victoria.jpg, B: victoria2.jpg")
        userInputImage = input("Please select an image: (A/B) ")

    if userInputFeature.upper() == "A":
        # SURF function
        if userInputImage.upper() == "A":
            SURF(image1, gray1)
        elif userInputImage.upper() == "B":
            SURF(image2, gray2)

    elif userInputFeature.upper() == "B":
        # ORB function
        if userInputImage.upper() == "A":
            ORB(image1, gray1)
        elif userInputImage.upper() == "B":
            ORB(image2, gray2)


main()
