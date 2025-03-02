import cv2
import numpy as np
import matplotlib.pyplot as plt 

def stitch_images(image1_path, image2_path, output_path):
    
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)
    image1_kp = cv2.drawKeypoints(image1, keypoints1, None, color=(0, 255, 0), flags=0)
    image2_kp = cv2.drawKeypoints(image2, keypoints2, None, color=(0, 255, 0), flags=0)

    # Show keypoints
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image1_kp, cv2.COLOR_BGR2RGB))
    plt.title("Keypoints in Image 1")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image2_kp, cv2.COLOR_BGR2RGB))
    plt.title("Keypoints in Image 2")
    plt.axis("off")
    plt.show()
    
    # Match features using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    match_img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.title("Feature Matches Between Images")
    plt.axis("off")
    plt.show()

    # Extract location of good matches
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Compute homography matrix
    H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
    
    # Get size of output panorama
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]
    
    # Warp image2 to align with image1
    warped_image2 = cv2.warpPerspective(image2, H, (width1 + width2, height1))
    
    # Place image1 onto the stitched canvas
    warped_image2[0:height1, 0:width1] = image1
    
    # Convert to grayscale and threshold to find non-black regions
    gray_warped = cv2.cvtColor(warped_image2, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_warped, 1, 255, cv2.THRESH_BINARY)
    
    # Find contours and crop to bounding box
    x, y, w, h = cv2.boundingRect(mask)
    stitched_cropped = warped_image2[y:y+h, x:x+w]
    
    cv2.imwrite(output_path, stitched_cropped)
    
stitch_images("panorama1.jpeg", "panorama2.jpeg", "stitched_panorama.jpeg")
