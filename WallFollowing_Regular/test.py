'''
Author       : Karen Li
Date         : 2023-08-12 13:15:26
LastEditors  : Karen Li
LastEditTime : 2023-08-12 18:21:07
FilePath     : /WallFollowing_V2/test.py
Description  : 
'''

import cv2
from State import State

if __name__ == "__main__":
    # Read an image from a jpg file
    img1 = cv2.imread("image1.jpg", cv2.IMREAD_ANYCOLOR)
    img2 = cv2.imread("image2.jpg", cv2.IMREAD_ANYCOLOR)

    # Check if the image was successfully read
    if img1 is None:
        print("Can't read the image file")
        exit()
    if img2 is None:
        print("Can't read the image file")
        exit()

    # Display the image
    test_state_1 = State(img1)
    test_state_2 = State(img2)
    q,t = test_state_1.get_match_coordinate(test_state_2)
    x1, y1, sx1, sy1 = test_state_1.confidence_ellipse(q)
    x2, y2, sx2, sy2 = test_state_2.confidence_ellipse(t)
    print(x1, y1, sx1, sy1)
    print(x2, y2, sx2, sy2)





    cv2.imshow("Image1", img1)
    cv2.imshow("Image2", img2)
    cv2.waitKey(0)  # This will pause the execution until a key is pressed
    cv2.destroyAllWindows()  # Close the image window
