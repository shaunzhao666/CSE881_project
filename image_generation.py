import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.feature import blob_log
import numpy as np
from io import BytesIO
import os
import math
import copy
import json

def resize_image(image, dimensions=(770, 500)):
    return cv2.resize(image, dimensions)

def add_white_border(img, thickness=5, axis=1):
    # """
    # Adds a white border to an image. 
    # axis=1 is for horizontal and axis=0 is for vertical.
    # """
    num_channels = img.shape[2] if len(img.shape) == 3 else 1  # Check if image is colored or grayscale
    color_value = 255
    
    if axis == 1:  # Horizontal
        if num_channels == 3:
            border = np.ones((img.shape[0], thickness, 3), dtype=np.uint8) * color_value
        else:
            border = np.ones((img.shape[0], thickness), dtype=np.uint8) * color_value
    else:  # Vertical
        if num_channels == 3:
            border = np.ones((thickness, img.shape[1], 3), dtype=np.uint8) * color_value
        else:
            border = np.ones((thickness, img.shape[1]), dtype=np.uint8) * color_value
            
    return border

# For blue sky with cloud or without cloud, the binary separation cannot be smaller than 170 or larger than 220
def binariseImage(img, thresholds):
    # Thresholding the Image to binarise it
    output_thresh = []
    for threshold in thresholds:
        # a pixel value greater than threshold would be set to 255.
        # ret is the threshold we use; thresh is the image after apply threshold
        ret,thresh = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)
        output_thresh.append(thresh)

    return output_thresh

def contour_detection(img):
    # Check if the image is not grayscale
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img  # Image is already grayscale

    # Check if the grayscale image is already binary by counting unique values
    if len(np.unique(gray)) > 2:
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    else:
        binary = gray

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def apply_dilation(image, kernel_size=(3,3)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def iterateArea(contours):
    # Finding the coordinates of the contours and their area
    coordinates = {}
    for i in range(len(contours)):
        cnt = contours[i] # cnt: the current contour
        M = cv2.moments(cnt)
        # average x, y - coordinate (weighted by pixel intensity)
        #print(M) # Mij = sumx,sumy(x^i y^j I(x, y)) # For binary images  -> area for m00
        cx = int(M['m10']/M['m00'])                 #-> cx: (sum of x for each pixcel)/#pixels
        cy = int(M['m01']/M['m00'])                 #-> cy: (sum of y for each pixcel)/#pixels
        area = cv2.contourArea(cnt)

        # It is to ensure that no two contours have the exact same area. 
        # If two contours have the same area,  small value of 0.0001 is added to the area 
        if area in coordinates:
            coordinates[area+0.0001] = (cx, cy)
        else:
            coordinates[area] = (cx, cy)
        
    sorted_area = []
    for i in reversed(sorted(coordinates.keys())):  
        sorted_area.append(i) #Just areas of the stars

    # Creating coordinates for each star in the constellation
    x = []
    y = []
    for area in sorted_area:
        x.append(coordinates[area][0])
        y.append(coordinates[area][1])
    
    return sorted_area, x, y

def Areablob(img):
    blobs = blob_log(img, max_sigma=50, num_sigma=50, threshold=.1)
    y = blobs[:, 0]
    x = blobs[:, 1]
    r = blobs[:, 2]
    return r, x, y
               
def process_image(image, is_type = 'NAP', con = 'Gemini'):
    # Resize the image
    resized = resize_image(image)
    
    ### Convert to grayscale if the image has 3 channels ###
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized

    # Concatenating the original and grayscale images side by side horizontally
    border = add_white_border(resized)
    concatenated = cv2.hconcat([resized, border, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)])  
    #cv2.imshow('Original and Grayscale', concatenated)
    
    if is_type == 'NAP':    
        ### Apply binary threshold ###
        # The binary separation cannot be smaller than 170 or larger than 220
        # Don't forget to print out the image.shape and image.dtype to see whether they are compatible
        thresh = binariseImage(gray, [150, 190, 210])
    elif is_type == 'AP':
        if con == 'Orion':
            gray = cv2.medianBlur(gray, 3)
        thresh = binariseImage(gray, [90, 110, 130])
    elif is_type == 'camera_night':
        thresh = binariseImage(gray, [90, 110, 130])
    else:
        thresh = binariseImage(gray, [150, 190, 210])

    # Create horizontal borders
    horiz_border_1 = add_white_border(gray, axis=1)
    horiz_border_2 = add_white_border(thresh[0], axis=1)

    # First, concatenate images horizontally
    top_row = cv2.hconcat([gray, horiz_border_1, thresh[0]])
    bottom_row = cv2.hconcat([thresh[1], horiz_border_2, thresh[2]])
    # Create a vertical border
    vert_border = add_white_border(top_row, axis=0)
    # Then, concatenate the results vertically
    binary_gray = cv2.vconcat([top_row, vert_border, bottom_row])
    #cv2.imshow('Binary_threshold', binary_gray) 

    ### Median Filter ###
    thresh0 = thresh[0]
    thresh1 = thresh[1]
    thresh2 = thresh[2]
    stars03 = cv2.medianBlur(thresh0, 3)
    stars05 = cv2.medianBlur(thresh0, 5)
    horiz_border_1 = add_white_border(stars03, axis=1)
    horiz_border_2 = add_white_border(stars05, axis=1)
    stars13 = cv2.medianBlur(thresh1, 3) 
    stars15 = cv2.medianBlur(thresh1, 5)  
    stars23 = cv2.medianBlur(thresh2, 3) 
    stars25 = cv2.medianBlur(thresh2, 5)  
    top_row = cv2.hconcat([stars03, horiz_border_1, stars05])
    middle_row = cv2.hconcat([stars13, horiz_border_2, stars15])
    bottom_row = cv2.hconcat([stars23, horiz_border_2, stars25])
    vert_border = add_white_border(top_row, axis=0)   
    median_image = cv2.vconcat([top_row, vert_border, middle_row, vert_border, bottom_row])
    #cv2.imshow("Apply noise",median_image)

    if is_type == 'NAP':    
        enhanced = apply_dilation(stars13, kernel_size=(2,2))
    elif is_type == 'AP':
        if con == 'Gemini' or con == 'Perseus':
            enhanced = apply_dilation(thresh2, kernel_size=(3,3))
        elif con == 'Orion':
            enhanced = apply_dilation(stars23, kernel_size=(3,3))
        else:
            enhanced = apply_dilation(stars03, kernel_size=(3,3))
    elif is_type == 'camera_night' or is_type == 'camera_longexpo':
        enhanced = apply_dilation(thresh1, kernel_size=(2,2))
    else:
        enhanced = apply_dilation(thresh1, kernel_size=(3,3))

    ### Finding contours ###
    contours = contour_detection(enhanced)  
    # Visualization or any other processing can be done here
    black_background = np.zeros_like(resized)
    output_contour = cv2.drawContours(black_background, contours, -1, (255, 255, 255), -1)  #-1 fills the contour interior
    #cv2.imshow("contors", output_contour)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ### Remove all zero area contours ###
    final_contours = removeZero(contours)
    # final_contours = 0

    return final_contours

### Remove all zero area contours ###
def removeZero(contours):         
    final_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 0]
    #print("Number of Contours = " + str(len(final_contours)))

    return final_contours

def rotate_point(x, y, angle, center):
    #"""Rotate a point by a given angle around a given center."""
    s = np.sin(np.radians(-angle))
    c = np.cos(np.radians(-angle))

    # translate point back to origin:
    x -= center[0]
    y -= center[1]

    # rotate point
    x_new = x * c - y * s + center[0]
    y_new = x * s + y * c + center[1]

    return x_new, y_new

def rotateContour(resized, filename, contours, angles, outputsize):
    ### Finding Area and location of stars
    areas, x_values, y_values = iterateArea(contours)

    # Input image dimensions
    rows, cols = resized.shape[:2]
    center = (cols // 2, rows // 2)

    for idx, angle in enumerate (angles):
        ### Scale factor calculation###
        # Get the rotation matrix for contour points
        M = cv2.getRotationMatrix2D(center, angle, 1)       
        rotated_contours = [cv2.transform(cnt, M) for cnt in contours]
        
        # Get bounding box for rotated contours
        all_rotated_contours = np.vstack(rotated_contours)
        min_x = np.min(all_rotated_contours[:, :, 0])
        max_x = np.max(all_rotated_contours[:, :, 0])
        min_y = np.min(all_rotated_contours[:, :, 1])
        max_y = np.max(all_rotated_contours[:, :, 1])
        w = max_x - min_x
        h = max_y - min_y

        # Calculate the scaling required for width and height
        scale_factor = min(outputsize/w, outputsize/h) - 0.1

        dx = (outputsize - w * scale_factor) / 2.0 - min_x * scale_factor
        dy = (outputsize - h * scale_factor) / 2.0 - min_y * scale_factor

        ### Output image prepare###
        black_background = np.zeros((outputsize, outputsize))
        fig, ax = plt.subplots(figsize=(outputsize/100.0, outputsize/100.0), dpi=100)
        ax.imshow(black_background, cmap='gray')

        ### Rotation of the stars with the current scale factor ###
        for area, cx, cy in zip(areas, x_values, y_values):
            # Rotate and scale the centroid of each star
            rotated_cx, rotated_cy = rotate_point(cx, cy, angle, center)
            scaled_cx = rotated_cx * scale_factor + dx
            scaled_cy = rotated_cy * scale_factor + dy

            # Scale the area to get the new radius
            scaled_area = area * (scale_factor ** 2)
            scaled_radius = np.sqrt(scaled_area / np.pi)
            
            # Draw a white circle on the black background
            circle = patches.Circle((scaled_cx, scaled_cy), scaled_radius, color='white', fill=True)
            ax.add_patch(circle)

            # # Directly get image using cv2.circle will make stars change shape a lot
            # cv2.circle(black_background, (int(scaled_cx), int(scaled_cy)), int(scaled_radius), (255, 255, 255), -1)  # -1 to fill the circle

        # Remove axis
        ax.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        # # Method 1 of coverting to image
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())

        # # Save the image to an in-memory file
        # buf = BytesIO()
        # plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0, dpi=100)
        # buf.seek(0)

        # # Read the image back using OpenCV
        # img_arr = np.asarray(bytearray(buf.read()), dtype=np.uint8)
        # output_image = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)

        # buf.close()   

        #Method2: Convert figure to OpenCV image
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # If want to get the RGB image can simply remove this line of code.
        output_image = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY) # For getting the gray scale image.

        # Display
        # window_name = f"Rotated by {angle} degrees"
        # cv2.imshow(window_name, output_image)
        # cv2.waitKey(0)
        # cv2.destroyWindow(window_name)

        # Check if the directory exists, and if not, create it
        directory_path = os.path.join('./TrainingSamples', filename)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        # Construct the save path with the index suffix
        save_filename = filename + "_" + str(idx) + ".png"
        save_path = os.path.join(directory_path, save_filename)
        cv2.imwrite(save_path, output_image)

def sampleCoord(imageName):
    # Get the directory of the current script
    samples_directory = os.path.join('./TrainingSamples', imageName)
    results_file_path = os.path.join(samples_directory, 'results.json')
    results = {} # Ensure that results is initialized.

    allowed_extensions = ['.jpg', '.jpeg', '.png']
    

    # Iterate through each file in the template directory to process one template at a time
    for filename in os.listdir(samples_directory):         
        # Get the contours of the input image
        img_path = os.path.join(samples_directory, filename)
        img = cv2.imread(img_path)

        # Check if the file has an allowed image extension
        if os.path.splitext(filename)[1].lower() not in allowed_extensions:
            continue

        # Check if image reading was successful
        if img is None:
            print(f"Failed to load {img_path}. Skipping.")
            continue
        
        contours = contour_detection(img) 

        ### Remove all zero area contours ###
        final_contours = removeZero(contours)
        area, x, y = iterateArea(final_contours)

        # Save the extracted data in the dictionary using filename as the key
        results[filename[:-4]] = {
            "area": area,
            "x": x,
            "y": y
        }

        # Save results to a JSON file
        with open(results_file_path, 'w') as file:
            json.dump(results, file)

    return results

if __name__ == "__main__":
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    constellation = ['UrsaMajor', 'Cassiopeia', 'Gemini', 'Orion', 'Perseus', 'SummerTriangle']
    condition = ['NAP', 'AP', 'camera_night', 'camera_afternoon', 'camera_longexpo']
    allowed_extensions = ['.jpg', '.jpeg', '.png']

    for confolder in constellation:
        for type in condition:
            data_path = os.path.join(script_dir, 'Constellation_images', confolder, type)

             # Check if the folder exists, if not, skip to the next one
            if not os.path.exists(data_path):
                print(f"Folder {data_path} does not exist. Skipping.")
                continue

            try:
                for filename in os.listdir(data_path):
                    # Check if the file has an allowed image extension
                    if os.path.splitext(filename)[1].lower() not in allowed_extensions:
                        continue         

                    img_path = os.path.join(data_path, filename)
                    img = cv2.imread(img_path)

                    # Check if image reading was successful
                    if img is None:
                        print(f"Failed to load {img_path}. Skipping.")
                        continue

                    contours = process_image(img, is_type = type, con = confolder)

                    angles = [i for i in range(0, 360, 15)]
                    rotateContour(img, filename[:-4], contours, angles, 256)

                    sampleCoord(filename[:-4])

            except Exception as e:
                print(f"An error occurred while processing {data_path}: {e}")
                







    
    

    