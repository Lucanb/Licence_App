'''
Author: Manohar Mukku
Date: 06.12.2018
Desc: Watershed Segmentation algorithm
GitHub: https://github.com/manoharmukku/watershed-segmentation
'''

import sys
import cv2
import numpy
import os

def neighbourhood(image, x, y):
    # Save the neighbourhood pixel's values in a dictionary
    neighbour_region_numbers = {}
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i == 0 and j == 0):
                continue
            if (x+i < 0 or y+j < 0): # If coordinates out of image range, skip
                continue
            if (x+i >= image.shape[0] or y+j >= image.shape[1]): # If coordinates out of image range, skip
                continue
            if (neighbour_region_numbers.get(image[x+i][y+j]) == None):
                neighbour_region_numbers[image[x+i][y+j]] = 1 # Create entry in dictionary if not already present
            else:
                neighbour_region_numbers[image[x+i][y+j]] += 1 # Increase count in dictionary if already present

    # Remove the key - 0 if exists
    if (neighbour_region_numbers.get(0) != None):
        del neighbour_region_numbers[0]

    # Get the keys of the dictionary
    keys = list(neighbour_region_numbers)

    # Sort the keys for ease of checking
    keys.sort()

    if (keys[0] == -1):
        if (len(keys) == 1): # Separate region
            return -1
        elif (len(keys) == 2): # Part of another region
            return keys[1]
        else: # Watershed
            return 0
    else:
        if (len(keys) == 1): # Part of another region
            return keys[0]
        else: # Watershed
            return 0

def watershed_segmentation(image):
    # Create a list of pixel intensities along with their coordinates
    intensity_list = []
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            # Append the tuple (pixel_intensity, xy-coord) to the end of the list
            intensity_list.append((image[x][y], (x, y)))

    # Sort the list with respect to their pixel intensities, in ascending order
    intensity_list.sort()

    # Create an empty segmented_image numpy ndarray initialized to -1's
    segmented_image = numpy.full(image.shape, -1, dtype=int)

    # Iterate the intensity_list in ascending order and update the segmented image
    region_number = 0
    for i in range(len(intensity_list)):
        # Print iteration number in terminal for clarity
        sys.stdout.write("\rPixel {} of {}...".format(i, len(intensity_list)))
        sys.stdout.flush()

        # Get the pixel intensity and the x,y coordinates
        intensity = intensity_list[i][0]
        x = intensity_list[i][1][0]
        y = intensity_list[i][1][1]

        # Get the region number of the current pixel's region by checking its neighbouring pixels
        region_status = neighbourhood(segmented_image, x, y)

        # Assign region number (or) watershed accordingly, at pixel (x, y) of the segmented image
        if (region_status == -1): # Separate region
            region_number += 1
            segmented_image[x][y] = region_number
        elif (region_status == 0): # Watershed
            segmented_image[x][y] = 0
        else: # Part of another region
            segmented_image[x][y] = region_status

    # Return the segmented image
    return segmented_image

def preprocess_images_in_folder(input_folder, output_folder):
    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all images in the input folder
    images = [img for img in os.listdir(input_folder) if img.endswith(".jpg")]

    for img_name in images:
        input_path = os.path.join(input_folder, img_name)
        output_path = os.path.join(output_folder, img_name)  # Save in the output folder

        # Read the input image in grayscale
        img = cv2.imread(input_path, 0)
        if img is None:
            print(f"Cannot open {input_path}")
            continue
        
        # Perform segmentation using the watershed algorithm
        segmented_image = watershed_segmentation(img)

        # Save the segmented image
        cv2.imwrite(output_path, segmented_image)
        print(f"Processed and saved: {output_path}")

if __name__ == "__main__":

    img = cv2.imread('/home/lolluckestar/Desktop/Licence_App/dataset/img1/ISIC_0024306.jpg', 0)

    if (img is None):
        # print ("Cannot open {} image".format(argv[0]))
        print ("Make sure you provide the correct image path")
        sys.exit(2)

    # Perform segmentation using watershed_segmentation on the input image
    segmented_image = watershed_segmentation(img)

    # Save the segmented image as png to disk
    cv2.imwrite("test/target.png", segmented_image)

    # Show the segmented image and original image side by side
    input_image = cv2.resize(img, (0,0), None, 0.5, 0.5)
    seg_image = cv2.resize(cv2.imread("test/target.png", 0), (0,0), None, 0.5, 0.5)
    numpy_horiz = numpy.hstack((input_image, seg_image))
    cv2.imshow('Input image ------------------------ Segmented image', numpy_horiz)
    cv2.waitKey(0)
    #     print("Usage: python script.py <path_to_input_images_folder> <path_to_output_images_folder>")
    #     sys.exit(1)
    
    # input_folder_path = sys.argv[1]
    # output_folder_path = sys.argv[2]
    # preprocess_images_in_folder(input_folder_path, output_folder_path)