import numpy as np
import cv2
import os
import sys
import argparse
import math

parser = argparse.ArgumentParser(description='dart detection')
parser.add_argument('-name', '-n', type=str, default='Dartboard/dart0.jpg')
args = parser.parse_args()

# Global variables
cascade_name = "Dartboardcascade/cascade.xml"


def detectAndDisplay(frame):
    detectedBoxes = []
    # 1. Prepare Image by turning it into Grayscale and normalising lighting
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    # 2. Perform Viola-Jones Object Detection
    darts = model.detectMultiScale(frame_gray, scaleFactor=1.01, minNeighbors=48, flags=0, minSize=(40, 40),
                                   maxSize=(230, 230))
    # 3. Print number of Faces found

    print("number of detected darts :" + str(len(darts)))
    # 4. Draw box around faces found
    for i in range(0, len(darts)):
        detectedBoxes.append(darts[i])
        start_point = (darts[i][0], darts[i][1])
        end_point = (darts[i][0] + darts[i][2], darts[i][1] + darts[i][3])
        colour = (0, 255, 0)
        thickness = 2
        frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)
    return detectedBoxes


def readGroundtruth(filename='GroundTruth.txt'):
    bounding_boxes = []
    # read bounding boxes as ground truth
    with open(filename) as f:
        # read each line in text file
        for line in f.readlines():
            content_list = line.split(",")
            img_name = content_list[0]
            x = float(content_list[1])
            y = float(content_list[2])
            width = float(content_list[3])
            height = float(content_list[4])
            bounding_boxes.append((img_name, x, y, width, height))
    return bounding_boxes


def draw_bounding_boxes(frame, bounding_boxes, imageNameEdit):
    # Draw each bounding box on the frame
    for img_name, x, y, width, height in bounding_boxes:
        if img_name == imageNameEdit:
            start_point = (int(x), int(y))
            end_point = (int(x + width), int(y + height))
            color = (0, 0, 255)  # Red in BGR
            thickness = 2
            frame = cv2.rectangle(frame, start_point, end_point, color, thickness)


def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the coordinates of the intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    # Check if there is an intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the union area
    union_area = w1 * h1 + w2 * h2 - intersection_area

    # Calculate the IOU
    iou = intersection_area / union_area

    return iou


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'Coordinates: ({x}, {y})')


def sobel_x(input):
    sobelxOutput = np.zeros((input.shape[0], input.shape[1]), dtype=np.float32)
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernelRadiusX = 1
    kernelRadiusY = 1
    paddedInput = cv2.copyMakeBorder(input, kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
                                     cv2.BORDER_REPLICATE)

    for i in range(0, input.shape[0]):
        for j in range(0, input.shape[1]):
            patch = paddedInput[i:i + 3, j:j + 3]
            sum = (np.multiply(patch, kernel)).sum()
            sobelxOutput[i, j] = sum

    return sobelxOutput


def sobel_y(input):
    sobelyOutput = np.zeros((input.shape[0], input.shape[1]), dtype=np.float32)
    kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    kernelRadiusX = 1
    kernelRadiusY = 1
    paddedInput = cv2.copyMakeBorder(input, kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
                                     cv2.BORDER_REPLICATE)

    for i in range(0, input.shape[0]):
        for j in range(0, input.shape[1]):
            patch = paddedInput[i:i + 3, j:j + 3]
            sum = (np.multiply(patch, kernel)).sum()
            sobelyOutput[i, j] = sum

    return sobelyOutput


def gradient_mag(input):
    x = sobel_x(input)  # Assuming sobel_x(input) correctly computes the x-direction gradient
    y = sobel_y(input)  # Assuming sobel_y(input) correctly computes the y-direction gradient
    result = np.sqrt(x ** 2 + y ** 2)  # Calculate the magnitude of the gradient
    return result


def gradient_ori(input):
    x = sobel_x(input)  # Assuming sobel_x(input) correctly computes the x-direction gradient
    y = sobel_y(input)
    result = np.arctan2(y, x + math.exp(-10))
    # Normalize the result to the range [0, 255]
    # normalized_result = ((result - result.min()) / (result.max() - result.min())) * 255
    # Convert the result to uint8 data type for visualization
    # result_float32 = normalized_result.astype(np.float32)
    # change_to_pi= (result/180)*math.pi
    return result


def thresholdPixel(input, ts):
    for i in range(0, input.shape[0]):
        for j in range(0, input.shape[1]):
            if input[i][j] > ts:
                input[i][j] = 255
            else:
                input[i][j] = 0
    return input


def merge_close_circles(detected_circles, min_distance=20):
    new_detected_circles = []
    skip_indices = set()  # Used to record circles that have already been processed

    for i in range(len(detected_circles)):
        if i in skip_indices:
            continue

        current_circle = detected_circles[i]
        overlapping_circles = []

        for j in range(len(detected_circles)):
            if i != j:
                other_circle = detected_circles[j]
                distance = np.sqrt((current_circle[0] - other_circle[0]) ** 2 +
                                   (current_circle[1] - other_circle[1]) ** 2)

                if distance < min_distance:
                    overlapping_circles.append(j)

        if not overlapping_circles:
            new_detected_circles.append(current_circle)
        else:
            # Add the current circle to skip_indices as it overlaps with others
            skip_indices.add(i)
            # Add the first overlapping circle to new_detected_circles and to skip_indices
            new_detected_circles.append(detected_circles[overlapping_circles[0]])
            skip_indices.update(overlapping_circles)

    return new_detected_circles



def hough_circle_detection(gradient_magnitude, gradient_orientation, ts, th, min_radius, max_radius):
    # Step 1: Threshold the gradient magnitude image.
    thresholded_image = thresholdPixel(gradient_magnitude, ts)

    # Step 2: Create a 3D Hough Space array.
    hough_space = np.zeros((gradient_magnitude.shape[0], gradient_magnitude.shape[1], max_radius - min_radius + 1),
                           dtype=np.uint32)

    # Step 3: Voting in Hough Space.
    for y in range(gradient_magnitude.shape[0]):
        for x in range(gradient_magnitude.shape[1]):
            if thresholded_image[y, x] != 0:
                for radius in range(min_radius, max_radius + 1):
                    center_x = int(x + radius * np.cos(gradient_orientation[y, x]))
                    center_y = int(y + radius * np.sin(gradient_orientation[y, x]))

                    if 0 <= center_x < gradient_magnitude.shape[1] - radius and 0 <= center_y < \
                            gradient_magnitude.shape[0] - radius:
                        hough_space[center_y, center_x, radius - min_radius] += 1

                    center_x0 = int(x - radius * np.cos(gradient_orientation[y, x]))
                    center_y0 = int(y - radius * np.sin(gradient_orientation[y, x]))
                    if 0 <= radius <= center_x0 < gradient_magnitude.shape[1] - radius and 0 <= radius <= center_y0 < \
                            gradient_magnitude.shape[0] - radius:
                        hough_space[center_y0, center_x0, radius - min_radius] += 1
    # Step 4: Thresholding the Hough Space.
    thresholded_hough = np.zeros((hough_space.shape[0], hough_space.shape[1], hough_space.shape[2]), dtype=np.uint32)
    for y in range(hough_space.shape[0]):
        for x in range(hough_space.shape[1]):
            for z in range(hough_space.shape[2]):
                if hough_space[y, x, z] > th:
                    thresholded_hough[y, x, z] = 1

    # Step 5: Find circle centers.
    centers = np.argwhere(thresholded_hough == 1)
    detected_circles = []

    for center in centers:
        detected_circles.append((center[0], center[1], center[2] + min_radius))

    # Step 6: Display the Hough Space.
    hough_space_sum = np.sum(hough_space, axis=2)

    # Step 7: merge circles

    new_detected_circles = merge_close_circles(detected_circles)

    return thresholded_image, hough_space_sum, new_detected_circles
    # return thresholded_image, hough_space_sum, detected_circles


# ==== MAIN ==============================================


averageTPR = 0
averageF1 = 0

for i in range(4, 16):

    imageName = f'Dartboard/dart{i}.jpg'
    imageNameEdit = imageName.split('/')[-1].split('.')[0]

    # imageName ='Dartboard/dart{i}.jpg'
    # imageNameEdit = imageName.split('/')[-1].split('.')[0]
    # ignore if no such file is present.
    if (not os.path.isfile(imageName)) or (not os.path.isfile(cascade_name)):
        print('No such file')
        sys.exit(1)

    # Read Input Image
    frame = cv2.imread(imageName, 1)

    # ignore if image is not array.
    if not (type(frame) is np.ndarray):
        print('Not image data')
        sys.exit(1)

    # 2. Load the Strong Classifier in a structure called `Cascade'
    model = cv2.CascadeClassifier()
    # if got error, you might need `if not model.load(cv2.samples.findFile(cascade_name)):' instead
    if not model.load(cascade_name):
        print('--(!)Error loading cascade model')
        exit(0)

    iou_threshold = 0.5  # Set threshold value
    ground_truth_boxes = []  # Replace with your list of ground truth bounding boxes

    # Extract the detected boxes for the current image
    print('Image :' + imageNameEdit)
    detected_boxes = detectAndDisplay(frame)
    bounding_boxes = readGroundtruth('GroundTruth.txt')
    draw_bounding_boxes(frame, bounding_boxes, imageNameEdit)

    # Extract the ground truth boxes for the current image
    if bounding_boxes is not None:
        for img_name, x, y, width, height in bounding_boxes:
            if img_name == imageNameEdit:
                ground_truth_boxes.append((x, y, width, height))

    best_matches = {}
    for m, gt_box in enumerate(ground_truth_boxes):
        best_iou = 0
        best_box = None
        for d_box in detected_boxes:
            iou = calculate_iou(d_box, gt_box)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_box = d_box
        if best_box is not None:
            best_matches[m] = best_box

    # best_matches contains each ground truth box matched with the best detected box
    for gt_index, box in best_matches.items():
        print(f"Ground truth box {ground_truth_boxes[gt_index]} best matched with detected box {box} with IOU: "
              f"{calculate_iou(box, ground_truth_boxes[gt_index])}")

    # TPR Calculation
    true_positives = len(best_matches)  # Number of correctly detected faces
    total_ground_truth = len(ground_truth_boxes)  # Total number of ground truth faces
    TPR = true_positives / total_ground_truth
    averageTPR += TPR
    print(f"True positive rate: {TPR}")

    false_positives = len(detected_boxes) - true_positives
    precision = true_positives / (true_positives + false_positives + 0.1)
    recall = TPR  # Recall is the same as TPR

    # F1 score
    F1_score = 2 * (precision * recall) / ((precision + recall) + 0.1)
    averageF1 += F1_score
    print(f"F1 score: {F1_score}")
    print("\n")

    resultName = f"Result_Image/result{i}.jpg"
    filenameForThreshold = f"Result_Image/thresholded_image{i}.jpg"
    filenameForHough = f"Result_Image/hough_space_sum{i}.jpg"
    # cv2.imwrite(resultName, frame)

    # Hough Circle Detection##############################################
    # Read Input Image from source
    frame_circle = cv2.imread(imageName, 1)

    if not (type(frame_circle) is np.ndarray):
        print('Not image data')
        sys.exit(1)

    gray_image = cv2.cvtColor(frame_circle, cv2.COLOR_BGR2GRAY)
    gray_image = gray_image.astype(np.float32)
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    gradient_magnitude = gradient_mag(gray_image)
    gradient_orientation = gradient_ori(gray_image)
    ts = 100
    th = 20
    min_radius = 20
    max_radius = 115

    thresholded_image, hough_space_sum, detectedCircles = hough_circle_detection(gradient_magnitude,
                                                                                 gradient_orientation, ts, th,
                                                                                 min_radius, max_radius)

    for circles in detectedCircles:
        cv2.circle(frame, (circles[1], circles[0]), circles[2], (255, 0, 0), 2)
    cv2.imwrite(filenameForThreshold, thresholded_image)

    hough_space_sum = hough_space_sum.astype(np.uint8)
    cv2.imwrite(filenameForHough, hough_space_sum)

    cv2.imwrite(resultName, frame)

print(f"Average TPR: {averageTPR / 16}")
print(f"Average F1 score: {averageF1 / 16}")







# cv2.imwrite("grayimage.jpg", gray_image)


# cv2.imwrite("hough_space_sum.jpg", hough_space_sum)

# cv2.imwrite("gradient_magnitude.jpg", gradient_magnitude)
# cv2.imwrite("gradient_orientation.jpg", gradient_orientation)
# This saves the image
# cv2.namedWindow('image')  # Name the window
# # cv2.setMouseCallback('image', click_event)
# cv2.imshow('image', frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
