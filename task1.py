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


# ==== MAIN ==============================================

averageTPR = 0
averageF1 = 0
for i in range(0, 16):
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

    resultName = f"Result_Image1/result{i}.jpg"
    cv2.imwrite(resultName, frame)

print(f"Average TPR: {averageTPR / 16}")
print(f"Average F1 score: {averageF1 / 16}")
print(averageTPR/16+averageF1/16)
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
