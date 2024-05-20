import cv2
from dbr import BarcodeReader
from transformers import pipeline
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import time
import math

pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")  
sam_checkpoint = "models/sam_vit_b_01ec64.pth" 
model_type = "vit_b" 
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

reader = BarcodeReader()
reader.init_license("YOUR_LICENSE_KEY")

def generate_masks(image, x, y):
    image = np.expand_dims(image, axis=2)
    image = np.repeat(image, 3, axis=2)
    predictor.set_image(image)
    input_point = np.array([[x, y]])
    input_label = np.array([1])
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    max_score_index = np.argmax(scores)
    return masks[max_score_index], input_point, input_label

def find_brightest_point(image):
    max_index = np.unravel_index(np.argmax(image, axis=None), image.shape)
    bright_x, bright_y = max_index[1], max_index[0]
    return bright_x, bright_y

def draw_min_area_rect(image, mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(max_contour)
    
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
    
    if rect[1][0] > rect[1][1]:
        longer_edge_length = rect[1][0]
        center1 = ((box[0][0] + box[1][0]) / 2, (box[0][1] + box[1][1]) / 2)
        center2 = ((box[2][0] + box[3][0]) / 2, (box[2][1] + box[3][1]) / 2)
    else:
        longer_edge_length = rect[1][1]
        center1 = ((box[1][0] + box[2][0]) / 2, (box[1][1] + box[2][1]) / 2)
        center2 = ((box[0][0] + box[3][0]) / 2, (box[0][1] + box[3][1]) / 2)
    line = np.array([center1, center2])
    
    a = center2[0] - center1[0]
    b = center2[1] - center1[1]
    dist = math.sqrt((a * a) + (b * b))
    direction_vector = line[1] - line[0]
    angle_with_y_axis = np.arctan2(direction_vector[0], direction_vector[1]) * 180 / np.pi
    return image, rect, angle_with_y_axis, dist

def apply_mask(image, mask):
    masked_image = np.zeros_like(image)
    masked_image[mask] = image[mask]
    return masked_image

def display_image_with_mask(image, mask, input_point, input_label):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.show()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def recognize_barcodes(image):
    barcode_results = reader.decode_buffer(image)
    if barcode_results is not None:
        for barcode_result in barcode_results:
            barcode_format = barcode_result.barcode_format_string
            barcode_text = barcode_result.barcode_text
            points = barcode_result.localization_result.localization_points

            print(f"Barcode Format: {barcode_format}, Barcode Text: {barcode_text}")

            try:
                barcode_text.encode('ascii')
            except UnicodeEncodeError:
                barcode_text = barcode_text.encode('ascii', 'replace').decode('ascii')
                print(f"Non-ASCII characters in barcode text replaced: {barcode_text}")

            for i in range(4):
                pt1 = (points[i][0], points[i][1])
                pt2 = (points[(i + 1) % 4][0], points[(i + 1) % 4][1])
                cv2.line(image, pt1, pt2, (0, 255, 0), 2)

            cv2.putText(image, f"{barcode_format}: {barcode_text}", (points[0][0], points[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def main():
    cap = cv2.VideoCapture(0) 

    if not cap.isOpened():
        print("Error: Couldn't open camera")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        cv2.imshow('Camera', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print("\nMaking depth...")
            start_time = time.time()
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            depth = pipe(image)["depth"] 
            depth_array = np.array(depth)
            cv2.imshow('Depth', depth_array)
            x, y = find_brightest_point(depth_array)
            mask,input_point, input_label = generate_masks(depth_array, x, y)
            display_image_with_mask(image, mask, input_point, input_label)

            num_true = np.sum(mask)
            if num_true > len(mask) * len(mask[0]) / 2:
                mask = np.logical_not(mask)

            masked_frame = apply_mask(frame, mask)

            frame_with_rects, rect, rotated, dist = draw_min_area_rect(masked_frame.copy(), mask)

            cam_x, cam_y = rect[0][0], rect[0][1]
            print("cam_coordinate_center:", (cam_x, cam_y))
            print("dist:", dist)

            frame_with_barcodes = recognize_barcodes(masked_frame)

            end_time = time.time()
            cv2.imshow('Result', frame_with_barcodes)
            print('inference time:', end_time - start_time)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
