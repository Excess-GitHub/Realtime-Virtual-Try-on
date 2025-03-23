import os
import cv2
import time
import torch
import numpy as np
import torch.nn as nn
import mediapipe as mp
from PIL import Image
from cvzone.HandTrackingModule import HandDetector
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

## UI FUNCTIONS
def extract_button_contour(img, hair_scale=1.0, offset=(0, 0)):
    img_binary = img[:, :, 3]

    contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found in the image")
    largest_contour = max(contours, key=cv2.contourArea)

    translated_contour = largest_contour + np.array(offset, dtype=np.int32)

    return translated_contour

def overlay_button(background, overlay, x, y, opacity):
    overlay_height, overlay_width = overlay.shape[:2]

    alpha = (overlay[:, :, 3] / 255.0) * opacity
    alpha_inv = 1.0 - alpha

    for c in range(0, 3):
        background[y:y+overlay_height, x:x+overlay_width, c] = (
            alpha * overlay[:, :, c] +
            alpha_inv * background[y:y+overlay_height, x:x+overlay_width, c]
        )

def overlay_image_with_border(background, overlay, x, y, border_thickness, border_color, selected=False):
    overlay_height, overlay_width = overlay.shape[:2]
    if selected:
        # Draw border around the selected image
        cv2.rectangle(
            background,
            (x - border_thickness, y - border_thickness),
            (x + overlay_width + border_thickness, y + overlay_height + border_thickness),
            border_color,
            border_thickness
        )
    # Overlay the image
    overlay_button(background, overlay, x, y, 1.0 if selected else 0.5)

def is_within_contour(landmarks, contour):
    return any(cv2.pointPolygonTest(contour, (int(landmark[0]), int(landmark[1])), False) >= 0 for landmark in landmarks)

## MAIN FUNCTIONS
def create_shirt_mask(shirt_image):
    shirt_mask = shirt_image[:, :, 3]

    # Optionally, clean up small noise using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    shirt_mask = cv2.morphologyEx(shirt_mask, cv2.MORPH_ERODE, kernel, iterations=3)

    return shirt_mask

def get_bounding_box_corners(binary_mask):
    """
    Get the corners of the bounding box for a binary mask.
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Calculate the bounding box around the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Define the four corners of the bounding box
        top_left = [x, y]
        top_right = [x + w, y]
        bottom_left = [x, y + h]
        bottom_right = [x + w, y + h]

        # Visualize the bounding box on the mask
        boxed_mask = cv2.rectangle(cv2.cvtColor(binary_mask.copy(), cv2.COLOR_GRAY2BGR), (x, y), (x + w, y + h), (255, 0, 0), 2)

        return True, np.array([top_left, top_right, bottom_left, bottom_right], dtype=np.float32)
    else:
        return False, np.array([])

def perspective_transform(source_points, target_points, img, output_shape, binary_mask):
    """
    Perform perspective transformation and apply a black background.
    """
    # Calculate the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(source_points, target_points)
    
    # Warp the image
    warped_img = cv2.warpPerspective(img, matrix, (output_shape[1], output_shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    warped_mask = cv2.warpPerspective(binary_mask, matrix, (output_shape[1], output_shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Ensure the background is black outside the shirt region
    warped_with_black = cv2.bitwise_and(warped_img, warped_img, mask=warped_mask)

    return warped_with_black, warped_mask

from scipy.spatial import KDTree

def fill_unfilled_regions_tree(warped_shirt, warped_mask, segmentation_mask):
    """
    Fill unfilled regions in the segmentation mask with colors sampled
    from the edges of the warped shirt.
    """
    # Compute the XOR mask (unfilled regions)
    unfilled_mask = cv2.bitwise_xor(warped_mask, segmentation_mask)

    # Find edge pixels in the warped mask
    edge_pixels = np.column_stack(np.where(warped_mask > 0))

    if len(edge_pixels) == 0:
        return warped_shirt

    # Build a KDTree for fast nearest-neighbor search
    tree = KDTree(edge_pixels)

    # Find unfilled region pixels
    unfilled_pixels = np.column_stack(np.where(unfilled_mask > 0))

    # Query all unfilled pixels in a single call
    _, nearest_edge_indices = tree.query(unfilled_pixels)

    # Map unfilled pixels to the nearest edge pixels
    nearest_edge_pixels = edge_pixels[nearest_edge_indices]

    # Get the colors of the nearest edge pixels
    nearest_colors = warped_shirt[nearest_edge_pixels[:, 0], nearest_edge_pixels[:, 1]]

    # Assign these colors to the unfilled pixels
    filled_shirt = warped_shirt.copy()
    filled_shirt[unfilled_pixels[:, 0], unfilled_pixels[:, 1]] = nearest_colors

    return filled_shirt

def generate_mask(image, model, processor, device):
    inputs = processor(images=image, return_tensors="pt").to(device)

    outputs = model(**inputs)
    logits = outputs.logits

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
    
    return pred_seg

def get_hair_bbox(rgb_image, timestamp_ms, model, options):
    with model.create_from_options(options) as segmenter:
        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Perform hair segmentation
        segmentation_result = segmenter.segment_for_video(mp_image, timestamp_ms)

        # Retrieve the category mask
        category_mask = segmentation_result.category_mask

        # Convert the category mask to a binary mask for the hair region
        hair_mask = np.where(category_mask.numpy_view() == 1, 255, 0).astype(np.uint8)

        # Find contours of the hair region
        contours, _ = cv2.findContours(hair_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Identify the largest contour (assumed to be the hair region)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)  # Get the bounding box

            top_left = (x, y)
            top_right = (x + w, y)
            bottom_left = (x, y + h)
            bottom_right = (x + w, y + h)

            return True, top_left, top_right, bottom_left, bottom_right
        else:
            return False, None, None, None, None


def main():
    ## Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "mattmdjaga/segformer_b2_clothes"

    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
    model = model.to(device)

    # HairSegmenter Model Setup
    BaseOptions = mp.tasks.BaseOptions
    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    model_file = open('hair_segmenter.tflite', "rb")
    model_data = model_file.read()
    model_file.close()

    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_buffer=model_data),
        running_mode=VisionRunningMode.VIDEO,
        output_category_mask=True
    )

    # Face Mesh Model Setup
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    # Webcam Setup
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    timestamp_ms = 0

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    ## UI Setup
    # Hand Detection Setup
    detector = HandDetector(maxHands=2, modelComplexity=0)

    # Button Setup
    scale = 0.25
    right = cv2.imread('button.png', cv2.IMREAD_UNCHANGED)

    button_width = int(right.shape[1] * scale)
    button_height = int(right.shape[0] * scale)

    right = cv2.resize(right, (button_width, button_height))
    left = cv2.flip(right, 1)
    shirt_button = cv2.imread('shirt_button.png', cv2.IMREAD_UNCHANGED)
    shirt_button = cv2.resize(shirt_button, (button_width, button_height))
    hair_button = cv2.imread('hair_button.png', cv2.IMREAD_UNCHANGED)
    hair_button = cv2.resize(hair_button, (button_width, button_height))

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    right_button_position = (frame_width - (button_width + 10), (frame_height // 2) - (button_height // 2))
    left_button_position = (10, (frame_height // 2) - (button_height // 2))
    shirt_button_position = (10, 10)
    hair_button_position = (frame_width - (button_width + 10), 10)

    right_button_outline = extract_button_contour(right, offset=right_button_position)
    left_button_outline = extract_button_contour(left, offset=left_button_position)
    shirt_button_outline = extract_button_contour(shirt_button, offset=shirt_button_position)
    hair_button_outline = extract_button_contour(hair_button, offset=hair_button_position)

    start_time = None
    duration = 1
    current_opacity_left = 0.2
    current_opacity_right = 0.2
    current_opacity_shirt = 0.2
    current_opacity_hair = 0.2
    update = False

    # Shirt carousel setup and shirt image preprocessing
    image_dir = "Shirts"
    image_names = os.listdir(image_dir)
    image_paths = [os.path.join(image_dir, name) for name in image_names]
    print(image_paths)
    shirt_images = [cv2.imread(img_path, cv2.IMREAD_UNCHANGED) for img_path in image_paths]
    shirt_masks = [create_shirt_mask(shirt_image) for shirt_image in shirt_images]
    shirt_bboxes = [get_bounding_box_corners(shirt_binary_mask)[1] for shirt_binary_mask in shirt_masks]
    image_width, image_height = 100, 100  # Resize all images to uniform dimensions
    carousel_shirt_icons = [cv2.resize(img, (image_width, image_height)) for img in shirt_images]
    shirt_current_index = 0

    # Get bounding box corners for the shirt and segmentation mask
    shirt_image = shirt_images[shirt_current_index % len(carousel_shirt_icons)]
    shirt_binary_mask = shirt_masks[shirt_current_index % len(carousel_shirt_icons)]
    shirt_corners = shirt_bboxes[shirt_current_index % len(carousel_shirt_icons)]

    # Hair carousel setup
    image_dir = "Hairstyles"
    image_names = os.listdir(image_dir)
    image_paths = [os.path.join(image_dir, name) for name in image_names]
    print(image_paths)
    hairstyle_images = [cv2.imread(img_path, cv2.IMREAD_UNCHANGED) for img_path in image_paths]
    image_width, image_height = 100, 100  # Resize all images to uniform dimensions
    carousel_hair_icons = [cv2.resize(img, (image_width, image_height)) for img in hairstyle_images]
    hair_current_index = 0

    hairstyle_image = hairstyle_images[hair_current_index % len(carousel_hair_icons)]
    hair_scale_y = 20
    hair_scale_x = 30

    mode = "hair"
    carousel_position_y = frame_height - 150  # Position of carousel near the bottom of the frame

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        frame = cv2.flip(frame, 1)
        img = Image.fromarray(frame)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms += 33

        # UI LOOP
        hands, frame = detector.findHands(frame, flipType=False, draw=False)

        current_opacity_left = 0.2
        current_opacity_right = 0.2
        current_opacity_shirt = 0.2
        current_opacity_hair = 0.2

        if len(hands) > 0:
            lm_list = hands[0]['lmList']    

            if is_within_contour(lm_list, shirt_button_outline):
                if start_time is None:
                    start_time = time.time()
                elapsed_time = time.time() - start_time
                current_opacity_shirt = min(1.0, elapsed_time / duration)
                if elapsed_time >= duration:
                    print("Shirt button pressed!")
                    mode = "shirt"
                    start_time = None
            elif is_within_contour(lm_list, hair_button_outline):
                if start_time is None:
                    start_time = time.time()
                elapsed_time = time.time() - start_time
                current_opacity_hair = min(1.0, elapsed_time / duration)
                if elapsed_time >= duration:
                    print("Hair button pressed!")
                    mode = "hair"
                    start_time = None
            elif is_within_contour(lm_list, left_button_outline):
                if start_time is None:
                    start_time = time.time()
                elapsed_time = time.time() - start_time
                current_opacity_left = min(1.0, elapsed_time / duration)
                if elapsed_time >= duration:
                    print("Left button pressed!")
                    if mode == "shirt":
                        shirt_current_index = (shirt_current_index - 1) % len(carousel_shirt_icons)
                    else:
                        hair_current_index = (hair_current_index - 1) % len(carousel_hair_icons)
                    start_time = None
                    update = True
            elif is_within_contour(lm_list, right_button_outline):
                if start_time is None:
                    start_time = time.time()
                elapsed_time = time.time() - start_time
                current_opacity_right = min(1.0, elapsed_time / duration)
                if elapsed_time >= duration:
                    print("Right button pressed!")
                    if mode == "shirt":
                        shirt_current_index = (shirt_current_index + 1) % len(carousel_shirt_icons)
                    else:
                        hair_current_index = (hair_current_index + 1) % len(carousel_hair_icons)
                    start_time = None
                    update = True
            else:
                start_time = None
        elif len(hands) > 1:
            if hands[0]['type'] == "Left":
                lm_list_left = hands[0]['lmList']
                lm_list_right = hands[1]['lmList']
            else:
                lm_list_left = hands[1]['lmList']
                lm_list_right = hands[0]['lmList']
            
            if is_within_contour(lm_list_left, shirt_button_outline) or (lm_list_right, shirt_button_outline):
                if start_time is None:
                    start_time = time.time()
                elapsed_time = time.time() - start_time
                current_opacity_shirt = min(1.0, elapsed_time / duration)
                if elapsed_time >= duration:
                    print("Shirt button pressed!")
                    mode = "shirt"
                    start_time = None
            elif is_within_contour(lm_list_left, hair_button_outline) or (lm_list_right, hair_button_outline):
                if start_time is None:
                    start_time = time.time()
                elapsed_time = time.time() - start_time
                current_opacity_hair = min(1.0, elapsed_time / duration)
                if elapsed_time >= duration:
                    print("Hair button pressed!")
                    mode = "hair"
                    start_time = None
            elif is_within_contour(lm_list_left, left_button_outline) or (lm_list_right, left_button_outline):
                if start_time is None:
                    start_time = time.time()
                elapsed_time = time.time() - start_time
                current_opacity_left = min(1.0, elapsed_time / duration)
                if elapsed_time >= duration:
                    print("Left button pressed!")
                    if mode == "shirt":
                        shirt_current_index = (shirt_current_index - 1) % len(carousel_shirt_icons)
                    else:
                        hair_current_index = (hair_current_index - 1) % len(carousel_hair_icons)
                    start_time = None
                    update = True
            elif is_within_contour(lm_list_left, right_button_outline) or (lm_list_right, right_button_outline):
                if start_time is None:
                    start_time = time.time()
                elapsed_time = time.time() - start_time
                current_opacity_right = min(1.0, elapsed_time / duration)
                if elapsed_time >= duration:
                    print("Right button pressed!")
                    if mode == "shirt":
                        shirt_current_index = (shirt_current_index + 1) % len(carousel_shirt_icons)
                    else:
                        hair_current_index = (hair_current_index + 1) % len(carousel_hair_icons)
                    start_time = None
                    update = True
            else:
                start_time = None
        
        if update == True:
            update = False
            if mode == "shirt":
                shirt_image = shirt_images[shirt_current_index % len(carousel_shirt_icons)]
                shirt_binary_mask = shirt_masks[shirt_current_index % len(carousel_shirt_icons)]
                shirt_corners = shirt_bboxes[shirt_current_index % len(carousel_shirt_icons)]
            else:
                hairstyle_image = hairstyle_images[hair_current_index % len(carousel_hair_icons)]

        # MAIN LOOP
        if mode == "shirt":
            ## Shirt
            pred_seg = generate_mask(img,  model=model, processor=processor, device=device)

            shirt_segmentation = (pred_seg == 4).astype(np.uint8) * 255
            shirt_segmentation_img = Image.fromarray(shirt_segmentation, mode='L')
            segmentation_mask = np.array(shirt_segmentation_img)

            ret, segmentation_corners = get_bounding_box_corners(segmentation_mask)
            if ret:
                # Apply perspective transformation
                warped_shirt, warped_mask = perspective_transform(shirt_corners, segmentation_corners, shirt_image[:, :, :3], output_shape=frame.shape[:2], binary_mask=shirt_binary_mask)

                # Overlay the warped shirt onto the original image
                mask_inverse = cv2.bitwise_not(segmentation_mask)
                background = cv2.bitwise_and(frame, frame, mask=mask_inverse)
                warped_shirt = fill_unfilled_regions_tree(warped_shirt, warped_mask, segmentation_mask)
                foreground = cv2.bitwise_and(warped_shirt, warped_shirt, mask=segmentation_mask)
                frame = cv2.add(foreground, background)
        else:
            ## Hair
            ret, top_left, top_right, bottom_left, bottom_right = get_hair_bbox(rgb_image, timestamp_ms, ImageSegmenter, options)
            if ret:
                # Detect facial landmarks
                results = face_mesh.process(rgb_image)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Extract key landmarks for the top of the head
                        bottom_points = [234, 454]  # Example landmarks
                        bottom_coords = [
                            (int(landmark.x * frame_width), int(landmark.y * frame_height))
                            for idx, landmark in enumerate(face_landmarks.landmark) if idx in bottom_points
                        ]

                        # Calculate bounding box points
                        bbox_points = np.array([
                            [top_left[0] - hair_scale_x, top_left[1] - hair_scale_y],  # Top-left
                            [top_right[0] + hair_scale_x, top_right[1] - hair_scale_y],  # Top-right
                            [bottom_coords[1][0] + hair_scale_x, bottom_coords[1][1]],  # Bottom-right
                            [bottom_coords[0][0] - hair_scale_x, bottom_coords[0][1]]   # Bottom-left
                        ], dtype=np.float32)

                        # Extract the region of interest (actual hair content) from the hair image
                        hair_alpha = hairstyle_image[:, :, 3]  # Extract the alpha channel
                        non_transparent_coords = cv2.findNonZero(hair_alpha)  # Find non-transparent regions
                        hair_x, hair_y, hair_w, hair_h = cv2.boundingRect(non_transparent_coords)

                        # Define the original points (corners of the cropped hair image)
                        hair_cropped = hairstyle_image[hair_y:hair_y + hair_h, hair_x:hair_x + hair_w]
                        hair_cropped_alpha = hair_cropped[:, :, 3] / 255.0  # Normalize alpha
                        hair_original_points = np.array([
                            [0, 0],                      # Top-left
                            [hair_cropped.shape[1], 0],  # Top-right
                            [hair_cropped.shape[1], hair_cropped.shape[0]],  # Bottom-right
                            [0, hair_cropped.shape[0]]   # Bottom-left
                        ], dtype=np.float32)

                        # Compute the perspective transform matrix
                        matrix = cv2.getPerspectiveTransform(hair_original_points, bbox_points)

                        # Warp the hair image to align with the bounding box
                        warped_hair = cv2.warpPerspective(hair_cropped, matrix, (frame.shape[1], frame.shape[0]))
                        warped_alpha = cv2.warpPerspective(hair_cropped_alpha, matrix, (frame.shape[1], frame.shape[0]))

                        # Blend the warped hair image with the frame
                        try:
                            for c in range(3):  # For each color channel (R, G, B)
                                frame[:, :, c] = (
                                    warped_alpha * warped_hair[:, :, c] +
                                    (1 - warped_alpha) * frame[:, :, c]
                                )
                        except Exception as e:
                            print(f"Error blending hair overlay: {e}")
                            continue
        
        overlay_button(frame, left, left_button_position[0], left_button_position[1], current_opacity_left)
        overlay_button(frame, right, right_button_position[0], right_button_position[1], current_opacity_right)
        overlay_button(frame, shirt_button, shirt_button_position[0], shirt_button_position[1], current_opacity_shirt)
        overlay_button(frame, hair_button, hair_button_position[0], hair_button_position[1], current_opacity_hair)

        # Display the image carousel
        if mode == "shirt":
            carousel_images = [
                carousel_shirt_icons[(shirt_current_index - 1) % len(carousel_shirt_icons)],  # Previous image (wrap around)
                carousel_shirt_icons[shirt_current_index % len(carousel_shirt_icons)],       # Current selected image
                carousel_shirt_icons[(shirt_current_index + 1) % len(carousel_shirt_icons)]  # Next image (wrap around)
            ]
        else:
            carousel_images = [
                carousel_hair_icons[(hair_current_index - 1) % len(carousel_hair_icons)],  # Previous image (wrap around)
                carousel_hair_icons[hair_current_index % len(carousel_hair_icons)],       # Current selected image
                carousel_hair_icons[(hair_current_index + 1) % len(carousel_hair_icons)]  # Next image (wrap around)
            ]
        for i, img in enumerate(carousel_images):
            x_pos = (frame_width // 2 - image_width // 2) + (i - 1) * (image_width + 20)
            selected = (i == 1)  # Middle image is selected
            overlay_image_with_border(frame, img, x_pos, carousel_position_y, 5, (0, 0, 0), selected)

        cv2.imshow('Segmentation', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()