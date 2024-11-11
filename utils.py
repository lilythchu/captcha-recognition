import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

label_from_filepath = lambda x: os.path.basename(x).split('-')[0]
label_from_filename = lambda x: x.split('-')[0]

def replace_black_pixels_with_average_color(image):
    # Define the kernel size for the neighborhood, e.g., a 3x3 area
    kernel_size = 3
    half_k = kernel_size // 2
    # Get the height and width of the image
    height, width, _ = image.shape
    # Copy the image to avoid modifying the original directly
    output_image = image.copy()
    # Iterate over each pixel in the image
    for y in range(half_k, height - half_k):
        for x in range(half_k, width - half_k):
            # Check if the current pixel is black
            if np.array_equal(image[y, x], [0, 0, 0]):
                # Extract the neighborhood around the pixel
                neighborhood = image[y - half_k:y + half_k + 1, x - half_k:x + half_k + 1]
                # Flatten the neighborhood and exclude black pixels from averaging
                non_black_pixels = [pixel for pixel in neighborhood.reshape(-1, 3) if not np.array_equal(pixel, [0, 0, 0])]
                # If there are non-black pixels in the neighborhood, calculate their average color
                if non_black_pixels:
                    avg_color = np.mean(non_black_pixels, axis=0).astype(int)
                    # Set the black pixel to the average color
                    output_image[y, x] = avg_color

    return output_image

def remove_black_pixels(image):
    threshold=20
    mask = (image[:, :, 0] < threshold) & (image[:, :, 1] < threshold) & (image[:, :, 2] < threshold)
    image[mask] = [255, 255, 255]
    return image

def vertical_split(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, width = gray_img.shape
    # Threshold the image to make sure it's binary (black and white)
    _, binary_image = cv2.threshold(gray_img, 250, 255, cv2.THRESH_BINARY)
    vertical_histogram = np.sum(binary_image == 0, axis=0)  # Sum where pixel value is 0 (foreground)
    # Find segment points (gaps) in the histogram where there are no foreground pixels
    segments = []
    start = None
    for x in range(width):
        if vertical_histogram[x] == 0:
            # If we are in a gap and previously found a start point, end the segment
            if start is not None:
                if x - start > 10:
                # Append the segmented chunk if it's larger than 10 pixels
                    segments.append(image[:, start:x])
                elif vertical_histogram[start:x].sum() > 50:
                # Maybe it's a I or L character, so we'll keep it
                    segments.append(image[:, start:x])
                start = None  # Reset start
        else:
            # If we're in a character region and start is not set, mark the start of a new chunk
            if start is None:
                start = x
    # Append the last segment if it exists
    if start is not None:
        segments.append(image[:, start:width])
    return segments

def split_connected_chars(segments):
    """
    split the largest segment into two parts if it's too wide
    """
    segment_wide = [seg.shape[1] for seg in segments]
    median_wide = np.median(segment_wide)
    # index of the widest segment
    idx = segment_wide.index(max(segment_wide))
    largest_seg = segments[idx]
    if 1.5 * median_wide < largest_seg.shape[1] < 2.5 * median_wide:
        # split the largest segment into two parts
        half_width = largest_seg.shape[1] // 2
        segments.pop(idx)
        segments.insert(idx, largest_seg[:, :half_width + 3])
        segments.insert(idx + 1, largest_seg[:, half_width - 3:])

    return segments

def crop_vertical_space(segment):
    gray = cv2.cvtColor(segment, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours and len(contours) == 1:
        x, y, w, h = cv2.boundingRect(contours[0])
        cropped_image = segment[y:y+h, x:x+w]
    else:
        # If no contours are found, the image might be empty or already cropped tightly
        cropped_image = segment
    return cropped_image


def segmenter(filepath):
    image = cv2.imread(filepath)
    image = remove_black_pixels(image) # use it if you want a faster segmentation
    # image = replace_black_pixels_with_average_color(image)
    segments = vertical_split(image)
    segments = [crop_vertical_space(segment) for segment in segments]
    label = label_from_filepath(filepath)
    if len(segments) == len(label) - 1:
        segments = split_connected_chars(segments)
    return segments

def plot_segments(filepath, segmenter):
    label = filepath.split('/')[-1].split('-')[0]
    objects = segmenter(filepath)
    _, ax = plt.subplots(1, len(objects), figsize=(15, 5))
    for i, obj in enumerate(objects):
        ax[i].imshow(obj)
        ax[i].axis('off')
    plt.show()
    plt.close()
    print(f'Label: {label}, Segmented: {len(objects)}')
    print('---------------------------------')

    img = cv2.imread(filepath)
    plt.imshow(img)
    plt.axis('off')
    plt.show()