import cv2
import numpy as np

def detect_car_color(car_img):
   
    if car_img is None or car_img.size == 0:
        return "Unknown"

    car_img = cv2.resize(car_img, (200, 200))

    hsv = cv2.cvtColor(car_img, cv2.COLOR_BGR2HSV)

    color_ranges = {
        "Blue": [
            (np.array([90, 50, 50]), np.array([130, 255, 255]))
        ],

        "Red": [
            (np.array([0, 70, 50]), np.array([10, 255, 255])),
            (np.array([170, 70, 50]), np.array([180, 255, 255]))
        ],

        "Green": [
            (np.array([35, 50, 50]), np.array([85, 255, 255]))
        ],

        "Yellow": [
            (np.array([20, 100, 100]), np.array([35, 255, 255]))
        ],

        "White": [
            (np.array([0, 0, 180]), np.array([180, 40, 255]))
        ],

        "Black": [
            (np.array([0, 0, 0]), np.array([180, 255, 40]))
        ],

        "Silver": [
            (np.array([0, 0, 80]), np.array([180, 40, 180]))
        ]
    }

    color_pixels = {}

    for color, ranges in color_ranges.items():

        total_pixels = 0

        for lower, upper in ranges:

            mask = cv2.inRange(hsv, lower, upper)

            total_pixels += cv2.countNonZero(mask)

        color_pixels[color] = total_pixels

    dominant_color = max(color_pixels, key=color_pixels.get)

    if color_pixels[dominant_color] < 500:
        return "Unknown"

    return dominant_color