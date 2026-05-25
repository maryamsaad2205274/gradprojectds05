import math


def euclidean_distance(p1, p2):
    return math.sqrt((p2["x"] - p1["x"]) ** 2 + (p2["y"] - p1["y"]) ** 2)


def calculate_measurement(calibration_points, facial_points, real_distance_mm):
    if len(calibration_points) != 2:
        raise ValueError("Please select the start and end of the red calibration tool.")

    if len(facial_points) != 2:
        raise ValueError("Please select exactly 2 facial points.")

    try:
        real_distance_mm = float(real_distance_mm)
    except (TypeError, ValueError):
        raise ValueError("Please enter a valid real calibration distance in mm.")

    if real_distance_mm <= 0:
        raise ValueError("Real calibration distance must be greater than 0.")

    calibration_pixel_distance = euclidean_distance(
        calibration_points[0], calibration_points[1]
    )

    if calibration_pixel_distance == 0:
        raise ValueError("Calibration tool start and end points cannot be identical.")

    mm_per_pixel = real_distance_mm / calibration_pixel_distance

    facial_pixel_distance = euclidean_distance(
        facial_points[0], facial_points[1]
    )
    facial_distance_mm = facial_pixel_distance * mm_per_pixel

    return {
        "calibration_pixel_distance": round(calibration_pixel_distance, 2),
        "mm_per_pixel": round(mm_per_pixel, 6),
        "facial_pixel_distance": round(facial_pixel_distance, 2),
        "facial_distance_mm": round(facial_distance_mm, 2),
    }
