import random
import cv2


class DetectionUtils:
    """
    Utility class for drawing bounding boxes and labels on images.
    """

    # Dictionary to store colors for each label
    label_colors = {}

    @staticmethod
    def generate_random_color():
        """
        Generates a random color.

        Returns:
            A tuple representing a color in BGR format.
        """
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    @staticmethod
    def get_color_for_label(label):
        """
        Retrieves the color for a given label, generating a new color if the label is not yet in the dictionary.

        Args:
            label: The label for which to retrieve the color.

        Returns:
            A tuple representing a color in BGR format.
        """
        if label not in DetectionUtils.label_colors:
            DetectionUtils.label_colors[label] = DetectionUtils.generate_random_color()
        return DetectionUtils.label_colors[label]

    @staticmethod
    def draw_bounding_boxes(image, detections):
        """
        Draws bounding boxes and labels on the image for each detection, each with a consistent color.

        Args:
            image: The image on which to draw.
            detections: A list of detection objects.
        """
        for detection in detections:
            box = detection["box"]
            label = detection["label"]
            color = DetectionUtils.get_color_for_label(label)

            # Draw a rectangle around the detected object
            cv2.rectangle(
                image,
                (box.origin_x, box.origin_y),
                (box.origin_x + box.width, box.origin_y + box.height),
                color,
                2,
            )

            # Put a label above the rectangle
            cv2.putText(
                image,
                label,
                (box.origin_x, box.origin_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
