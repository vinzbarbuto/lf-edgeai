import random
import cv2
from typing import List
import sys
import numpy as np
from tflite_support.task import processor


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


class SegmentationUtils:
    """
    Utility class for drawing masks on images.
    """

    # Visualization parameters
    _FPS_AVERAGE_FRAME_COUNT = 10
    _FPS_LEFT_MARGIN = 24  # pixels
    _LEGEND_TEXT_COLOR = (0, 0, 255)  # red
    _LEGEND_BACKGROUND_COLOR = (255, 255, 255)  # white
    _LEGEND_FONT_SIZE = 1
    _LEGEND_FONT_THICKNESS = 1
    _LEGEND_ROW_SIZE = 20  # pixels
    _LEGEND_RECT_SIZE = 16  # pixels
    _LABEL_MARGIN = 10
    _OVERLAY_ALPHA = 0.5
    _PADDING_WIDTH_FOR_LEGEND = 150  # pixels

    @staticmethod
    def segmentation_map_to_image(
        segmentation: processor.SegmentationResult,
    ) -> tuple[np.ndarray, List[processor.ColoredLabel]]:
        """Convert the SegmentationResult into a RGB image.

        Args:
            segmentation: An output of a image segmentation model.

        Returns:
            seg_map_img: The visualized segmentation result as an RGB image.
            found_colored_labels: The list of ColoredLabels found in the image.
        """
        # Get the list of unique labels from the model output.
        masks = np.frombuffer(segmentation.category_mask, dtype=np.uint8)
        found_label_indices, inverse_map, counts = np.unique(
            masks, return_inverse=True, return_counts=True
        )
        count_dict = dict(zip(found_label_indices, counts))

        # Sort the list of unique label so that the class with the most pixel comes
        # first.
        sorted_label_indices = sorted(
            found_label_indices, key=lambda index: count_dict[index], reverse=True
        )
        found_colored_labels = [
            segmentation.colored_labels[idx] for idx in sorted_label_indices
        ]

        # Convert segmentation map into RGB image of the same size as the input image.
        # Note: We use the inverse map to avoid running the heavy loop in Python and
        # pass it over to Numpy's C++ implementation to improve performance.
        found_colors = [item.color for item in found_colored_labels]
        output_shape = [segmentation.width, segmentation.height, 3]
        seg_map_img = (
            np.array(found_colors)[inverse_map].reshape(output_shape).astype(np.uint8)
        )

        return seg_map_img, found_colored_labels

    @staticmethod
    def visualize(
        input_image: np.ndarray,
        segmentation_map_image: np.ndarray,
        display_mode: str,
        fps: float,
        colored_labels: List[processor.ColoredLabel],
    ) -> np.ndarray:
        """Visualize segmentation result on image.

        Args:
            input_image: The [height, width, 3] RGB input image.
            segmentation_map_image: The [height, width, 3] RGB segmentation map image.
            display_mode: How the segmentation map should be shown. 'overlay' or
            'side-by-side'.
            fps: Value of fps.
            colored_labels: List of colored labels found in the segmentation result.

        Returns:
            Input image overlaid with segmentation result.
        """
        # Show the input image and the segmentation map image.
        if display_mode == "overlay":
            # Overlay mode.
            overlay = cv2.addWeighted(
                input_image,
                SegmentationUtils._OVERLAY_ALPHA,
                segmentation_map_image,
                SegmentationUtils._OVERLAY_ALPHA,
                0,
            )
        elif display_mode == "side-by-side":
            # Side by side mode.
            overlay = cv2.hconcat([input_image, segmentation_map_image])
        else:
            sys.exit(f"ERROR: Unsupported display mode: {display_mode}.")

        # Show the FPS
        fps_text = "FPS = " + str(int(fps))
        text_location = (
            SegmentationUtils._FPS_LEFT_MARGIN,
            SegmentationUtils._LEGEND_ROW_SIZE,
        )
        cv2.putText(
            overlay,
            fps_text,
            text_location,
            cv2.FONT_HERSHEY_PLAIN,
            SegmentationUtils._LEGEND_FONT_SIZE,
            SegmentationUtils._LEGEND_TEXT_COLOR,
            SegmentationUtils._LEGEND_FONT_THICKNESS,
        )

        # Initialize the origin coordinates of the label.
        legend_x = overlay.shape[1] + SegmentationUtils._LABEL_MARGIN
        legend_y = (
            overlay.shape[0] // SegmentationUtils._LEGEND_ROW_SIZE
            + SegmentationUtils._LABEL_MARGIN
        )

        # Expand the frame to show the label.
        overlay = cv2.copyMakeBorder(
            overlay,
            0,
            0,
            0,
            SegmentationUtils._PADDING_WIDTH_FOR_LEGEND,
            cv2.BORDER_CONSTANT,
            None,
            SegmentationUtils._LEGEND_BACKGROUND_COLOR,
        )

        # Show the label on right-side frame.
        for colored_label in colored_labels:
            rect_color = colored_label.color
            start_point = (legend_x, legend_y)
            end_point = (
                legend_x + SegmentationUtils._LEGEND_RECT_SIZE,
                legend_y + SegmentationUtils._LEGEND_RECT_SIZE,
            )
            cv2.rectangle(
                overlay,
                start_point,
                end_point,
                rect_color,
                -SegmentationUtils._LEGEND_FONT_THICKNESS,
            )

            label_location = (
                legend_x
                + SegmentationUtils._LEGEND_RECT_SIZE
                + SegmentationUtils._LABEL_MARGIN,
                legend_y + SegmentationUtils._LABEL_MARGIN,
            )
            cv2.putText(
                overlay,
                colored_label.category_name,
                label_location,
                cv2.FONT_HERSHEY_PLAIN,
                SegmentationUtils._LEGEND_FONT_SIZE,
                SegmentationUtils._LEGEND_TEXT_COLOR,
                SegmentationUtils._LEGEND_FONT_THICKNESS,
            )
            legend_y += (
                SegmentationUtils._LEGEND_RECT_SIZE + SegmentationUtils._LABEL_MARGIN
            )

        return overlay
