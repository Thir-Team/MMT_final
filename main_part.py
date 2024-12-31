import os
import cv2
import numpy as np
from sklearn.metrics import mean_squared_error
from typing import List, Tuple

class GameUIDetector:
    def __init__(self):
        self.ui_data = []  # Stores tuples of (mask, ui_image, label)

    def generate_mask_and_ui_image(self, images: List[np.ndarray], similarity_threshold: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a mask and UI image by finding regions with sufficient similarity in a set of images.
        Args:
            images: List of input images.
            similarity_threshold: Proportion of images that must have similar pixel values to include in the mask.

        Returns:
            mask: The generated mask highlighting common regions.
            ui_image: The representative UI image.
        """
        if not images:
            raise ValueError("No images provided")

        # Convert all images to grayscale
        gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

        # Stack images into a single array for processing
        stacked_images = np.stack(gray_images, axis=-1)

        # Compute the median image to represent the UI Image
        ui_image = np.median(stacked_images, axis=-1).astype(np.uint8)

        # Compute the similarity mask
        similar_count = np.sum(np.abs(stacked_images - ui_image[..., None]) < 10, axis=-1)  # Allow small differences
        mask = (similar_count / stacked_images.shape[-1] >= similarity_threshold).astype(np.uint8) * 255

        return mask, ui_image

    def train_from_folder(self, training_input: str, game_name: str, screen_type: str):
        """Train the model using images in the specified folder.
        Args:
            training_input: Path to the folder containing training images.
            game_name: Name of the game.
            screen_type: Type of screen (e.g., Home, Battle).
        """
        # Get all image paths in the folder
        image_paths = [os.path.join(training_input, fname) for fname in os.listdir(training_input) if fname.endswith(('.png', '.jpg', '.jpeg'))]
        images = [cv2.imread(path) for path in image_paths]

        if not images:
            raise ValueError(f"No valid images found in {training_input}")

        # Resize all images to the size of the first image
        target_size = (images[0].shape[1], images[0].shape[0])
        images = [cv2.resize(img, target_size) for img in images]

        # Apply a slight blur to account for minor color differences
        images = [cv2.GaussianBlur(img, (5, 5), 0) for img in images]

        # Generate mask and UI image
        mask, ui_image = self.generate_mask_and_ui_image(images)

        # Save mask and UI image to the output directory
        base_output_dir = "train_result"
        output_dir = os.path.join(base_output_dir, game_name, screen_type)
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, "mask.png"), mask)
        cv2.imwrite(os.path.join(output_dir, "ui_image.png"), ui_image)

        # Append to the internal data for detection
        self.ui_data.append((mask, ui_image, f"Game: {game_name}, Screen: {screen_type}"))

    def detect(self, input_image_path: str) -> str:
        """Detect which game's UI the input image belongs to.
        Args:
            input_image_path: The path to the input image.

        Returns:
            The label of the most similar UI.
        """
        input_image = cv2.imread(input_image_path)
        input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        # Resize the input image to match the size of the stored UI images
        if self.ui_data:
            target_size = self.ui_data[0][1].shape  # Get the shape of the first UI image
            input_gray = cv2.resize(input_gray, (target_size[1], target_size[0]))

        # Apply a slight blur to account for minor differences
        input_gray = cv2.GaussianBlur(input_gray, (5, 5), 0)

        best_label = None
        best_similarity = float('inf')

        for mask, ui_image, label in self.ui_data:
            # Apply the mask to both the input image and the stored UI image
            masked_input = cv2.bitwise_and(input_gray, input_gray, mask=mask)
            masked_ui_image = cv2.bitwise_and(ui_image, ui_image, mask=mask)

            # Compute similarity (mean squared error)
            similarity = mean_squared_error(masked_input.flatten(), masked_ui_image.flatten())

            if similarity < best_similarity:
                best_similarity = similarity
                best_label = label

        return best_label

if __name__ == "__main__":
    detector = GameUIDetector()

    # Example: Training from a folder
    detector.train_from_folder("training_input", "Arknights", "Battle")

    # Example detection
    detecting_input_dir = "detecting_input"
    detecting_images = [os.path.join(detecting_input_dir, fname) for fname in os.listdir(detecting_input_dir) if fname.endswith(('.png', '.jpg', '.jpeg'))]

    if detecting_images:
        result = detector.detect(detecting_images[0])  # Use the first image in the folder
        print(f"Detected: {result}")
    else:
        print("No images found in detecting_input directory.")
