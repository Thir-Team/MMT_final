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

        # Compute the similarity mask, allowing for transparency handling
        similar_count = np.zeros(ui_image.shape, dtype=np.uint8)
        for img in gray_images:
            diff = np.abs(img.astype(np.int16) - ui_image.astype(np.int16))
            similar_regions = (diff < 10).astype(np.uint8)  # Allow small differences for transparency effects
            similar_count += similar_regions

        mask = (similar_count >= similarity_threshold * len(images)).astype(np.uint8) * 255

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
        images = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in image_paths]

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

    def load_train_results(self, base_output_dir: str = "train_result"):
        """Load training results from the specified directory into memory.
        Args:
            base_output_dir: Base directory containing the training results.
        """
        self.ui_data = []
        for game_name in os.listdir(base_output_dir):
            game_dir = os.path.join(base_output_dir, game_name)
            if not os.path.isdir(game_dir):
                continue
            for screen_type in os.listdir(game_dir):
                screen_dir = os.path.join(game_dir, screen_type)
                if not os.path.isdir(screen_dir):
                    continue
                mask_path = os.path.join(screen_dir, "mask.png")
                ui_image_path = os.path.join(screen_dir, "ui_image.png")
                if os.path.exists(mask_path) and os.path.exists(ui_image_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    ui_image = cv2.imread(ui_image_path, cv2.IMREAD_GRAYSCALE)
                    label = f"Game: {game_name}, Screen: {screen_type}"
                    self.ui_data.append((mask, ui_image, label))

    def detect(self, input_image_path: str) -> str:
        """Detect which game's UI the input image belongs to.
        Args:
            input_image_path: The path to the input image.

        Returns:
            The label of the most similar UI.
        """
        input_image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
        input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        # Apply a slight blur to account for minor differences
        input_gray = cv2.GaussianBlur(input_gray, (5, 5), 0)

        best_label = None
        best_similarity = float('inf')

        for mask, ui_image, label in self.ui_data:
            target_size = ui_image.shape  # Get the shape of the first UI image
            resized_input_gray = cv2.resize(input_gray, (target_size[1], target_size[0]))
            # Apply the mask to both the input image and the stored UI image
            
            masked_input = cv2.bitwise_and(resized_input_gray, resized_input_gray, mask=mask)
            masked_ui_image = cv2.bitwise_and(ui_image, ui_image, mask=mask)

            # Handle transparency by ignoring fully transparent regions
            non_zero_mask = mask > 0
            if np.any(non_zero_mask):
                similarity = mean_squared_error(masked_input[non_zero_mask].flatten(), masked_ui_image[non_zero_mask].flatten())
            else:
                similarity = float('inf')

            if similarity < best_similarity:
                best_similarity = similarity
                best_label = label

        return best_label

if __name__ == "__main__":
    detector = GameUIDetector()
    
    # Example: Training from a folder
    # detector.train_from_folder("training_input", "Brawl Stars", "Home")
    """
    # Load training results from the train_result directory
    detector.load_train_results()

    # Example detection
    detecting_input_dir = "detecting_input"
    detecting_images = [os.path.join(detecting_input_dir, fname) for fname in os.listdir(detecting_input_dir) if fname.endswith(('.png', '.jpg', '.jpeg'))]
    
    if detecting_images:
        for image_path in detecting_images:  # Iterate through all images in the folder
            result = detector.detect(image_path)
            print(f"Image: {image_path}, Detected: {result}")
    else:
        print("No images found in detecting_input directory.")
    """