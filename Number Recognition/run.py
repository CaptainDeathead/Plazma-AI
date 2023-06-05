import numpy as np
import pickle
import cv2
from colorama import Fore, Style

with open('Number Recognition/trained_model.pkl', 'rb') as file:
    net = pickle.load(file)

def preprocess_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the image to 25x25 pixels
    resized = cv2.resize(gray, (28, 28))
    # Reshape the image to a column vector
    flattened = resized.flatten()
    # Normalize the pixel values
    normalized = flattened / 255.0
    # Reshape the vector to match the input shape of the neural network
    input_data = np.reshape(normalized, (784, 1))
    return input_data

def run():
    # Create a window to display the drawing canvas
    global canvas
    canvas = np.zeros((300, 300, 3), dtype="uint8") * 255
    window_name = "Drawing Canvas"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_circle)

    # Flag to indicate if getting input is active
    input_active = True

    while True:
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("c"):
            canvas = np.zeros((300, 300, 3), dtype="uint8") * 255
        elif key == ord("p"):
            input_image = preprocess_image(canvas)
            prediction = net.predict(input_image)

            # Get the top two predicted digits and their confidence scores
            top_digits = np.argsort(prediction.ravel())[-2:]
            top_confidences = prediction.ravel()[top_digits]

            # Print the predicted digits and their confidence scores
            print(f"Predicted Digits: {top_digits[1]} ({top_confidences[1]*100:.2f}%), {top_digits[0]} ({top_confidences[0]*100:.2f}%)")
            # print the predicted digit in color
            print(f"{Fore.GREEN}Predicted Digit: {top_digits[1]}{Style.RESET_ALL}")

    cv2.destroyAllWindows()


def draw_circle(event, x, y, flags, param):
    global is_drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        is_drawing = True
        cv2.circle(canvas, (x, y), 10, (255, 255, 255), -1)  # Change color to white
    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False
    elif event == cv2.EVENT_MOUSEMOVE and is_drawing:
        cv2.circle(canvas, (x, y), 10, (255, 255, 255), -1)  # Change color to white

if __name__ == '__main__':
    run()
