import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Function for Mediapipe detection
def mediapipe_detection(image, model):
    """Processes an image using Mediapipe and returns results.

    Args:
        image (numpy.ndarray): Input image in BGR format.
        model: Mediapipe model for processing the image.

    Returns:
        Tuple: Processed image (in BGR format) and Mediapipe results.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    results = model.process(image)  # Perform detection
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for display
    return image, results

# Function to draw landmarks on the image
def draw_styled_landmarks(image, results):
    """Draws landmarks on the image for pose, face, and hands.

    Args:
        image (numpy.ndarray): Image to draw on.
        results: Mediapipe results containing detected landmarks.
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    # Draw pose, face, and hands
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

# Function to extract keypoints from Mediapipe results
def extract_keypoints(results):
    """Extracts keypoints from pose, face, and hand landmarks.

    Args:
        results: Mediapipe results containing detected landmarks.

    Returns:
        numpy.ndarray: Flattened array of all keypoints.
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

# Function to visualize prediction probabilities
def prob_viz(res, actions, input_frame, colors):
    """Displays prediction probabilities as bars on the image.

    Args:
        res (numpy.ndarray): Prediction probabilities.
        actions (list): List of action labels.
        input_frame (numpy.ndarray): Input image to overlay visualization.
        colors (list): List of colors for each action.

    Returns:
        numpy.ndarray: Image with probability bars overlaid.
    """
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num % len(colors)], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

# Load the model
model = load_model("E:\\PyCharm\\pythonProject\\AI\\pre\\30.h5")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Define actions and random colors
actions = ['book', 'drink', 'computer', 'before', 'chair', 'go', 'clothes', 'who', 'candy',
           'cousin', 'deaf', 'fine', 'help', 'no', 'thin', 'walk', 'year', 'yes', 'all',
           'black', 'cool', 'finish', 'hot', 'like', 'many', 'mother', 'now', 'orange',
           'table', 'thanksgiving']
colors = [(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)) for _ in range(len(actions))]

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Variables for tracking sequences
sequence = []
sentence = []
predictions = []
threshold = 0.5

# Perform detection
with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame and perform detection
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)

        # Extract keypoints
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-20:]  # Keep the last 20 frames

        if len(sequence) == 20:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))

            # Check for stable predictions
            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    sentence.append(actions[np.argmax(res)])
                    sentence = sentence[-5:]

            # Visualize prediction probabilities
            image = prob_viz(res, actions, image, colors)

        # Display the detected sentence
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('OpenCV Feed', image)

        # Exit on pressing 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
