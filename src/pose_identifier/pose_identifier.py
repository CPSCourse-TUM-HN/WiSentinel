import math
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from PIL import Image

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils


def get_angle(a, b, c):
    """Returns the angle (in degrees) at point b given three landmarks a-b-c"""
    ab = [a.x - b.x, a.y - b.y]
    cb = [c.x - b.x, c.y - b.y]
    dot = ab[0] * cb[0] + ab[1] * cb[1]
    ab_mag = math.hypot(*ab)
    cb_mag = math.hypot(*cb)
    if ab_mag * cb_mag == 0:
        return 0
    angle = math.acos(dot / (ab_mag * cb_mag))
    return math.degrees(angle)


def classify_pose(landmarks):
    print("--- Classifying Pose ---")
    if landmarks is None:
        print("No landmarks detected.")
        return "no_person"

    # Extract key points
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]  # Added nose for head position

    # Calculate vertical position of torso (average of shoulders)
    torso_y = (left_shoulder.y + right_shoulder.y) / 2

    # Calculate vertical position of head (using nose)
    head_y = nose.y

    # Calculate vertical difference between head and torso
    head_torso_v_diff = torso_y - head_y  # Positive if head is higher than torso

    print(f"Head Y: {head_y:.2f}")
    print(f"Torso Y (avg shoulders): {torso_y:.2f}")
    print(f"Head-Torso Vertical Difference: {head_torso_v_diff:.2f}")

    # Body height (rough estimate) - maybe useful later
    body_height = (
        abs(left_shoulder.y - left_ankle.y)
        if (left_shoulder.y and left_ankle.y)
        else None
    )

    # Torso height
    torso = abs(left_shoulder.y - left_hip.y)

    # Shoulders horizontal and arms extended
    shoulder_dist = abs(left_shoulder.x - right_shoulder.x)
    left_arm_angle = get_angle(left_shoulder, left_elbow, left_wrist)
    right_arm_angle = get_angle(right_shoulder, right_elbow, right_wrist)

    print(f"\nLeft arm angle: {left_arm_angle:.2f}")
    print(f"Right arm angle: {right_arm_angle:.2f}")
    print(
        f"Left wrist Y relative to shoulder Y: {abs(left_wrist.y - left_shoulder.y):.2f}"
    )
    print(
        f"Right wrist Y relative to shoulder Y: {abs(right_wrist.y - right_shoulder.y):.2f}"
    )
    print(f"Shoulder horizontal distance: {shoulder_dist:.2f}")

    # T-Pose: Check if wrists are roughly at shoulder height and arms are extended
    wrist_shoulder_threshold_y = 0.05  # Threshold for vertical alignment
    arm_angle_min = 160
    arm_angle_max = 200

    print("\nChecking for T-pose conditions:")
    print(
        f"  Wrist vertical alignment (Left): {abs(left_wrist.y - left_shoulder.y) < wrist_shoulder_threshold_y}"
    )
    print(
        f"  Wrist vertical alignment (Right): {abs(right_wrist.y - right_shoulder.y) < wrist_shoulder_threshold_y}"
    )
    print(f"  Left arm angle: {arm_angle_min < left_arm_angle < arm_angle_max}")
    print(f"  Right arm angle: {arm_angle_min < right_arm_angle < arm_angle_max}")

    if (
        abs(left_wrist.y - left_shoulder.y) < wrist_shoulder_threshold_y
        and abs(right_wrist.y - right_shoulder.y) < wrist_shoulder_threshold_y
        and arm_angle_min < left_arm_angle < arm_angle_max
        and arm_angle_min < right_arm_angle < arm_angle_max
    ):
        print("T-pose conditions met.")
        return "T-pose"
    else:
        print("T-pose conditions not met.")

    # Crouching: knees bent + hips lowered
    left_knee_angle = get_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = get_angle(right_hip, right_knee, right_ankle)
    hips_low = left_hip.y > left_shoulder.y

    print(f"\nLeft knee angle: {left_knee_angle:.2f}")
    print(f"Right knee angle: {right_knee_angle:.2f}")
    print(f"Hips low (left_hip.y > left_shoulder.y): {hips_low}")

    print("\nChecking for crouching conditions:")
    print(f"  Left knee angle < 120: {left_knee_angle < 120}")
    print(f"  Right knee angle < 120: {right_knee_angle < 120}")
    print(f"  Hips low: {hips_low}")

    if left_knee_angle < 120 and right_knee_angle < 120 and hips_low:
        print("Crouching conditions met.")
        return "crouching"
    else:
        print("Crouching conditions not met.")

    # Standing: head significantly above torso + hips relatively high
    standing_head_v_diff_min = 0.075  # Minimum vertical difference for standing
    standing_hip_threshold_y = 0.6  # Adjusted threshold for standing hips

    print(f"\nLeft hip Y: {left_hip.y:.2f}")
    print(f"Right hip Y: {right_hip.y:.2f}")
    print(f"Head-Torso Vertical Difference: {head_torso_v_diff:.2f}")
    print("\nChecking for standing conditions:")
    print(
        f"  Left hip Y < {standing_hip_threshold_y}: {left_hip.y < standing_hip_threshold_y}"
    )
    print(
        f"  Right hip Y < {standing_hip_threshold_y}: {right_hip.y < standing_hip_threshold_y}"
    )
    print(
        f"  Head-Torso Vertical Difference > {standing_head_v_diff_min}: {head_torso_v_diff > standing_head_v_diff_min}"
    )

    if (
        left_hip.y < standing_hip_threshold_y
        and right_hip.y < standing_hip_threshold_y
        and head_torso_v_diff > standing_head_v_diff_min
    ):
        print("Standing conditions met.")
        return "standing"
    else:
        print("Standing conditions not met.")

    # Lying: hips relatively low (close to bottom) AND head NOT significantly above torso
    lying_hip_threshold_y = 0.45  # Threshold for lying hips
    lying_head_v_diff_max = 0.10  # Maximum vertical difference for lying

    print(f"\nLeft hip Y: {left_hip.y:.2f}")
    print(f"Right hip Y: {right_hip.y:.2f}")
    print(f"Head-Torso Vertical Difference: {head_torso_v_diff:.2f}")
    print("\nChecking for lying conditions:")
    print(
        f"  Left hip Y > {lying_hip_threshold_y}: {left_hip.y > lying_hip_threshold_y}"
    )
    print(
        f"  Right hip Y > {lying_hip_threshold_y}: {right_hip.y > lying_hip_threshold_y}"
    )
    print(
        f"  Head-Torso Vertical Difference <= {lying_head_v_diff_max}: {head_torso_v_diff <= lying_head_v_diff_max}"
    )

    if (
        left_hip.y > lying_hip_threshold_y
        and right_hip.y > lying_hip_threshold_y
        and head_torso_v_diff <= lying_head_v_diff_max
    ):
        print("Lying conditions met.")
        return "lying"
    else:
        print("Lying conditions not met.")

    print("\nNo specific pose conditions met.")
    return "unknown"


# File Upload Part
file_name = "IdentifierImage.jpg"
img = cv2.imread(file_name)

# Check if the image was loaded successfully
if img is None:
    print(f"Error: Could not load image from {file_name}")
else:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print("Running pose detection 10 times...")
    for i in range(10):  # Run 10x
        print(f"Processing iteration {i+1}/10...")

        results = pose.process(img_rgb)

        img_display = img.copy()

        # Draw pose
        if results.pose_landmarks:
            print("Person detected.")
            mp_drawing.draw_landmarks(
                img_display, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
        else:
            print("No person detected in this iteration.")

    # Display the final result after the loop (or remove loop display and uncomment this)
    if results.pose_landmarks:
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
    else:
        print("No person detected after all iterations.")

# Extracts all the Keypoints
if results.pose_landmarks:
    landmarks = results.pose_landmarks.landmark
    keypoints = []
    for lm in landmarks:
        keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])

    keypoints = np.array(keypoints).reshape(1, -1)
    print("Keypoints shape:", keypoints.shape)
else:
    keypoints = None
    print("No keypoints to classify")


# Uses 3. to classify poses, prints result

if results.pose_landmarks:
    landmarks = results.pose_landmarks.landmark
    predicted_pose_label = classify_pose(landmarks)

    # Print the predicted pose label
    print(predicted_pose_label)
else:
    print("No prediction possible.")
