import mediapipe as mp
import cv2
import numpy as np
import os
import time

# Constants
ml = 150
max_x, max_y = 250 + ml, 50
curr_tool = "select tool"
time_init = True
rad = 40
var_inits = False
thick = 4
prevx, prevy = 0, 0

# Define white box region for CAPTCHA
captcha_box = {
    'x1': 100,  # Left coordinate of the white box
    'y1': 100,  # Top coordinate of the white box
    'x2': 540,  # Right coordinate of the white box
    'y2': 300   # Bottom coordinate of the white box
}

# Submit button coordinates
button_x1, button_y1 = 500, 10
button_x2, button_y2 = 630, 60

# Get tools function
def getTool(x):
    if x < 150 + ml:
        return "draw"
    else:
        return "erase"

def index_raised(yi, y9):
    return (y9 - yi) > 40

# Initialize Mediapipe Hands and Drawing utilities
hands = mp.solutions.hands
hand_landmark = hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)
draw = mp.solutions.drawing_utils

# Initialize black canvas and precision layer
canvas = np.zeros((480, 640, 3), dtype="uint8")  # Black canvas
precision_layer = np.zeros_like(canvas)

# Load tools image
tools = cv2.imread("tools.png")
tools = tools.astype('uint8')

mask = np.ones((480, 640), dtype='uint8') * 255

cap = cv2.VideoCapture(0)
while True:
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)

    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    op = hand_landmark.process(rgb)

    # Clear precision layer for each frame
    precision_layer[:] = 0

    if op.multi_hand_landmarks:
        for i in op.multi_hand_landmarks:
            draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS)
            x, y = int(i.landmark[8].x * 640), int(i.landmark[8].y * 480)

            if x < max_x and y < max_y and x > ml:
                if time_init:
                    ctime = time.time()
                    time_init = False
                ptime = time.time()

                cv2.circle(frm, (x, y), rad, (255, 255, 255), 2)  # White circle
                rad -= 1

                if (ptime - ctime) > 0.8:
                    curr_tool = getTool(x)
                    print("your current tool set to: ", curr_tool)
                    time_init = True
                    rad = 40

            else:
                time_init = True
                rad = 40

            if curr_tool == "draw":
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    cv2.line(canvas, (prevx, prevy), (x, y), (255, 255, 255), thick)  # Drawing with white color
                    prevx, prevy = x, y
                else:
                    prevx = x
                    prevy = y

            elif curr_tool == "erase":
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    cv2.circle(canvas, (x, y), 30, (0, 0, 0), -1)  # Erasing with black color (matching the canvas)

    # Draw the white CAPTCHA box
    cv2.rectangle(canvas, (captcha_box['x1'], captcha_box['y1']), 
                           (captcha_box['x2'], captcha_box['y2']), 
                           (255, 255, 255), 2)  # White border for the box

    # Draw the precision circle on the precision layer
    if op.multi_hand_landmarks:
        for i in op.multi_hand_landmarks:
            x, y = int(i.landmark[8].x * 640), int(i.landmark[8].y * 480)
            # Use white circle if selecting a tool, otherwise white
            precision_color = (255, 255, 255) if curr_tool != "select tool" else (255, 255, 255)
            cv2.circle(precision_layer, (x, y), 20, precision_color, 2)

    # Overlay the tools image
    canvas[:max_y, ml:max_x] = cv2.addWeighted(tools, 0.7, canvas[:max_y, ml:max_x], 0.3, 0)

    # Combine the precision layer with the canvas
    combined_canvas = cv2.add(canvas, precision_layer)

    # Display current tool on the canvas
    cv2.putText(combined_canvas, curr_tool, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Draw the submit button on the frame
    cv2.rectangle(combined_canvas, (button_x1, button_y1), (button_x2, button_y2), (0, 255, 0), -1)
    cv2.putText(combined_canvas, "Submit", (button_x1 + 10, button_y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Check for click on submit button
    if op.multi_hand_landmarks:
        for i in op.multi_hand_landmarks:
            x, y = int(i.landmark[8].x * 640), int(i.landmark[8].y * 480)
            if button_x1 < x < button_x2 and button_y1 < y < button_y2:
                # Create folder if it doesn't exist
                if not os.path.exists("img_to_check"):
                    os.makedirs("img_to_check")

                # Crop the CAPTCHA box region from the canvas
                captcha_region = canvas[captcha_box['y1']:captcha_box['y2'], captcha_box['x1']:captcha_box['x2']]
                
                # Save the CAPTCHA box region as an image in the folder
                cv2.imwrite(os.path.join("img_to_check", "captcha_box.png"), captcha_region)
                print("Screenshot of the CAPTCHA box saved in img_to_check/captcha_box.png")

    # Show the integrated canvas with drawing and precision marker
    cv2.imshow("Drawing Canvas", combined_canvas)

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break
