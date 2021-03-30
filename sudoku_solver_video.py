from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2
import imutils
from PreProcessing import Preprocessor
import solve_algorithm as solver
from cvTools.ConvNets.LeNet import LeNet


def grid_already_detected(grid):
    if os.path.exists("temp.txt"):
        with open("temp.txt", "r") as f:
            g = f.read().splitlines()[0]
            if g == grid:
                return True
    with open("temp.txt", "w") as f:
        f.write(grid)
    return False


def remove_temp():
    if os.path.exists("temp.txt"):
        os.remove("temp.txt")


args = imutils.get_args(single_image=True, use_video=True)
useLeNet = True

if useLeNet:
    model = load_model("models\\lenet_mnist_augment_decay.h5")
else:
    model = load_model("models\\digitnet_mnist_augment_decay_4.h5")

if args["video"]:
    cap = cv2.VideoCapture(args["video"])
else:
    cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if args["video"] and not ret:
        break
    frame = imutils.resize(frame, height=800)
    height, width = frame.shape[:2]

    preprocessor = Preprocessor(28, 28, debug=False, showDigitsOnly=False)
    sudoku, sudoku_gray, keypts_img, keypts_grid, img_without_grid = preprocessor.extract_grid(frame)
    if sudoku is not None:
        rows, cols = sudoku.shape[:2]
        cell_height, cell_width = rows // 9, cols // 9
        grid = ""
        for i in range(9):  # i is for the y co-ordinate
            for j in range(9):  # j is for the x co-ordinate
                cell_ = sudoku_gray[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width]
                digit = preprocessor.extract_digit(cell_)
                if digit is not None:
                    digit = np.expand_dims(digit, axis=(0, -1))
                    digit = digit*1./255
                    pred = model.predict(digit)
                    pred = pred.sum(axis=0)
                    pred = np.argmax(pred)
                    if useLeNet:
                        pred += 1
                    grid += str(pred)
                else:
                    grid += "."

        answer = solver.solve(grid)

        if answer:
            if not grid_already_detected(grid):
                print("Original Grid\n")
                solver.display(solver.grid_values(grid))
                print("\nSolution\n")
                solver.display(answer)
            answer_digits = list(answer.values())

            font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
            for i in range(9):
                for j in range(9):
                    digit = answer_digits[i * 9 + j]
                    text_size = cv2.getTextSize(digit, font, 1, 2)[0]
                    textX = j * cell_width + (cell_width - text_size[0]) // 2
                    textY = i * cell_height + (cell_height + text_size[1]) // 2
                    if grid[i * 9 + j] == ".":
                        cv2.putText(sudoku, digit, (textX, textY), font, 1, (0, 255, 0), 2)

            h, mask = cv2.findHomography(keypts_grid, keypts_img)
            just_grid = cv2.warpPerspective(sudoku, h, (width, height))
            frame = cv2.add(img_without_grid, just_grid)
            cv2.drawContours(frame, [keypts_img.reshape(4, 1, 2)], -1, (0, 255, 0), 2)


        else:
            print("Couldn't solve grid.")

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) == ord('q'):
        break

remove_temp()
cap.release()
cv2.destroyAllWindows()
