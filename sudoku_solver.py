from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2
import imutils
from PreProcessing import Preprocessor
from DigitAugment import DigitAugmenter
from Generator import SolutionGenerator
import solve_algorithm as solver


def predict(models_, img_, augmenter_, augment_size, num_models):
    assert num_models<=5
    imgs_ = augmenter_.generate_batch(img_, augment_size, process=False)
    pred = models_[0].predict(imgs_)
    for k in range(1, num_models):
        pred += models[k].predict(imgs_)
    pred = pred.sum(axis=0)
    pred = np.argmax(pred)
    return pred


def recognize_grid(sudoku_gray, augment_size=20, num_models=5):
    grid = ""
    for i in range(9):  # i is for the y co-ordinate
        for j in range(9):  # j is for the x co-ordinate
            cell_ = sudoku_gray[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width]
            digit = preprocessor.extract_digit(cell_)
            if digit is not None:
                grid += str(predict(models, digit, augmenter, augment_size, num_models))
            else:
                grid += "."
    return grid


def putTextOnGrid(sudoku_, answer_digits_, cell_width_, cell_height_):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(9):
        for j in range(9):
            digit = answer_digits_[i * 9 + j]
            text_size = cv2.getTextSize(digit, font, 1, 2)[0]
            textX = j * cell_width_ + (cell_width_ - text_size[0]) // 2
            textY = i * cell_height_ + (cell_height_ + text_size[1]) // 2
            if grid[i * 9 + j] == ".":
                cv2.putText(sudoku_, digit, (textX, textY), font, 1, (0, 255, 0), 2)
    return sudoku_


args = imutils.get_args(single_image=True, use_video=True)
augmenter = DigitAugmenter(
    transforms={"rotation": [-10, 10], "horizontal_shift": [-0.15, 0.15], "vertical_shift": [-0.15, 0.15]},
    rescale=True)
models = [load_model(f"models\\digitnet_mnist_augment_decay_{i}.h5") for i in range(1, 6)]

img = cv2.imread(args["image"])
if img is None:
    raise FileNotFoundError("Please provide a valid path to the image.")
img = imutils.resize(img, height=800)
height, width = img.shape[:2]

debug = False
showDigitsOnly = True

preprocessor = Preprocessor(28, 28, debug, showDigitsOnly)
sudoku, sudoku_gray, keypts_img, keypts_grid, img_without_grid = preprocessor.extract_grid(img)
if sudoku is None:
    raise Exception("Couldn't detect box.")

rows, cols = sudoku.shape[:2]
cell_height, cell_width = rows // 9, cols // 9

grid = recognize_grid(sudoku_gray, augment_size=15, num_models=5)

print("Original Grid\n")
solver.display(solver.grid_values(grid))
answer = solver.solve(grid)
if not answer:
    raise Exception("Couldn't solve grid.")

print("\nSolution\n")
solver.display(answer)
answer_digits = list(answer.values())
path_name = args['image'].split('\\')[-1].split('.')[0]

generator = SolutionGenerator(60, 60)
generator.generate_solution(grid, answer_digits, save_name=path_name)

sudoku = putTextOnGrid(sudoku, answer_digits, cell_width, cell_height)
imutils.showImage(sudoku, "Solved Sudoku")

h, mask = cv2.findHomography(keypts_grid, keypts_img)
just_grid = cv2.warpPerspective(sudoku, h, (width, height))
output = cv2.add(img_without_grid, just_grid)
imutils.showImage(output, "Final Solution")

save_path = f"solutions\\original_{path_name}.png"
cv2.imwrite(save_path, output)
print(f"Solution written to {os.path.abspath(save_path)}.")
