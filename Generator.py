from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os


class DigitGenerator:
    def __init__(self, samples_per_digit, width=28, height=28, font_size=17, test_split=0.2):
        self.fonts = [ImageFont.truetype(f'fonts\\{font}', font_size) for font in os.listdir("fonts")]
        self.num_samples_per_digit = samples_per_digit
        self.width = width
        self.height = height
        self.font_size = font_size
        self.test_split = test_split

    def split_into_train_test(self, imgs):
        """
        Splits a list of (img, label) into train and test splits.
        :param imgs: a list of (img, label)
        :return: X_train, X_test, y_train, y_test
        """
        np.random.shuffle(imgs)
        train_size = int(len(imgs) * (1-self.test_split))
        X_train, y_train = list(zip(*imgs[:train_size]))
        X_test, y_test = list(zip(*imgs[train_size:]))
        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

    def generate_digit(self, digit):
        msg = str(digit)
        samples = []
        if self.num_samples_per_digit > len(self.fonts):
            fonts = self.fonts
            fonts.extend(np.random.choice(self.fonts, self.num_samples_per_digit - len(self.fonts)))
        else:
            fonts = list(np.random.choice(self.fonts, self.num_samples_per_digit))
        for i in range(self.num_samples_per_digit):
            img = Image.new('L', (self.width, self.height), color=0)
            fnt = fonts[i]
            horizontal_lever = np.random.choice([-1, 0, 1])
            vertical_lever = np.random.choice([-1, 0, 1])
            d = ImageDraw.Draw(img)
            d.text((self.width//2 + horizontal_lever, self.height//2+vertical_lever), msg, fill=255, font=fnt, anchor="mm")
            samples.append((np.array(img), digit))
        return self.split_into_train_test(samples)

    def generate_digits(self, includeZero=True):
        start = 0
        if not includeZero:
            start = 1
        X_train, X_test, y_train, y_test = self.generate_digit(start)
        for i in range(start+1, 10):
            X_train1, X_test1, y_train1, y_test1 = self.generate_digit(i)
            X_train = np.concatenate((X_train, X_train1))
            X_test = np.concatenate((X_test, X_test1))
            y_train = np.concatenate((y_train, y_train1))
            y_test = np.concatenate((y_test, y_test1))
        return X_train, X_test, y_train, y_test


class SolutionGenerator:
    def __init__(self, cell_width, cell_height):
        self.cell_width = cell_width
        self.cell_height = cell_height

    def draw_lines(self, d, w, h, lw_thin=1, lw_thick=3, horizontal=True):
        for i in range(10):
            if i % 3 == 0:
                width = lw_thick
            else:
                width = lw_thin
            if horizontal:
                d.line([(0, (i+2)//3*lw_thick + i*self.cell_height + (i - ((i+2)//3))*lw_thin), (w, (i+2)//3*lw_thick + i*self.cell_height + (i - ((i+2)//3))*lw_thin)], fill="black", width=width)
            else:
                d.line([((i + 2) // 3 * lw_thick + i * self.cell_width + (i - ((i + 2) // 3)) * lw_thin, 0), ((i + 2) // 3 * lw_thick + i * self.cell_width + (i - ((i + 2) // 3)) * lw_thin, h)], fill="black", width=width)

    def generate_solution(self, grid, answers, save_name="solution.png"):
        lw_thin = 1; lw_thick = 3
        width = 9*self.cell_width + 4*lw_thick + 6*lw_thin
        height = 9*self.cell_height + 4*lw_thick + 6*lw_thin
        img = Image.new('RGB', (width, height), color=(255, 255, 254))
        d = ImageDraw.Draw(img)
        font = ImageFont.truetype(f'fonts\\Graduate-Regular.ttf', 18)
        self.draw_lines(d, width, height, lw_thin, lw_thick, True)
        self.draw_lines(d, width, height, lw_thin, lw_thick, False)
        for i in range(9):
            for j in range(9):
                index = 9*i+j
                cell_top_left = (j//3*lw_thick + (j-j//3)*lw_thin + j*self.cell_width, i//3*lw_thick + (i-i//3)*lw_thin + i*self.cell_height)
                cell_middle = (cell_top_left[0] + self.cell_width // 2, cell_top_left[1] + self.cell_height // 2)
                msg = answers[index]
                if grid[index] == ".":
                    ImageDraw.floodfill(img, cell_middle, (57, 255, 20))
                d.text(cell_middle, msg, fill=(0, 0, 0), font=font, anchor="mm")
        save_path = f"solutions\\generated_{save_name}.png"
        img.save(save_path, "PNG")
        print(f"Generated solution saved to {os.path.abspath(save_path)}.")
