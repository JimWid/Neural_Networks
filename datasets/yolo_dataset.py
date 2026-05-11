import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class YOLO_Dataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, grid_size, num_boxes, num_class, transform=None):
        super().__init__()

        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir

        self.transform = transform

        self.S = grid_size
        self.B = num_boxes
        self.C = num_class

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []

        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, w, h = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, w, h])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        # Converts to cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))

        for box in boxes:
            class_label, x, y, w, h = box.tolist()
            class_label = int(class_label)

            # i, j = row, column
            i, j = int(self.S * y), int(self.S * x)

            x_cell = self.S * x - j
            y_cell = self.S * y - i
            
            # Width and Height of the bounding box, relative to the cell
            width_cell, height_cell = (w * self.S, h * self.S)

            # if object hasn't already found an object for specific cell i, j
            # set this cell that there is an object
            # this restricts each cell to only detect one object
            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] == 1

                # Box coordinates
                box_coords = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 21:25] = box_coords

                # One-Hot encoding for class_labels
                label_matrix[i, j, class_label] = 1

        return image, label_matrix


