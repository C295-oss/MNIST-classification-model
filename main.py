import torch
from torch import nn
import torch.nn.functional as nnf
import numpy as np

from PyQt6.QtWidgets import QWidget, QApplication, QMainWindow, QToolBar, QDockWidget
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QAction, QImage
from PyQt6.QtCore import QSize, Qt, QPoint
import PyQt6.QtCore

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class CNN(nn.Module):
    def __init__(self,
                in_channels: int,
                class_num: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                            out_channels=16,
                            kernel_size=3,
                            padding=1
                            )
        self.conv2 = nn.Conv2d(in_channels=16,
                            out_channels=32,
                            kernel_size=3,
                            padding=1
                            )
        self.conv3 = nn.Conv2d(in_channels=32,
                            out_channels=64,
                            kernel_size=3,
                            padding=1
                            )

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(64 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, class_num)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Convolution, relu, and pooling
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Canvas(QWidget):
    def __init__(self):
        super().__init__()

        self.painted = False  # To track if something is painted on the canvas
        self.previousPoint = None
        
        # Create a low-resolution canvas (28x28)
        self.canvas = QPixmap(28, 28)
        self.canvas.fill(QColor("black"))

        # Pen for drawing on the canvas
        self.pen = QPen()
        self.pen.setColor(QColor("white"))
        self.pen.setWidth(1)  # Smaller pen size for low-resolution drawing

    def paintEvent(self, event):
        painter = QPainter(self)
        
        # Scale up the low-resolution canvas to fit the display area
        scaled_canvas = self.canvas.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio)
        painter.drawPixmap(0, 0, scaled_canvas)

    def mouseMoveEvent(self, event):
        pos = event.pos()
        self.painted = True  # Set painted flag to true

        # Map the mouse position to the low-resolution canvas coordinates
        low_res_pos = QPoint(pos.x() * 28 // 400, pos.y() * 28 // 400)

        painter = QPainter(self.canvas)
        painter.setPen(self.pen)
        
        # Smooth out drawn lines
        if self.previousPoint:
            painter.drawLine(self.previousPoint, low_res_pos)
        else:
            painter.drawPoint(low_res_pos)

        painter.end()
        self.update()
        self.previousPoint = low_res_pos

    def mouseReleaseEvent(self, event):
        self.previousPoint = None

    def clearCanvas(self):
        self.painted = False  # Set painted flag to False
        self.canvas.fill(QColor("black"))
        self.update()  # Trigger a repaint event

    def getCanvasAsTensor(self):
        # Extract image data from QPixmap
        image = self.canvas.toImage()

        width = image.width()
        height = image.height()

        # print(width, height)
        # Convert QImage to numpy array
        ptr = image.constBits()
        ptr.setsize(image.sizeInBytes())
        arr = np.array(ptr).reshape(28, 28, 4)  # Assuming the image has 4 channels (RGBA)

        # Convert to grayscale (if your model expects grayscale input)
        grayscale_arr = arr[:, :, 0]  # Take only the red channel (since R=G=B in grayscale)

        # Normalize the array if needed
        normalized_arr = grayscale_arr / 255.0  # Normalize to [0, 1] range

        # Convert to PyTorch tensor
        tensor = torch.tensor(normalized_arr, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

        return tensor


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        
        # Setup gui
        self.setFixedSize(800, 450)
        self.setWindowTitle("MNIST CNN")

        # Add a canvas
        self.canvas = Canvas()
        self.setCentralWidget(self.canvas)

        # Add a toolbar & the clear button and action
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        clear_action = QAction("Clear", self)
        clear_action.triggered.connect(self.canvas.clearCanvas)
        toolbar.addAction(clear_action)
        

        # X and Y axis for graph
        self.x = [0,1,2,3,4,5,6,7,8,9]
        self.y = [10,20,30,40,50,60,70,80,90, 100]

        # Add bar graph
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.bar(self.x, self.y)
        
        graph = FigureCanvas(self.fig)
        graph.setFixedSize(400, 400)  # Set the widget size in pixels

        dock_widget = QDockWidget("Graph", self)
        dock_widget.setWidget(graph)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock_widget)

        # Create prediction action
        predict_action = QAction("Predict", self)
        predict_action.triggered.connect(self.makePrediction)
        toolbar.addAction(predict_action)

        # Load and setup model
    def makePrediction(self):
        if self.canvas.painted:
            tensor = self.canvas.getCanvasAsTensor()
            model = CNN(in_channels=1, class_num=10)
            model.load_state_dict(torch.load('model.pth'))

            model.eval()
            with torch.no_grad():
                output = model(tensor)
                probabilities = nnf.softmax(output, dim=1).squeeze()
                predicted_digit = probabilities.argmax().item()

                print(f"Predicted Digit: {predicted_digit}")
                print(f"Prediction Probabilities: {probabilities.tolist()}")

                self.ax.clear()
                self.ax.bar(self.x, probabilities.tolist())
                self.fig.canvas.draw()
        else:
            print("Canvas empty")


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
