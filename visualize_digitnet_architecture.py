import imutils
from cvTools.utils.visualize_architecture import visualize_dimensions
from cvTools.ConvNets.DigitNet import DigitNet

model = DigitNet.build(28, 28, 1, 10)
visualize_dimensions(model, "DigitNet", path="plots")

