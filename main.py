from skimage import io
from matplotlib import pylab as plt

plt.figure(figsize=(30, 30))
for ind, i in enumerate(["easy1"]):
    file_name = 'data/{}.jpg'.format(i)
    img = io.imread(file_name)
    plt.imshow(img, cmap="gray")
    plt.show()
