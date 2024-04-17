import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def update(i):
    im.set_array(image_array[i])
    return im,


files = glob.glob(r"figures/trainProgress/*.png")
image_array=[]
for my_file in files:
    image = Image.open(my_file)
    image_array.append(image)

fig, ax = plt.subplots()

im = ax.imshow(image_array[0], animated=True)

animation_fig = animation.FuncAnimation(fig, update, frames=len(image_array), interval=800, blit=True, repeat_delay=True)
plt.axis('off')
plt.show()

