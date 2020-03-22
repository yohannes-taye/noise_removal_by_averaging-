import numpy as np
from math import floor, ceil
from PIL import Image
from numpy import asarray

def get_array_value(x: int, y: int, array: np.ndarray):
    """Returns the value of the array at position x,y."""
    return array[y, x]

def convert_to_red(img: np.ndarray):
    # with np.nditer(img, op_flags=['readwrite']) as it:
    #     for x in it:
    #         x[...] = []
    
    img.setflags(write=1)
    for i, label_i in enumerate(img): 
        for j, label_j in enumerate(img): 
            #can i modify it now? 
            img[i][j] = [0, 0, 0]
            print(img[i][j])

'''
A function to convert from an ndarray and save 
it to the specified location
'''
def save_image(img: np.ndarray, address):
    image = Image.fromarray(img)

    return image.save(address)


def bilinear_interpolation2(x: float, y: float, img: np.ndarray) -> float:
    """Returns the bilinear interpolation of a pixel in the image.
    :param x: x-position to interpolate
    :param y: y-position to interpolate
    :param img: image, where the pixel should be interpolated
    :returns: value of the interpolated pixel
    """
    if x < 0 or y < 0:
        raise ValueError("x and y pixel position have to be positive!")
    if img.shape[1] - 1 < x or img.shape[0] - 1 < y:
        raise ValueError(
            "x and y pixel position have to be smaller than image" "dimensions."
        )

    x_rounded_up = int(ceil(x))
    x_rounded_down = int(floor(x))
    y_rounded_up = int(ceil(y))
    y_rounded_down = int(floor(y))

    ratio_x = x - x_rounded_down
    ratio_y = y - y_rounded_down

    interpolate_x1 = interpolate(
        get_array_value(x_rounded_down, y_rounded_down, img),
        get_array_value(x_rounded_up, y_rounded_down, img),
        ratio_x,
    )
    interpolate_x2 = interpolate(
        get_array_value(x_rounded_down, y_rounded_up, img),
        get_array_value(x_rounded_up, y_rounded_up, img),
        ratio_x,
    )
    interpolate_y = interpolate(interpolate_x1, interpolate_x2, ratio_y)

    return interpolate_y

def generate_image(): 
    array = np.zeros([100, 200], dtype=np.uint8)

    for x in range(200): 
        for y in range(100): 
            if (x % 16) // 8 == (y % 16) // 8: 
                array[y, x] = 0
            else: 
                array[y, x] = 255
    
    return array

def nearest_neighbour_interpolation(img: np.ndarray, scale: float): 

    # Generate blank image with new dimention 
    # Project original the newly generated image on 
    #  on the original image to locate pixels 
    # Assign the RGB value to the new image 
    
    row, col, _ = img.shape
    new_row = row * scale 
    new_col = col * scale 

    print("original size: ", (row, col))
    print("new size: ", (new_row, new_col))
    new_image = np.zeros([new_row , new_col, 3], dtype=np.uint8)
    
    for x in range(new_row): 
        for y in range(new_col): 
            proj_x = round(x / scale)
            proj_y = round(y / scale)
            if proj_x < row and proj_y < col: 
                new_image[x, y]  = img[proj_x, proj_y]
                # print((proj_x, proj_y))
    save_image(new_image, "./img/omg.jpeg")
    # for x in range(row): 
    #     for y in range(col): 
    #         print(img[x, y])
def load_images(source: str): 
    images = []
    source = "./img/lena_noised_"
    for i in range(1, 4): 
        temp_img_src = source + str(i) + ".jpg"
        print("loading image " + temp_img_src)
        temp_image = Image.open(temp_img_src)
        temp_array = asarray(temp_image)
        images.append(temp_array)
    return images 

def remove_noise_avg(images, count): 
    if len(images) > 0: 
        sample_img = images[0]
        row, col, _ = sample_img.shape 
        new_image = np.zeros([row, col, 3], dtype=np.uint8)
        for x in range(row): 
                for y in range(col): 
                    temp_avg = [0, 0, 0]
                    for i in range(count): 
                        temp_avg = np.add(temp_avg, images[i][x][y])

                    new_image[x][y] =  temp_avg // count
                    
        save_image(new_image, "./img/ohh_my_god_it_finally_worked.jpg")


    '''
    1. Load all the images first 
    2. Sum up all the RGB components of all the loaded images
    3. Average the values and assign to the new pixel 
    '''
    
    
def bilinear_interpolation(img: np.ndarray, scale: float): 
    row, col, _ = img.shape
    new_row = row * scale 
    new_col = col * scale 

    print("original size: ", (row, col))
    print("new size: ", (new_row, new_col))
    new_image = np.zeros([new_row , new_col, 3], dtype=np.uint8)
    
    for x in range(new_row): 
        for y in range(new_col): 
            proj_x = round(x / scale)
            proj_y = round(y / scale)
            if proj_x < row and proj_y < col: 
                x_l = floor(proj_x)                
                x_u = ceil(proj_x)
                ratio_x = proj_x - x_l

                y_l = floor(proj_y)
                y_u = ceil(proj_y)
                ratio_y = proj_y - y_l

                interpolate_x1 = interpolate_rgb(img[x_l, y_l], img[x_u, y_l], ratio_x)
                interpolate_x2 = interpolate_rgb(img[x_l, y_u], img[x_u, y_u], ratio_x)
                interpolate_y = interpolate_rgb(interpolate_x1, interpolate_x2, ratio_y)
                new_image[x, y] = interpolate_y
    save_image(new_image, "./img/ohh_my_god_it_finally_worked.jpeg")
    
def interpolate_rgb(rgb1, rgb2, ratio): 
    r = interpolate(rgb1[0], rgb2[0], ratio)
    g = interpolate(rgb1[1], rgb2[1], ratio)
    b = interpolate(rgb1[2], rgb2[2], ratio)

    return np.ndarray((3,), buffer=np.array([r, g, b]),  dtype=int)
    
def interpolate(first_value: float, second_value: float, ratio: float) -> float:
    """Interpolate with a linear weighted sum."""
    return first_value * (1 - ratio) + second_value * ratio




if __name__ == "__main__":
    images = load_images("")
    
    remove_noise_avg(images, len(images))
    
