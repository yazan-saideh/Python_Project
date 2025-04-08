def preprocess_iamge(image):
    pass

def augmnet_image():
    pass


def Resize(image, new_height, new_width):
    image_row = len(image)
    image_col = len(image[0])

    out_put_image = [[0 for _ in range(new_width)] for _ in range(new_height)]

    row_scale = (image_row - 1) / (new_height - 1)
    col_scale = (image_col - 1) / (new_width - 1)

    for i in range(new_height):
        y = i * row_scale
        for j in range(new_width):
            x = j * col_scale
            out_put_image[i][j] = bilinear_interpolation(image, y, x)

    return out_put_image


def Normalize(image):
    pass

def bilinear_interpolation(image, y, x):
    image_row = len(image)
    image_col = len(image[0])
    x0 = round(x)
    y0 = round(y)
    x1 = min(x0, image_col - 1)
    y1 = min(y0, image_row - 1)
    dx = x1 - x
    dy = y1 - y
    final = image[y0][x0] * (1 - dx) * (1 - dy) + image[y1][x0] * dy * (1 - dx) + image[y0][x1] * (1 - dy) * dx + \
            image[y1][x1] * dy * dx
    return round(final)
