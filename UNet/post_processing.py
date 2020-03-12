import glob
import numpy as np
import cv2


def correct_dilatation(pixel_value, pixel_position_x, pixel_position_y, image, momentum):
    if pixel_value > 255 * 0.90:
        y_size = 1
    elif pixel_value > 255 * 6:
        y_size = 2
    elif pixel_value > 255 * 5:
        y_size = 4
    elif pixel_value > 255 * 0.4:
        y_size = 6
    elif pixel_value > 255 * 0.3:
        y_size = 8
    elif pixel_value > 255 * 0.2:
        y_size = 12
    elif pixel_value > 255 * 0.1:
        y_size = 16
    elif pixel_value > 255 * 0.05:
        y_size = 24
    else:
        image[pixel_position_y, pixel_position_x] = 0
        return momentum

    new_momentum = int(round((momentum + y_size) / 4))

    kernel = np.ones((new_momentum, 1), np.uint8)
    top_y = pixel_position_y - new_momentum
    top_y = 0 if top_y < 0 else top_y

    image[top_y: pixel_position_y, [pixel_position_x]] = \
        cv2.dilate(image[top_y: pixel_position_y, [pixel_position_x]], kernel)
    return new_momentum


images_path = glob.glob("./results/images/*.png")
labels_path = glob.glob("./results/labels/*.png")
predictions_path = glob.glob("./results/predictions/*.png")

predictions_list = []

# Group the corresponding image and label together
for prediction in predictions_path:
    predictions_list.append(cv2.bitwise_not(cv2.imread(prediction, 0)))

predictions = np.asarray(predictions_list)

test_prediction = predictions[1].copy()
test_prediction = cv2.bitwise_not(test_prediction)
test_prediction = np.array(np.clip(test_prediction * 1.1, 0, 255), dtype=np.uint8)
test_prediction = cv2.medianBlur(test_prediction, 5)

momentum = 1

for y in range(1, len(test_prediction)):
    for x in range(0, len(test_prediction[y])):
        # print("Y: {0} - X: {1} - Value: {2}".format(y, x, test_prediction[y][x]))
        momentum = correct_dilatation(pixel_value=test_prediction[y][x],
                                      pixel_position_y=y,
                                      pixel_position_x=x,
                                      image=test_prediction,
                                      momentum=momentum)


smooth_image = cv2.pyrUp(test_prediction)
smooth_image = cv2.medianBlur(smooth_image, 11)
smooth_image = cv2.pyrDown(smooth_image)
_, smooth_image = cv2.threshold(smooth_image, 10, 255, cv2.THRESH_BINARY)
# erode = cv2.blur(test_prediction, (5, 5))
# erode = cv2.erode(erode, np.ones((2, 2), np.uint8), iterations=1)
# kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
# erode = cv2.filter2D(erode, -1, kernel)

cv2.imshow('Starting image', predictions[1])
cv2.imshow('Test', cv2.bitwise_not(test_prediction))
cv2.imshow('Smooth Image', cv2.bitwise_not(smooth_image))
cv2.waitKey(0)

# kernel = np.ones((10, 10), np.uint8)
# initial_image = predictions[0] # [:, 0:256] == [y, x]
#
# result = cv2.dilate(initial_image, kernel, iterations=1)
#
# pre_result_erosion = cv2.dilate(predictions[0], np.ones((10, 15), np.uint8), iterations=1)
# result_erosion = cv2.erode(pre_result_erosion, np.ones((10, 15), np.uint8), iterations=1)
# result_erosion_dilate = cv2.dilate(result_erosion, kernel, iterations=1)
#
# cv2.imshow('Input', predictions[0])
# cv2.imshow('Dilation', result)
# cv2.imshow('Erosion', result_erosion)
# cv2.imshow('Dilation with erosion', result_erosion_dilate)
# cv2.waitKey(0)
