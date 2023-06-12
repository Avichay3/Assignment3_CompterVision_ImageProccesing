
import math
from typing import List
import numpy as np
from numpy.linalg import LinAlgError
import cv2
import pygame
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')


def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from)
    """
    return 211780267


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------

def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size
    """
    def preprocess_image(image):
        # convert image to grayscale if it has more than 2 dimensions
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            plt.gray()
        return image

    def create_windowed_patches(image, row, col):
        # create window patches from the image centered given row and column
        half_win_size = win_size // 2
        return image[
            row - half_win_size: row + half_win_size + 1,
            col - half_win_size: col + half_win_size + 1,
        ].flatten()

    image_shape = im1.shape
    im1 = preprocess_image(im1)
    im2 = preprocess_image(im2)

    if win_size % 2 == 0:
        return "window_size should be an odd number"

    kernel_x = np.array([[-1, 0, 1]])
    kernel_y = kernel_x.T

    Ix = cv2.filter2D(im2, -1, kernel_x, borderType=cv2.BORDER_REPLICATE)
    Iy = cv2.filter2D(im2, -1, kernel_y, borderType=cv2.BORDER_REPLICATE)
    It = im2 - im1

    original_points = []  # list for the original points.
    vec_per_point = []  # list for the OP vectors for each point in @original_points.

    for row in range(step_size, image_shape[0] - win_size + 1, step_size):
        for col in range(step_size, image_shape[1] - win_size + 1, step_size):
            # Create windowed patches for Ix, Iy, and It at the current row and col
            Ix_windowed = create_windowed_patches(Ix, row, col)
            Iy_windowed = create_windowed_patches(Iy, row, col)
            It_windowed = create_windowed_patches(It, row, col)

            # construct the matrix A and the vector b
            A = np.vstack((Ix_windowed, Iy_windowed)).T  # A = [Ix, Iy]
            b = (A.T @ (-1 * It_windowed).T).reshape(2, 1)

            # calculate ATA and check the eigenvalues
            ATA = A.T @ A
            ATA_eig_vals = np.sort(np.linalg.eigvals(ATA))
            if ATA_eig_vals[0] <= 1 or ATA_eig_vals[1] / ATA_eig_vals[0] >= 100:
                continue
            # compute the inverse of ATA and solve for the current flow vector
            ATA_INV = np.linalg.inv(ATA)
            curr_vec = ATA_INV @ b

            # store the original point and its corresponding flow vector
            original_points.append([col, row])
            vec_per_point.append([curr_vec[0, 0], curr_vec[1, 0]])
    return np.array(original_points), np.array(vec_per_point)

def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int,
                     stepSize: int, winSize: int) -> np.ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """

    # if the image is RGB (3 dimensions), convert to gray
    if len(img1.shape) > 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    if len(img2.shape) > 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    # if the images don't have the same shape we will throw an error
    if img1.shape != img2.shape:
        raise Exception("The images must be in the same size")

    # first, find the pyramids for img1 and also img2.
    firstImgPyramid = gaussianPyr(img1, k)
    firstImgPyramid.reverse()
    secondImgPyramid = gaussianPyr(img2, k)
    secondImgPyramid.reverse()

    # find the OP for the first img
    original_points, vec_per_point = opticalFlow(firstImgPyramid[0], secondImgPyramid[0], stepSize, winSize)

    for i in range(1, k):
        orig_pyr_ind, vec_pyr_ind = opticalFlow(firstImgPyramid[i], secondImgPyramid[i], stepSize, winSize)
        for j in range(len(original_points)):
            original_points[j] = [element * 2 for element in original_points[j]]
            vec_per_point[j] = [element * 2 for element in vec_per_point[j]]

        # add the OP vectors for each of the pyramids.
        for pixel, uv_current in zip(orig_pyr_ind, vec_pyr_ind):
            if not np.any(np.all(original_points == pixel, axis=1)):
                original_points = np.append(original_points, [pixel], axis=0)
                vec_per_point = np.append(vec_per_point, [uv_current], axis=0)
            else:
                index = np.where(np.all(original_points == pixel, axis=1))[0][0]
                vec_per_point[index][0] += uv_current[0]
                vec_per_point[index][1] += uv_current[1]

    ans = np.zeros(shape=(img1.shape[0], img1.shape[1], 2))
    # reshape to 3D array (X, Y, 2)
    for ind in range(len(original_points)):
        px = original_points[ind][1]
        py = original_points[ind][0]
        ans[px][py][0] = vec_per_point[ind][0]
        ans[px][py][1] = vec_per_point[ind][1]
    return ans


def optFlow(im1: np.ndarray, im2: np.ndarray, blockSize=11, maxCorners=5000, qualityLevel=0.00001, minDistance=1):
    # convert images to uint8 if necessary
    im1 = im1.astype('uint8')
    im2 = im2.astype('uint8')
    # set the parameters for corner detection
    features = dict(maxCorners=maxCorners, qualityLevel=qualityLevel, minDistance=minDistance, blockSize=blockSize)
    # detecting the best corners in the first image
    best_points = cv2.goodFeaturesToTrack(im1, mask=None, **features)  # find strong corners or features in image
    # optical flow estimation using the Lucas-Kanade algorithm with pyramids
    movements = cv2.calcOpticalFlowPyrLK(im1, im2, best_points, None)[0]
    return movements, best_points


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def calculate_median_optical_flow(mat: np.ndarray):
    """
    This function calculates the median optical flow vector from the input matrix.
    :param mat: Array of optical flow vectors (shape: (n, 1, 2))
    """
    num_vectors = mat.shape[0]
    u_values = mat[:, 0, 0]
    v_values = mat[:, 0, 1]
    median_u = np.median(u_values)
    median_v = np.median(v_values)
    return median_u, median_v


def find_best_translation_vector(img1: np.ndarray, img2: np.ndarray, mat: np.ndarray):
    """
    This function returns the best translation vector that aligns img1 to img2.
    :param img1: First image
    :param img2: Second image
    :param mat: Array of optical flow vectors (shape: (n, 1, 2))
    """
    best_fit = float("inf")
    best_vec = [0, 0]
    for ind in range(len(mat)):
        tx = mat[ind, 0, 0]
        ty = mat[ind, 0, 0]

        translation_matrix = np.array([[1, 0, tx], [0, 1, ty],[0, 0, 1]], dtype=float)
        curr_img2 = cv2.warpPerspective(img1, translation_matrix, img1.shape[::-1])
        curr_fit = ((img2 - curr_img2) ** 2).sum()

        if curr_fit < best_fit:
            best_fit = curr_fit
            best_vec = [tx, ty]

        if curr_fit == 0:
            break

    return best_vec



def calculate_translation_matrix(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    Calculates the translation matrix based on the optical flow between im1 and im2.
    :param im1: First image (grayscale)
    :param im2: Second image (grayscale)
    :return: Translation matrix
    """
    movements, best_points = optFlow(im1, im2)
    translation_vector = movements - best_points
    tx, ty = find_best_translation_vector(im1, im2, translation_vector)
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])



def find_best_rotation_angle(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Finds the best rotation angle between two images by evaluating all possible angles (0-359).
    :param img1: First image
    :param img2: Second image
    :return: Best rotation angle
    """
    best_angle = 0
    best_fit = float("inf")
    for angle in range(360):  # Iterate over all possible angles
        theta = math.radians(angle)
        cos_value = math.cos(theta)
        sin_value = math.sin(theta)
        # create the rotation transformation matrix
        rotation_mat = np.array([[cos_value, -sin_value, 0], [sin_value, cos_value, 0], [0, 0, 1]], dtype=np.float32)
        # apply rotation to img1
        rotated_img = cv2.warpPerspective(img1, rotation_mat, img1.shape[::-1])
        # calculate the mean squared error (MSE) between rotated_img and img2
        curr_fit = mean_squared_error(img2, rotated_img)

        if curr_fit < best_fit:
            best_fit = curr_fit
            best_angle = angle

        if best_fit == 0:
            break

    return best_angle


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    Finds the rigid transformation matrix using the Lucas-Kanade algorithm.
    :param im1: Input image 1 in grayscale format.
    :param im2: Image 1 after applying rigid transformation.
    :return: Rigid transformation matrix.
    """
    best_angle = find_best_rotation_angle(im1, im2)
    theta = math.radians(best_angle)  # create the rotation matrix based on the best angle
    cos_value = math.cos(theta)
    sin_value = math.sin(theta)
    rotation_mat = np.array([[cos_value, -sin_value, 0], [sin_value, cos_value, 0], [0, 0, 1]], dtype=np.float32)
    rotated_im1 = cv2.warpPerspective(im1, rotation_mat, im1.shape[::-1])  # applying rotation to im1
    translation_mat = calculate_translation_matrix(rotated_im1, im2)  # calculating the translation matrix to complete the rigid transformation
    rigid_mat = translation_mat @ rotation_mat  # combine the rotation and translation matrices
    return rigid_mat


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    Finds the translation matrix using correlation-based matching.
    :param im1: Input image 1 in grayscale format.
    :param im2: Image 1 after translation.
    :return: Translation matrix by correlation.
    """
    img_shape = np.max(im1.shape) // 2
    im1FFT = np.fft.fft2(np.pad(im1, img_shape))
    im2FFT = np.fft.fft2(np.pad(im2, img_shape))
    prod = im1FFT * im2FFT.conj()
    res = np.fft.fftshift(np.fft.ifft2(prod))
    correlation = res.real[1 + img_shape:-img_shape + 1, 1 + img_shape:-img_shape + 1]
    p1y, p1x = np.unravel_index(np.argmax(correlation), correlation.shape)
    p2y, p2x = np.array(im2.shape) // 2
    translation_mat = np.array([[1, 0, (p2x - p1x - 1)], [0, 1, (p2y - p1y - 1)], [0, 0, 1]], dtype=float)
    return translation_mat



def getAngle(point1, point2):
    """
    This function calculate the angle between @point1 to @point2 by checking the intersection
    point of these points by creating two lines to the mass of center (0,0).
    :param point1: [x,y]
    :param point2: [x,y]
    :return: float - angle
    """
    vec1 = pygame.math.Vector2(point1[0] - 0, point1[1] - 0)
    vec2 = pygame.math.Vector2(point2[0] - 0, point2[1] - 0)
    return vec1.angle_to(vec2)


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    min_error = float('inf')
    best_rotation_mat = best_rotated_img = 0
    # for every angle between 0 and 359 we calculate the correlation.
    # and we find the best angle by checking the correlation.
    # and then we find the best rotated image by checking the correlation.
    for angle in range(360):
        rotation_mat = np.array([[math.cos(angle), -math.sin(angle), 0],
                      [math.sin(angle), math.cos(angle), 0],
                      [0, 0, 1]], dtype=np.float32)
        after_rotation = cv2.warpPerspective(im1, rotation_mat, im1.shape[::-1])  # rotating the image
        curr_mse = mean_squared_error(im2, after_rotation)  # calculating the error
        if curr_mse < min_error:
            min_error = curr_mse
            best_rotation_mat = rotation_mat
            best_rotated_img = after_rotation.copy()
        if curr_mse == 0:
            break

    translation = findTranslationCorr(best_rotated_img, im2)  # finding the translation from the rotated
    return translation @ best_rotation_mat  # combining the translation and the rotation.


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Warps image2 (im2) onto image1 (im1) using the transformation matrix T.
    :param im1: Input image 1 in grayscale format.
    :param im2: Input image 2 in grayscale format.
    :param T: 3x3 transformation matrix where each pixel in im2 is mapped under homogenous coordinates to im1 (p2 = Tp1)
    :return: warp image 2 according to T and display both image1 and the wrapped version of the image2 in the same figure.
    """
    if len(im1.shape) > 2:  # 3 or more dimensions
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    if len(im2.shape) > 2:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    warped_img = np.zeros(im2.shape, dtype=im2.dtype)  # create an empty array for the warped image with the same shape as im2
    inv_T = np.linalg.inv(T)   # calculate the inverse of the transformation matrix T
    # iterate over each pixel in im2
    for row in range(im2.shape[0]):
        for col in range(im2.shape[1]):
            # calculate the coordinates of the point in im1 using inverse transformation
            current_vec = np.array([row, col, 1]) @ inv_T
            dx, dy = int(round(current_vec[0])), int(round(current_vec[1]))
            # check if the point has valid coordinates in im1
            if 0 <= dx < im1.shape[0] and 0 <= dy < im1.shape[1]:
                warped_img[row, col] = im1[dx, dy]  # Put the pixel in the warped image

    return warped_img


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------




def get_sigma(k_size: int):
    """
    This function calculate the sigma value for a given kernel size in the context of Gaussian filtering
    """
    return 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    kernel = cv2.getGaussianKernel(5, sigma=get_sigma(5))
    kernel = np.dot(kernel, kernel.T)
    gauss_pyramid = gaussianPyr(img, levels)
    gauss_pyramid.reverse()

    lap_pyramid = [gauss_pyramid[0]]
    for i in range(len(gauss_pyramid) - 1):
        # expand to next level
        expandedImg = (gaussExpand(gauss_pyramid[i], kernel))

        try:  # if we can add the two images without changing their size
            curr_layer = (cv2.subtract(gauss_pyramid[i + 1], expandedImg))
        except Exception:
            if len(expandedImg) != len(gauss_pyramid[i + 1]):
                expandedImg = expandedImg[0:len(gauss_pyramid[i + 1]), :]
            if len(expandedImg[0]) != len(gauss_pyramid[i + 1][0]):
                expandedImg = expandedImg[:, 0:len(gauss_pyramid[i + 1][0])]
            curr_layer = (cv2.subtract(gauss_pyramid[i + 1], expandedImg))

        lap_pyramid.append(curr_layer)
    lap_pyramid.reverse()
    return lap_pyramid


def laplacianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Reshapes the original image from a Laplacian pyramid.
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    kernel_size = 5  # set the kernel size for the Gaussian kernel
    sigma = get_sigma(kernel_size)  # calculate the sigma value based on the kernel size
    kernel = cv2.getGaussianKernel(kernel_size, sigma) @ cv2.getGaussianKernel(kernel_size, sigma).T  # create the Gaussian kernel
    output_img = lap_pyr[-1]  # initialize the output image with the top level of the Laplacian pyramid
    for i in range(len(lap_pyr) - 1, 0, -1):
        expanded_img = gaussExpand(output_img, kernel)  # expand the output image using Gaussian pyramid expansion
        try:
            output_img = cv2.add(expanded_img, lap_pyr[i - 1])  # Add the expanded image and the Laplacian layer
        except Exception:
            # Handle the case where the sizes of the expanded image and Laplacian layer do not match
            expanded_img = expanded_img[:lap_pyr[i - 1].shape[0], :lap_pyr[i - 1].shape[1]]  # resize the expanded image
            output_img = cv2.add(expanded_img, lap_pyr[i - 1])  # add the resized expanded image and the Laplacian layer

    return output_img



def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid.
    :param img: Original image.
    :param levels: Pyramid depth.
    :return: Gaussian pyramid (list of images).
    """
    gauss_pyramid = [img]  # initialize the Gaussian pyramid with the original image
    for _ in range(1, levels):
        # apply Gaussian blur to the previous level of the pyramid
        curr_img = cv2.GaussianBlur(gauss_pyramid[-1], (5, 5), 0)
        #  reducing the size of the blurred image by taking every second pixel in both dimensions
        curr_img = curr_img[::2, ::2]
        # adding the size of the blurred image to the Gaussian pyramid
        gauss_pyramid.append(curr_img)

    return gauss_pyramid



def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    """
    Expands a Gaussian pyramid level one step up
    :param img: Pyramid image at a certain level
    :param gs_k: The kernel to use in expanding
    :return: The expanded level
    """
    # check the img type and build output accordingly
    if len(img.shape) > 2:
        expanded_img = np.zeros((2 * len(img), 2 * len(img[0]), 3))
    else:
        expanded_img = np.zeros((2 * len(img), 2 * len(img[0])))
    # duplicate pixels
    expanded_img[::2, ::2] = img
    expanded_img[expanded_img < 0] = 0
    expanded_img = cv2.filter2D(expanded_img, -1, gs_k * 4, borderType=cv2.BORDER_REPLICATE)
    return expanded_img


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """
    # compute Laplacian pyramids for the input images
    first_lap_pyr = laplaceianReduce(img_1, levels=levels)
    second_lap_pyr = laplaceianReduce(img_2, levels=levels)
    # compute Gaussian pyramid for the blend mask
    gauss_pyr_mask = gaussianPyr(mask, levels=levels)
    # blend each level of the pyramids using the blend mask
    blended_pyr = [gauss_pyr_mask[i] * first_lap_pyr[i] + (1 - gauss_pyr_mask[i]) * second_lap_pyr[i] for i in
                   range(len(gauss_pyr_mask))]

    # reconstruct the blended image from the blended pyramid
    blended_img = laplacianExpand(blended_pyr)
    # finally create a naive blend by replacing pixels in image 1 with image 2 where the mask is zero
    naive_img = img_1.copy()
    naive_img[mask == 0] = img_2[mask == 0]

    return naive_img, blended_img