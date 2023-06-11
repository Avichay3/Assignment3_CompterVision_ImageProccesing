import sys
from typing import List

import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 211780267

# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------

" Write a function which takes an image and returns the optical flow by using the LK algorithm:  "
def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10,
                win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    # Convert images to grayscale if necessary.
    # "All functions should be able to accept both gray-scale and color images"
    if len(im1.shape) > 2:  # if the shape has more than 2 dimensions
        img1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = im1
    if len(im2.shape) > 2:  # if the shape has more than 2 dimensions
        img2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    else:
        img2_gray = im2

    # check if win_size (window size) is odd.
    if win_size % 2 == 0:
        return "win_size must be an odd number"
    # proper handling of the window boundaries and ensures that the analysis is centered around the
    # desired pixel with an equal number of pixels on each side. Use in floor division.
    half_win_size = win_size // 2

    # calculate image gradients
    # "In order to compute the optical flow, you will first need to compute the gradients Ix and Iy and then
    # over a window centered around each pixel we calculate"
    Ix, Iy = cv2.Sobel(img2_gray, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(img2_gray, cv2.CV_64F, 0, 1, ksize=3)
    # now calculates the difference in pixel intensity values between the two images,
    # the 'It' represents the temporal gradient(difference in pixel intensity)
    It = img2_gray.astype(np.float64) - img1_gray.astype(np.float64)

    original_points = []  # for store the original points (pixel coordinates)
    vec_per_point = []  # for store the corresponding optical flow vectors for each point

    # iterate over image blocks with the step size from the input
    for row in range(half_win_size, im1.shape[0] - half_win_size, step_size):
        for column in range(half_win_size, im1.shape[1] - half_win_size, step_size):
            # extract windowed patches
            #  helps us by selecting and isolating a specific region of interest in the image for further
            #  analysis and estimation of optical flow.
            window = img2_gray[row - half_win_size: row + half_win_size + 1,
                     column - half_win_size: column + half_win_size + 1]
            # based on the current row and column indices:
            Ix_windowed = Ix[row - half_win_size: row + half_win_size + 1,
                          column - half_win_size: column + half_win_size + 1]
            Iy_windowed = Iy[row - half_win_size: row + half_win_size + 1,
                          column - half_win_size: column + half_win_size + 1]
            It_windowed = It[row - half_win_size: row + half_win_size + 1,
                          column - half_win_size: column + half_win_size + 1]
            # flatten the windowed gradient images into 1D arrays
            A = np.column_stack((Ix_windowed.flatten(), Iy_windowed.flatten()))
            b = -np.expand_dims(It_windowed.flatten(), axis=1)  # subtract a dimension

            ATA = np.dot(A.T, A)  # A.T it is the transpose matrix of A, and we calculate the matrixes multiplication
            ATA_eig_vals = np.linalg.eigvals(
                ATA)  # numpy function that computes the eigenvalues of matrix. I love python!!

            # the next check helps us filter out points where the optical flow estimation might be not good
            if ATA_eig_vals[0] <= 1 or ATA_eig_vals[1] / ATA_eig_vals[0] >= 100:
                continue

            curr_vec = np.dot(np.linalg.inv(ATA), b) # calculate the optical flow vector by multiplying the inverse of 'ATA' matrix with 'b' matrix
            original_points.append([column, row])  # append the current pixel to original_points list
            vec_per_point.append(
                [curr_vec[0, 0], curr_vec[1, 0]])  # append the corresponding optical flow vector to the second list
    return original_points, vec_per_point


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
    # Create pyramids for both images
    im1_pyrs = gaussianPyr(img1, k)[::-1]
    im2_pyrs = gaussianPyr(img2, k)[::-1]

    points = []
    vectors = []

    # Iterate over image pyramids
    for i in range(k):
        # Perform optical flow on the current pyramid level
        curr_points, curr_vectors = opticalFlow(im1_pyrs[i], im2_pyrs[i], step_size=stepSize, win_size=winSize)

        # Scale points and vectors by the appropriate factor
        scale_factor = 2 ** i
        curr_points_scaled = [[point[0] * scale_factor, point[1] * scale_factor] for point in curr_points]
        curr_vectors_scaled = [[vector[0] * scale_factor, vector[1] * scale_factor] for vector in curr_vectors]

        # Update points and vectors
        for j, point in enumerate(curr_points_scaled):
            if point not in points:
                points.append(point)
                vectors.append(curr_vectors_scaled[j])
            else:
                index = points.index(point)
                vectors[index][0] += curr_vectors_scaled[j][0]
                vectors[index][1] += curr_vectors_scaled[j][1]

    return np.array((np.array(vectors), np.array(points), 2))


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    param im1: image 1 in grayscale format.
    param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    # calculate optical flow using the LK algorithm
    points, vectors = opticalFlow(im1.astype(np.float), im2.astype(np.float), step_size=20, win_size=5)

    # Filter out small changes using median
    med_u = np.median(vectors[:, 0])
    med_v = np.median(vectors[:, 1])
    filtered_vectors = vectors[np.logical_and(np.abs(vectors[:, 0]) > med_u, np.abs(vectors[:, 1]) > med_v)]

    # Calculate average movement
    t_x = np.median(filtered_vectors[:, 0]) * 2
    t_y = np.median(filtered_vectors[:, 1]) * 2

    mat = np.float32([
        [1, 0, t_x],
        [0, 1, t_y],
        [0, 0, 1]
    ])
    return mat


def bestAngle(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Find the best angle between two images (0-359) by minimizing the mean squared error.
    :param img1: First image
    :param img2: Second image
    :return: The best angle
    """
    best_angle = 0
    best_fit = float("inf")

    for angle in range(360):
        rotation_mat = getRotationMatrix(angle)
        rotated_img = applyRotation(img1, rotation_mat)
        fit = computeFit(img2, rotated_img)

        if fit < best_fit:
            best_fit = fit
            best_angle = angle

        if best_fit == 0:
            break

    return best_angle


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    Find the rigid transformation matrix using the Lucas-Kanade algorithm.
    :param im1: First image in grayscale format
    :param im2: Second image after rigid transformation
    :return: Rigid transformation matrix
    """
    best_angle = bestAngle(im1, im2)
    rotation_mat = getRotationMatrix(best_angle)
    rotated_img = applyRotation(im1, rotation_mat)
    translation_mat = findTranslationLK(rotated_img, im2)
    rigid_mat = combineTransformations(translation_mat, rotation_mat)

    return rigid_mat


def getRotationMatrix(angle: float) -> np.ndarray:
    """
    Get the rotation transformation matrix for a given angle.
    :param angle: Rotation angle in degrees
    :return: Rotation transformation matrix
    """
    angle_rad = math.radians(angle)
    cos_value = math.cos(angle_rad)
    sin_value = math.sin(angle_rad)
    rotation_mat = np.array([[cos_value, -sin_value, 0],
                             [sin_value, cos_value, 0],
                             [0, 0, 1]], dtype=np.float32)
    return rotation_mat


def applyRotation(img: np.ndarray, rotation_mat: np.ndarray) -> np.ndarray:
    """
    Apply the rotation transformation to an image.
    :param img: Input image
    :param rotation_mat: Rotation transformation matrix
    :return: Rotated image
    """
    rotated_img = cv2.warpPerspective(img, rotation_mat, img.shape[::-1])
    return rotated_img


def computeFit(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute the fit between two images using mean squared error.
    :param img1: First image
    :param img2: Second image
    :return: Mean squared error fit
    """
    fit = mean_squared_error(img1, img2)
    return fit


def combineTransformations(translation_mat: np.ndarray, rotation_mat: np.ndarray) -> np.ndarray:
    """
    Combine translation and rotation matrices to get the final rigid transformation matrix.
    :param translation_mat: Translation matrix
    :param rotation_mat: Rotation matrix
    :return: Rigid transformation matrix
    """
    rigid_mat = translation_mat @ rotation_mat
    return rigid_mat



def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    Find the translation matrix using the correlation-based method.
    :param im1: First image in grayscale format.
    :param im2: Second image after translation.
    :return: Translation matrix.
    """
    # Compute correlation matrix
    correlation = computeCorrelation(im1, im2)

    # Find highest correlation point
    p1x, p1y, p2x, p2y = findHighestCorrelation(correlation, im2.shape)

    # Compute translation matrix
    translation_mat = computeTranslationMatrix(p1x, p1y, p2x, p2y)

    return translation_mat

def computeCorrelation(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    Compute the correlation matrix between two images.
    :param im1: First image.
    :param im2: Second image.
    :return: Correlation matrix.
    """
    # Perform FFT and correlation
    fft_im1 = np.fft.fft2(im1)
    fft_im2 = np.fft.fft2(im2)
    correlation = np.fft.fftshift(np.fft.ifft2(fft_im1 * np.conj(fft_im2))).real

    return correlation

def findHighestCorrelation(correlation: np.ndarray, im2_shape: tuple) -> tuple:
    """
    Find the coordinates of the highest correlation point in the correlation matrix.
    :param correlation: Correlation matrix.
    :param im2_shape: Shape of the second image.
    :return: Coordinates of the highest correlation point (x1, y1, x2, y2).
    """
    # Find index of highest correlation value
    max_index = np.unravel_index(np.argmax(correlation), correlation.shape)

    # Get coordinates of the highest correlation point
    p1y, p1x = max_index
    p2y, p2x = np.array(im2_shape) // 2

    return p1x, p1y, p2x, p2y

def computeTranslationMatrix(p1x: int, p1y: int, p2x: int, p2y: int) -> np.ndarray:
    """
    Compute the translation matrix based on the correlation points.
    :param p1x: x-coordinate of the first correlation point.
    :param p1y: y-coordinate of the first correlation point.
    :param p2x: x-coordinate of the second correlation point.
    :param p2y: y-coordinate of the second correlation point.
    :return: Translation matrix.
    """
    # Calculate translation values
    t_x = p2x - p1x - 1
    t_y = p2y - p1y - 1

    # Construct translation matrix
    translation_mat = np.array([[1, 0, t_x], [0, 1, t_y], [0, 0, 1]], dtype=np.float)

    return translation_mat





def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    min_error = np.float('inf')
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
    param im1: input image 1 in grayscale format.
    param im2: input image 2 in grayscale format.
    param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    pass


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    pass


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    pass


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    pass


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray,
             mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """
    pass
