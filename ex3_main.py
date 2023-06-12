from matplotlib import pyplot as plt
import numpy as np
from ex3_utils import *
import time


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def lkDemo(img_path):
    print("------------------------LK Demo--------------------------")
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -.2],
                  [0, 1, -.1],
                  [0, 0, 1]], dtype=float)
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    st = time.time()
    pts, uv = opticalFlow(img_1.astype(float), img_2.astype(float), step_size=20, win_size=5)
    et = time.time()

    print("Time: {:.2f}".format(et - st))
    print(np.median(uv, 0))
    print(np.mean(uv, 0))

    displayOpticalFlow(img_2, pts, uv)


def hierarchicalkDemo(img_path):
    print("-------------------------------Hierarchical LK Demo Test-----------------------------------")
    im1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    im1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    im1 = cv2.resize(im1, (0, 0), fx=0.5, fy=0.5)
    t = np.array([[1, 0, -0.2],
                  [0, 1, -0.1],
                  [0, 0, 1]], dtype=float)
    im2 = cv2.warpPerspective(im1, t, (im1.shape[1], im1.shape[0]))

    ans = opticalFlowPyrLK(im1.astype(float), im2.astype(float), 4, 20, 5)

    pts = np.array([])
    uv = np.array([])
    for i in range(ans.shape[0]):
        for j in range(ans.shape[1]):
            if ans[i][j][1] != 0 and ans[i][j][0] != 0:
                uv = np.append(uv, ans[i][j][0])
                uv = np.append(uv, ans[i][j][1])
                pts = np.append(pts, j)
                pts = np.append(pts, i)
    pts = pts.reshape(int(pts.shape[0] / 2), 2)
    uv = uv.reshape(int(uv.shape[0] / 2), 2)
    print(np.median(uv, 0))
    print(np.mean(uv, 0))
    displayOpticalFlow(im2, pts, uv)



def compareLK(img_path):
    print("----------------------------Compare LK & Hierarchical LK Test--------------------------------")
    im1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    im1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    im1 = cv2.resize(im1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -0.2],
                  [0, 1, -0.1],
                  [0, 0, 1]], dtype=float)
    im2 = cv2.warpPerspective(im1, t, (im1.shape[1], im1.shape[0]))
    pts, uv = opticalFlow(im1.astype(float), im2.astype(float), step_size=20, win_size=5)

    ans = opticalFlowPyrLK(im1.astype(float), im2.astype(float), 4, 20, 5)
    ptspyr = np.array([])
    uvpyr = np.array([])
    for i in range(ans.shape[0]):
        for j in range(ans.shape[1]):
            if ans[i][j][1] != 0 and ans[i][j][0] != 0:
                uvpyr = np.append(uvpyr, ans[i][j][0])
                uvpyr = np.append(uvpyr, ans[i][j][1])
                ptspyr = np.append(ptspyr, j)
                ptspyr = np.append(ptspyr, i)
    ptspyr = ptspyr.reshape(int(ptspyr.shape[0] / 2), 2)
    uvpyr = uvpyr.reshape(int(uvpyr.shape[0] / 2), 2)

    f, ax = plt.subplots(3, 1, figsize=(6, 8))
    if len(im2.shape) == 2:
        adjustColor(ax, im2, pts, uv, ptspyr, uvpyr, "gray")
    else:
        adjustColor(ax, im2, pts, uv, ptspyr, uvpyr, None)

    ax[2].quiver(pts[:, 0], pts[:, 1], uv[:, 0], uv[:, 1], color='r')
    ax[2].quiver(ptspyr[:, 0], ptspyr[:, 1], uvpyr[:, 0], uvpyr[:, 1], color='y')
    f.tight_layout()
    plt.show()

def adjustColor(ax, img, points, points_vectors, pyr_points, pyr_points_vectors, color_map):
    ax[0].imshow(img, cmap=color_map)
    ax[0].quiver(points[:, 0], points[:, 1], points_vectors[:, 0], points_vectors[:, 1], color='r')
    ax[1].imshow(img, cmap=color_map)
    ax[1].quiver(pyr_points[:, 0], pyr_points[:, 1], pyr_points_vectors[:, 0], pyr_points_vectors[:, 1], color='r')
    ax[2].imshow(img, cmap=color_map)

def displayOpticalFlow(img: np.ndarray, pts: np.ndarray, uvs: np.ndarray):
    plt.imshow(img, cmap='gray')
    plt.quiver(pts[:, 0], pts[:, 1], uvs[:, 0], uvs[:, 1], color='r')
    plt.show()






# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------

def translationlkdemo(img_path):
    print("------------------------Translation LK Demo Test---------------------------")
    # loading the original image and resize it
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=0.5, fy=0.5)
    img_1 = cv2.resize(img_1, (0, 0), fx=0.5, fy=0.5)
    # define the original transformation matrix
    orig_mat = np.array([[1, 0, -50],[0, 1, -50],[0, 0, 1]], dtype=float)
    # warp the original image using the original matrix
    img_2 = cv2.warpPerspective(img_1, orig_mat, img_1.shape[::-1])
    cv2.imwrite('img1.jpg', img_2)
    # find the translation matrix using LK algorithm
    my_mat = findTranslationCorr(img_1, img_2)
    # warp the original image using the obtained matrix
    my_warp = cv2.warpPerspective(img_1, my_mat, (img_1.shape[1], img_1.shape[0]))
    cv2.imwrite('img2.jpg', my_warp)
    # creating the plot with subplots, all in one
    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    # display the original image in the first subplot
    axs[0, 0].set_title('original Image')
    axs[0, 0].imshow(img_1, cmap='gray')

    # display the warped image using the original matrix in the second subplot
    axs[0, 1].set_title('warped Image')
    axs[0, 1].imshow(img_2, cmap='gray')

    # display the difference between the warped image and the LK-warp image in the third subplot
    axs[0, 2].set_title('difference')
    axs[0, 2].imshow(img_2 - my_warp, cmap='gray')

    # Display the warped image using the original matrix (CV Translation) in the fourth subplot
    axs[1, 0].set_title('cv Translation')
    axs[1, 0].imshow(img_2, cmap='gray')

    # display the warped image (My Translation) in the fifth subplot
    axs[1, 1].set_title('my Translation')
    axs[1, 1].imshow(my_warp, cmap='gray')

    # hiding the empty subplot in the last column
    axs[1, 2].axis('off')

    # adjust the layout and display the plot
    plt.tight_layout()
    plt.show()

    # calculating the MSE between the LK-warp image and the warped image using of the original matrix
    mse = MSE(my_warp, img_2)
    print("MSE =", mse)



def rigidlkdemo(img_path):
    print("----------------------------Rigid lk demo Test------------------------------")
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    angle = 0.8
    rigid = np.array([[np.cos(angle), -np.sin(angle), -1],
                      [np.sin(angle), np.cos(angle), -1],
                      [0, 0, 1]], dtype=np.float32)

    img_2 = cv2.warpPerspective(img_1, rigid, img_1.shape[::-1])
    cv2.imwrite('img_rig1.jpg', img_2)
    f, ax = plt.subplots(1, 2)
    ax[0].set_title('CV Rigid')
    ax[0].imshow(img_2, cmap='gray')
    my_rigid = findRigidLK(img_1, img_2)
    my_warp = cv2.warpPerspective(img_1, my_rigid, img_1.shape[::-1])
    cv2.imwrite('img_rig2.jpg', my_warp)
    ax[1].set_title('My Rigid')
    ax[1].imshow(my_warp, cmap='gray')
    plt.show()


def translationcorrdemo(img_path):
    print("------------------------------Translation corr demo Test-----------------------------")
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    orig_mat = np.array([[1, 0, 35],
                         [0, 1, 80],
                         [0, 0, 1]], dtype=float)
    img_2 = cv2.warpPerspective(img_1, orig_mat, img_1.shape[::-1])
    cv2.imwrite('img_trans1.jpg', img_2)
    my_mat = findTranslationCorr(img_1, img_2)
    my_warp = cv2.warpPerspective(img_1, my_mat, (img_1.shape[1], img_1.shape[0]))
    cv2.imwrite('img_trans2.jpg', my_warp)
    f, ax = plt.subplots(1, 3)
    ax[0].set_title('cv translation')
    ax[0].imshow(img_2, cmap='gray')

    ax[1].set_title('my translation')
    ax[1].imshow(my_warp, cmap='gray')

    ax[2].set_title('difference')
    ax[2].imshow(img_2 - my_warp, cmap='gray')

    plt.show()
    print("MSE = ", MSE(my_warp, img_2))


def rigidcorrdemo(img_path):
    print("------------------------------Rigid corr demo Test--------------------------------")
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=.5)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=.5)

    theta = -58 * np.pi / 180
    orig_mat = np.array([[np.cos(theta), -np.sin(theta), -5],
                         [np.sin(theta), np.cos(theta), -1],
                         [0, 0, 1]], dtype=float)
    img_2 = cv2.warpPerspective(img_1, orig_mat, img_1.shape[::-1])
    cv2.imwrite('imRigidB1.jpg', img_2)
    st = time.time()
    my_mat = findRigidCorr(img_1.astype(float), img_2.astype(float))
    et = time.time()

    print("Time: {:.4f}".format(et - st))
    print("my mat: \n", my_mat, "\n\n original mat: \n", orig_mat)
    my_rigid = cv2.warpPerspective(img_1, my_mat, img_1.shape[::-1])
    cv2.imwrite('imRigidB2.jpg', my_rigid)
    f, ax = plt.subplots(1, 2)
    ax[0].set_title('CV Rigid')
    ax[0].imshow(img_2, cmap='gray')

    ax[1].set_title('My Rigid')
    ax[1].imshow(my_rigid, cmap='gray')

    plt.show()
    print("MSE =", MSE(my_rigid, img_2))


def imageWarpingDemo(img_path):
    print("---------------------------Image Warping Demo Test------------------------------")

    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, 2],
                  [0, 1, 1],
                  [0, 0, 1]], dtype=float)

    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    st = time.time()
    im2 = warpImages(img_1.astype(float), img_2.astype(float), t)
    et = time.time()
    print("Time: {:.4f}".format(et - st))
    f, ax = plt.subplots(1, 2)
    ax[0].set_title('My warping')
    ax[0].imshow(im2)

    ax[1].set_title('CV warping')
    ax[1].imshow(img_2)
    plt.show()




# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def pyrGaussianDemo(img_path):
    print("---------------------------------Gaussian Pyramid Demo Test-----------------------------------")
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 4
    gau_pyr = gaussianPyr(img, lvls)
    h, w = gau_pyr[0].shape[:2]
    canv_h = h
    widths = np.cumsum([w // (2 ** i) for i in range(lvls)])
    widths = np.hstack([0, widths])
    canv_w = widths[-1]
    canvas = np.zeros((canv_h, canv_w, 3))
    for lv_idx in range(lvls):
        h = gau_pyr[lv_idx].shape[0]
        canvas[:h, widths[lv_idx]:widths[lv_idx + 1], :] = gau_pyr[lv_idx]

    plt.imshow(canvas)
    plt.show()


def pyrLaplacianDemo(img_path):
    print("-------------------------------Laplacian Pyramid Demo Test--------------------------------")
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY) / 255
    # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 7
    lap_pyr = laplaceianReduce(img, lvls)
    re_lap = laplacianExpand(lap_pyr)
    f, ax = plt.subplots(2, lvls + 1)
    plt.gray()
    for i in range(lvls):
        ax[0, i].imshow(lap_pyr[i])
        ax[1, i].hist(lap_pyr[i].ravel(), 256, [lap_pyr[i].min(), lap_pyr[i].max()])

    ax[0, -1].set_title('Original Image')
    ax[0, -1].imshow(re_lap)
    ax[1, -1].hist(re_lap.ravel(), 256, [0, 1])
    plt.show()


def blendDemo():
    print("-----------------------------------Blending demo Test----------------------------------------")
    im1 = cv2.cvtColor(cv2.imread('input/sunset .jpg'), cv2.COLOR_BGR2RGB) / 255
    im2 = cv2.cvtColor(cv2.imread('input/cat .jpg'), cv2.COLOR_BGR2RGB) / 255
    mask = cv2.cvtColor(cv2.imread('input/mask_cat.jpg'), cv2.COLOR_BGR2RGB) / 255
    n_blend, im_blend = pyrBlend(im1, im2, mask, 4)
    f, ax = plt.subplots(2, 3)
    ax[0, 0].imshow(im1)
    ax[0, 1].imshow(im2)
    ax[0, 2].imshow(mask)
    ax[1, 0].imshow(n_blend)
    ax[1, 1].imshow(np.abs(n_blend - im_blend))
    ax[1, 2].imshow(im_blend)
    plt.show()


def MSE(a: np.ndarray, b: np.ndarray):  # mean squared error calculator
    return np.square(a - b).mean()


def main():
    print("ID:", myID())
    #lkDemo('input/boxMan.jpg')
    #hierarchicalkDemo('input/boxMan.jpg')
    #compareLK('input/boxMan.jpg')
    #translationlkdemo('input/pyr_bit.jpg')
    #rigidlkdemo('input/cat .jpg')
    #translationcorrdemo('input/cat .jpg')
    #rigidcorrdemo('input/sunset .jpg')
    #imageWarpingDemo('input/sunset .jpg')
    #pyrGaussianDemo('input/pyr_bit.jpg')
    #pyrLaplacianDemo('input/pyr_bit.jpg')
    #blendDemo()


if __name__ == '__main__':
    main()