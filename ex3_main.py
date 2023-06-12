from matplotlib import pyplot as plt
from ex3_utils import *
import time
import cv2


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def lkDemo(img_path):
    print("------------------LK Demo-------------------")
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -.2],
                  [0, 1, -.1],
                  [0, 0, 1]], dtype=np.float)
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    st = time.time()
    pts, uv = opticalFlow(img_1.astype(np.float), img_2.astype(np.float), step_size=20, win_size=5)
    et = time.time()

    print("Time: {:.4f}".format(et - st))
    print(np.median(uv, 0))
    print(np.mean(uv, 0))

    displayOpticalFlow(img_2, pts, uv)


def displayOpticalFlow(img: np.ndarray, pts: np.ndarray, uvs: np.ndarray):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.quiver(pts[:, 0], pts[:, 1], uvs[:, 0], uvs[:, 1], color='r')
    ax.set_title('Optical Flow')
    plt.savefig("output/lucas-kanade")
    plt.show()


def displayOpticalFlowh(img: np.ndarray, pts: np.ndarray, uvs: np.ndarray):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.quiver(pts[:, 0], pts[:, 1], uvs[:, 0], uvs[:, 1], color='r')
    ax.set_title('Hierarchical Optical Flow')
    plt.savefig("output/hierarchical lucas-kanade")
    plt.show()


def hierarchicalkDemo(img_path):
    """
    ADD TEST
    :param img_path: Image input
    :return:

    """
    print("---------------------Hierarchical LK Demo-----------------------")
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, 3],
                  [0, 1, -5],
                  [0, 0, 1]], dtype=np.float32)
    img_2 = cv2.warpPerspective(img_1, t, (img_1.shape[1], img_1.shape[0]))

    st = time.time()
    result = opticalFlowPyrLK(img_1.astype(np.float32), img_2.astype(np.float32), 4, stepSize=20, winSize=5)
    et = time.time()
    print("Time: {:.4f}".format(et - st))

    uv, pts = opticalFlowPyrLK(img_1.astype(np.float32), img_2.astype(np.float32), 4, stepSize=20, winSize=5)
    displayOpticalFlow(img_1, pts, uv)
    displayOpticalFlowh(img_1, pts, uv)


def compareLK(img_path):
    """
    ADD TEST
    Compare the two results from both functions.
    param img_path: Image input
    :return:
    """
    print("----------------Compare LK & Hierarchical LK--------------------")
    im1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    im1 = cv2.resize(im1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -0.2],
                  [0, 1, -0.1],
                  [0, 0, 1]], dtype=np.float32)
    height, width = im1.shape[:2]

    im2 = cv2.warpPerspective(im1, t, (width, height))

    x_y, u_v = opticalFlow(im1.astype(np.float32), im2.astype(np.float32), 20, 5)
    ans = opticalFlowPyrLK(im1.astype(np.float32), im2.astype(np.float32), 4, 20, 5)
    x_y_pyr = ans[..., 0, :2].reshape(-1, 2)
    u_v_pyr = ans[..., 0, 2:].reshape(-1, 2)

    fig, axes = plt.subplots(2, 2)

    titles = ['LK', 'Pyramid LK']
    for ax, title, xy, uv in zip(axes.flatten(), titles, [x_y, x_y_pyr], [u_v, u_v_pyr]):
        ax.set_title(title)
        ax.imshow(im2, cmap='gray')
        ax.quiver(xy[:, 0], xy[:, 1], uv[:, 0], uv[:, 1], color='r')

    axes[1, 1].set_title('Pyramid LK Overlay')
    axes[1, 1].quiver(x_y[:, 0], x_y[:, 1], u_v[:, 0], u_v[:, 1], color='r')
    axes[1, 1].quiver(x_y_pyr[:, 0], x_y_pyr[:, 1], u_v_pyr[:, 0], u_v_pyr[:, 1], color='g')

    plt.tight_layout()
    plt.show()


def displayOpticalFlow(img: np.ndarray, pts: np.ndarray, uvs: np.ndarray):
    plt.imshow(img, cmap='gray')
    plt.quiver(pts[:, 0], pts[:, 1], uvs[:, 0], uvs[:, 1], color='r')
    plt.show()


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def imageWarpingDemo(img_path):
    """
    ADD TEST
    :param img_path: Image input
    :return:
    """
    print("---------------------Image Warping Demo---------------------")
    image1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    t = np.array([[1, 0, 120],
                  [0, 1, -30],
                  [0, 0, 1]], dtype=float)

    image2 = cv2.warpPerspective(image1, t, image1.shape[::-1])

    plt.imshow(image1, cmap='gray')
    plt.imshow(image2, cmap='gray')

    img = warpImages(image1, image2, t)
    plt.imshow(img, cmap='gray')
    plt.show()


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def pyrGaussianDemo(img_path):
    print("------------------------Gaussian Pyramid Demo-------------------------")
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
    print("-----------------Laplacian Pyramid Demo---------------------")
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY) / 255
    # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 7

    lap_pyr = laplaceianReduce(img, lvls)
    re_lap = laplaceianExpand(lap_pyr)

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
    im1 = cv2.cvtColor(cv2.imread('input/sunset.jpg'), cv2.COLOR_BGR2RGB) / 255
    im2 = cv2.cvtColor(cv2.imread('input/cat.jpg'), cv2.COLOR_BGR2RGB) / 255
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

    cv2.imwrite('sunset_cat.png', cv2.cvtColor((im_blend * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def main():
    print("ID:", myID())
    img_path = 'input/boxMan.jpg'
    lkDemo(img_path)
    hierarchicalkDemo(img_path)
    compareLK(img_path)
    imageWarpingDemo(img_path)
    pyrGaussianDemo('input/pyr_bit.jpg')
    pyrLaplacianDemo('input/pyr_bit.jpg')
    blendDemo()


if __name__ == '__main__':
    main()
