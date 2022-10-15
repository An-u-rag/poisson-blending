import cv2
import numpy as np
from scipy import sparse

from align_target import align_target


def poisson_blend(source_image, target_image, target_mask):
    # source_image: image to be cloned
    # target_image: image to be cloned into
    # target_mask: mask of the target image
    cv2.imshow('Mask', target_mask)
    cv2.imshow('source', source_image)
    cv2.imshow('target', target_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # get number of nonzero points and their indices in the mask
    imp_pix = np.argwhere(target_mask)
    n = len(imp_pix)

    # Loop through height, width and channels of the target image
    h, w, c = target_image.shape

    # Define A and b matrices
    A = np.zeros((n, n))
    b = np.zeros((c, n))

    # Create a matrix of index count on the size of target image for reference
    index = 0
    indices = np.zeros((h, w), np.int32)
    for k in range(n):
        i = imp_pix[k][0]
        j = imp_pix[k][1]
        indices[i, j] = index
        index += 1

    for i in range(h):
        for j in range(w):
            # Skip pixel if it does not lie in the target mask
            if target_mask[i, j] == 0:
                continue

            for k in range(c):
                A[indices[i, j], indices[i, j]] = 4
                # Gradient vector calculation for each color channel
                v_left = 0
                if target_mask[i, j-1] == 0:
                    b[k, indices[i,j]] += target_image[i, j-1, k]
                else:
                    A[indices[i, j], indices[i, j - 1]] = -1
                    b[k, indices[i,j]] += int(source_image[i, j, k]) - int(source_image[i, j-1, k])

                v_right = 0
                if target_mask[i, j+1] == 0:
                    b[k, indices[i, j]] += target_image[i, j+1, k]
                else:
                    A[indices[i, j], indices[i, j + 1]] = -1
                    b[k, indices[i,j]] += int(source_image[i, j, k]) - int(source_image[i, j+1, k])

                v_up = 0
                if target_mask[i - 1, j] == 0:
                    b[k, indices[i, j]] += target_image[i-1, j, k]
                else:
                    A[indices[i, j], indices[i - 1, j]] = -1
                    b[k, indices[i,j]] += int(source_image[i, j, k]) - int(source_image[i-1, j, k])

                v_down = 0
                if target_mask[i + 1, j] == 0:
                    b[k, indices[i, j]] += target_image[i+1, j, k]
                else:
                    A[indices[i, j], indices[i + 1, j]] = -1
                    b[k, indices[i,j]] += int(source_image[i, j, k]) - int(source_image[i+1, j, k])


    A = sparse.csr_matrix(A)
    b = sparse.csr_matrix(b)
    v_b = sparse.linalg.spsolve(A, b[0].T)
    v_g = sparse.linalg.spsolve(A, b[1].T)
    v_r = sparse.linalg.spsolve(A, b[2].T)

    print(b.shape)

    print(f'Blue error is {np.linalg.norm(A * v_b - b[0])}')
    print(f'Green error is {np.linalg.norm(A * v_g - b[1])}')
    print(f'Red error is {np.linalg.norm(A * v_r - b[2])}')

    for k in range(n):
        i = imp_pix[k][0]
        j = imp_pix[k][1]
        target_image[i, j, 0] = abs((v_b[k]))
        target_image[i, j, 1] = abs((v_g[k]))
        target_image[i, j, 2] = abs((v_r[k]))

    blended_image = target_image.astype(np.uint8)

    return blended_image

if __name__ == '__main__':
    # read source and target images
    source1_path = './venv/images2/source1.jpg'
    source2_path = './venv/images2/source2.jpg'
    target_path = './venv/images2/target.jpg'
    source1_image = cv2.imread(source1_path)
    source2_image = cv2.imread(source2_path)
    target_image = cv2.imread(target_path)

    # align target images
    im1_source, mask1 = align_target(source1_image, target_image)

    # poisson blend
    blended_image = poisson_blend(im1_source, target_image, mask1)

    im2_source, mask2 = align_target(source2_image, blended_image)

    # poisson blend
    blended_image = poisson_blend(im2_source, blended_image, mask2)

    cv2.imshow("Blended Image", blended_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()