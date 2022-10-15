import numpy as np
import cv2
from scipy import sparse

if __name__ == '__main__':
    # read source img
    img1_path = './venv/images/target.jpg'
    img2_path = './venv/images/target1.jpg'
    img3_path = './venv/images/large.jpg'
    img4_path = './venv/images/large1.jpg'
    img = cv2.imread(img4_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print('Original Range: ', np.amin(img), np.amax(img))
    cv2.imshow('Original', img)
    h, w = img.shape
    # Define A and b for reconstruction
    A = np.zeros((h * w, h * w))
    b = np.zeros(h * w)

    # implement reconstruction
    index = 0
    b_const = 210
    # Get the Gradients of the img and form A matrix
    for i in range(h):
        for j in range(w):
            # Set corner pixels to 1 in A matrix and constant value in b vector
            if (i * w + j) == 0 or (i * w + j) == w - 1 or (i * w + j) == (h - 1) * w or (i * w + j) == (h * w) - 1:
                # Constraints
                A[index, i * w + j] = 1
                b[index] = b_const
                index += 1
                continue

            if j == 0 or j == w - 1:  # vertical derivatives
                A[index, i * w + j] = 2
                A[index, (i + 1) * w + j] = -1
                A[index, (i - 1) * w + j] = -1
                b[index] = 2 * img[i, j] - img[i + 1, j] - img[i - 1, j]

            elif i == 0 or i == h - 1:  # horizontal derivatives
                A[index, i * w + j] = 2
                A[index, i * w + j + 1] = -1
                A[index, i * w + j - 1] = -1
                b[index] = 2 * img[i, j] - img[i, j + 1] - img[i, j - 1]

            else:
                A[index, i * w + j] = 4
                A[index, i * w + j + 1] = -1
                A[index, i * w + j - 1] = -1
                A[index, (i + 1) * w + j] = -1
                A[index, (i - 1) * w + j] = -1
                b[index] = 4 * img[i, j] - img[i, j - 1] - img[i + 1, j] - img[i - 1, j] - img[i, j + 1]

            index += 1


    A = sparse.csr_matrix(A)
    b = sparse.csr_matrix(b)
    img_hat = sparse.linalg.spsolve(A, b.T)
    print(f'Least Square error = {(np.linalg.norm(A*img_hat - b))}')

    img_hat = img_hat.reshape(h, w)

    # Normalization returns the original image back but nullifies the effect of b_const
    # img_hat = (img_hat - np.amin(img_hat)) * (255 / (np.amax(img_hat) - np.amin(img_hat)))
    img_hat = img_hat.astype(np.float64)/ 255.
    img = img.astype(np.float64) / 255.
    print(f'Image error = {(np.linalg.norm(img_hat - img))}')
    #print("Range of pixel intensities in reconstructed image: ", np.amin(img_hat), np.amax(img_hat))
    cv2.imshow('Reconstructed', img_hat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
