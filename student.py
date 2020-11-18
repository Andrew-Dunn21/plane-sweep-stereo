import time
from math import floor
import numpy as np
import cv2
from scipy.sparse import csr_matrix

def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- 3 x 3 camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """
    
    h,w,d = points.shape
    if d != 3:
        raise Exception("These points don't add up...")
    #Set up an output array
    proj = np.zeros((1,h*w, 2), dtype=np.float32)
    #Copy points for safety
    pints = np.copy(points)
    #Smash the pints for easy peasy looping
    pints = pints.reshape((1, h*w,d))
    #All we do is ride that loop
    for i in range(h*w):
        #Get the 3D point from pints
        x,y,z = pints[0,i,:]
        v = np.array([[x],[y],[z],[1]], dtype=np.float32)
        camv = np.dot(Rt,v)
        if camv[2,0] != 0:
            camv /= camv[2,0]
        else:
            raise Exception("This is sus behavior")
        pixv = np.dot(K,camv)
        if pixv[2,0] != 0:
            pixv /= pixv[2,0]
        else:
            raise Exception("Just full of sus today")
        proj[0,i,:] = pixv[:2,0]
    #Reshape and return proj
    return proj.reshape((h,w,2))


def unproject_corners_impl(K, width, height, depth, Rt):
    """
    Unproject the corners of a height-by-width image
    into world coordinates at depth d.
    Input:
        K -- 3x3 camera intrinsics calibration matrix
        width -- width of hte image in pixels
        height -- height of hte image in pixels
        depth -- depth the 3D point is unprojected to
        Rt -- 3 x 4 camera extrinsics calibration matrix
    Output:
        points -- 2 x 2 x 3 array of 3D positions of the image corners
    """
    #Pick up the pieces to unproject
    Ki = np.linalg.inv(K)
    Ri = np.eye(4,dtype=np.float32)
    Ri[:3,:] = Rt
    Ri = np.linalg.inv(Ri)
    P = np.zeros((4,3), dtype=np.float32)
    P[:3,:] = np.eye(3, dtype=np.float32)
    #Because I'm lazy
    h = height-1
    w = width-1
    #Make an output
    points = np.zeros((2,2,3), dtype=np.float32)
    #Establish corners
    bL = np.array([
        [0],
        [0],
        [1]],dtype=np.float32)
    bR = np.array([
        [w],
        [0],
        [1]],dtype=np.float32)
    tL = np.array([
        [0],
        [h],
        [1]],dtype=np.float32)
    tR = np.array([
        [w],
        [h],
        [1]],dtype=np.float32)
    #No loop, directly handle
    bL = np.dot(P, depth*np.dot(Ki, bL))
    bL[3,0] = 1
    bL = np.dot(Ri, bL)
    bR = np.dot(P, depth*np.dot(Ki, bR))
    bR[3,0] = 1
    bR = np.dot(Ri, bR)
    tL = np.dot(P, depth*np.dot(Ki, tL))
    tL[3,0] = 1
    tL = np.dot(Ri, tL)
    tR = np.dot(P, depth*np.dot(Ki, tR))
    tR[3,0] = 1
    tR = np.dot(Ri, tR)
    
    #Manually set output
    points[0,0,:] = bL[:3,0]
    points[0,1,:] = bR[:3,0]
    points[1,0,:] = tL[:3,0]
    points[1,1,:] = tR[:3,0]
    
    return points

def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are flattened into vectors with the default numpy row
    major order.  For example, the following 3D numpy array with shape
    2 (channels) x 2 (height) x 2 (width) patch...

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    gets unrolled using np.reshape into a vector in the following order:

    v = [ x111, x121, x211, x221, x112, x122, x212, x222 ]

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer side length of (square) NCC patch region; must be odd
    Output:
        normalized -- height x width x (channels * ncc_size**2) array
    """
    h, w, c = image.shape

    normalized = np.zeros((h, w, c, ncc_size, ncc_size), dtype=np.float32)

    k = ncc_size // 2 # half-width of the patch size

    # the following code fills in `normalized`, which can be thought
    # of as a height-by-width image where each pixel is
    #   a channels-by-ncc_size-by-ncc_size array

    for i in range(ncc_size):
        for j in range(ncc_size):
            # i, j is the top left corner of the the patch
            # so the (i=0,j=0) is the pixel in the top-left corner of the patch
            # which is an offset of (-k, -k) from the center pixel

            # example: image is 10x10x3; ncc_size is 3
            # the image pixels for the top left of all patches come from
            # (0, 0) thru (7, 7) because (7, 7) is an offset of -1, -1 
            # (corresponding to i=0, j=0) from the bottom-right-most patch, 
            # which is centered at (8, 8)
            # generalizing to patch halfsize k, it's (h-2k, w-2k)
            # generalizing to any offset into the patch,
            #   the top left will be i, j
            #   the bottom right will be h-2k + i, w-2k + j
            normalized[k:h-k, k:w-k, :, i, j] = image[i:h-2*k+i, j:w-2*k+j, :]

    # For each patch, subtract out its per-channel mean
    # Then divide the patch by its (not-per-channel) vector norm.
    # Patches with norm < 1e-6 should become all zeros.

    # compute and subtract per-channel mean
    means = np.mean(normalized, axis=(3, 4))
    normalized[:, :, :] -= means[:, :, :, np.newaxis, np.newaxis]

    # reshape patch into vector
    new_norm = normalized[:, :].reshape(h, w, -1)

    # compute vector norm and divide each patch by it if norm is non-zero
    v_norm = np.linalg.norm(new_norm, axis=2)
    v_norm[v_norm < 1e-6] = 0
    new_norm[v_norm > 0] = new_norm[v_norm > 0] / v_norm[v_norm > 0, np.newaxis]

    return new_norm


def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """

    h, w, p = image1.shape

    ncc = np.zeros((h, w))

    # compute the dot product for each patch and store in ncc array
    # I'm not sure how to vectorize this right now
    for i in range(h):
        for j in range(w):
            ncc[i, j] = np.dot(image1[i, j], image2[i, j])

    return ncc
