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
    
    #I'm trying to think of a way to vectorize this for efficiency
    #LMK if you have any ideas
    h,w = points.shape[:2]
    #The output array
    outs = np.zeros((h,w,2))
    #Loop the points
    for i in range(h):
        for j in range(w):
            x,y,z = points[i,j]
            u,v,t = np.dot(K,np.dot(Rt,np.array([[x],[y],[z],[1]])))
            if t != 0:
                u = u/t
                v = v/t
            outs[i,j,:] = (u,v)
    return outs


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

    #Things we need to deproject:
    Ki = np.linalg.inv(K)
    Ri = np.eye(4)
    Ri[:3,:3] = Rt[:,:3]
    Ri = np.transpose(Ri)
    Ri2 = np.linalg.inv(Ri)
    
    #Doing this carefully because prior errors
    Ti = np.eye(4)
    T = np.ones((4,1))
    T[:3,0] = Rt[:,3]
    T = np.dot(Ri,T)
    Ti[:,3] = T[:,0]
    Tinv = np.linalg.inv(Ti)
    P = np.zeros((4,3))
    P[:3,:] = np.eye(3)
    P[3,2] = 1
    
    #Designate the corners as np arrays
    bL = np.array([
        [0],
        [0],
        [1]],dtype=np.float32)
    bR = np.array([
        [width-1],
        [0],
        [1]],dtype=np.float32)
    tL = np.array([
        [0],
        [height-1],
        [1]],dtype=np.float32)
    tR = np.array([
        [width-1],
        [height-1],
        [1]],dtype=np.float32)
    
    #Deproject factoring in depth
    bL = np.dot(Tinv, np.dot(Ri, np.dot(P, depth*np.dot(Ki,bL))))
    bR = np.dot(Tinv, np.dot(Ri, np.dot(P, depth*np.dot(Ki,bR))))
    tL = np.dot(Tinv, np.dot(Ri, np.dot(P, depth*np.dot(Ki,tL))))
    tR = np.dot(Tinv, np.dot(Ri, np.dot(P, depth*np.dot(Ki,tR))))
    
    #Norm if necessary, we should never get 0, though
    if bL[3,0] != 1:
        w = bL[3,0]
        bL /= w
        bR /= w
        tL /= w
        tR /= w
    
    #Put the corners into an output array
    points = np.zeros((2,2,3),dtype=np.float32)
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

    raise NotImplementedError()
