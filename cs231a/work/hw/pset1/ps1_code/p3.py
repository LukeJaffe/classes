#!/usr/bin/env python3

# CS231A Homework 1, Problem 3
import numpy as np
from utils import mat2euler
import math

def compute_vanishing_point(points):
    '''
    COMPUTE_VANISHING_POINTS
    Arguments:
        points - a list of all the points where each row is (x, y). Generally,
                it will contain four points: two for each parallel line.
                You can use any convention you'd like, but our solution uses the
                first two rows as points on the same line and the last
                two rows as points on the same line.
    Returns:
        vanishing_point - the pixel location of the vanishing point
    '''
    # Unpack points
    (k1x, k1y), (k2x, k2y), (l1x, l1y), (l2x, l2y) = points

    # Compute slopes
    mk = (k2y - k1y) / (k2x - k1x)
    ml = (l2y - l1y) / (l2x - l1x)

    # Compute intercepts
    bk = k1y - (mk * k1x)
    bl = l1y - (ml * l1x)

    # Compute intersection x
    vx = (bl - bk) / (mk - ml)

    # Compute intersection y
    vy = mk * vx + bk

    return vx, vy

def compute_K_from_vanishing_points(vanishing_points):
    '''
    COMPUTE_K_FROM_VANISHING_POINTS
    Arguments:
        vanishing_points - a list of vanishing points

    Returns:
        K - the intrinsic camera matrix (3x3 matrix)
    '''
    ### Compute w using SVD
    vp = vanishing_points
    # Construct system of equations using vanishing points
    A = np.array([
        [vp[0][0]*vp[1][0]+vp[0][1]*vp[1][1], vp[0][0]+vp[1][0], vp[0][1]+vp[1][1], 1],
        [vp[0][0]*vp[2][0]+vp[0][1]*vp[2][1], vp[0][0]+vp[2][0], vp[0][1]+vp[2][1], 1],
        [vp[1][0]*vp[2][0]+vp[1][1]*vp[2][1], vp[1][0]+vp[2][0], vp[1][1]+vp[2][1], 1],
    ])
    # Perform SVD on system of equations
    u, s, vt = np.linalg.svd(A, full_matrices=True)
    # Solution w to Aw = 0 is last row of matrix V
    w = vt.T[:, -1]

    # Test w
    print('\nTest SVD:')
    null = np.dot(A, w)
    print('null:', null)

    # Construct omega (W) using the elements of w
    W = np.array([
        [w[0], 0,    w[1]],
        [0,    w[0], w[2]],
        [w[1], w[2], w[3]]
    ])

    # Test omega
    print('\nTest omega:')
    vp1 = np.array(vp[0]+(1,))[:, np.newaxis]
    vp2 = np.array(vp[1]+(1,))[:, np.newaxis]
    vp3 = np.array(vp[2]+(1,))[:, np.newaxis]
    print('null1:', np.dot(np.dot(vp1.T, W), vp2))
    print('null2:', np.dot(np.dot(vp1.T, W), vp3))
    print('null3:', np.dot(np.dot(vp2.T, W), vp3))

    # Compute K inverse from omega using cholesky factorization
    C = np.linalg.cholesky(W)
    # Take the (pseudo-)inverse to get K
    K = np.linalg.pinv(C.T)
    # Normalize K
    K /= K[-1, -1]

    return K

def compute_angle_between_planes(vanishing_pair1, vanishing_pair2, K):
    '''
    COMPUTE_K_FROM_VANISHING_POINTS
    Arguments:
        vanishing_pair1 - a list of a pair of vanishing points computed from lines within the same plane
        vanishing_pair2 - a list of another pair of vanishing points from a different plane than vanishing_pair1
        K - the camera matrix used to take both images

    Returns:
        angle - the angle in degrees between the planes which the vanishing point pair comes from2
    '''
    # Compute omega inverse using camera matrix
    W_inv = np.dot(K, K.T) 

    # Unpack vanishing points and add 1 to end
    vanishing_pair1[0] += (1,)
    vanishing_pair1[1] += (1,)
    vanishing_pair2[0] += (1,)
    vanishing_pair2[1] += (1,)
    v1, v2 = np.array(vanishing_pair1)
    v3, v4 = np.array(vanishing_pair2)

    # Compute l1 and l2
    l1 = np.cross(v1, v2)
    l2 = np.cross(v3, v4)

    # Compute cosine of angle using omega
    cos_theta = l1.T.dot(W_inv.dot(l2))/(np.sqrt(l1.T.dot(W_inv.dot(l1)))*np.sqrt(l2.T.dot(W_inv.dot(l2))))
    theta = np.arccos(cos_theta)

    # Convert from radians to degrees
    theta_deg = np.degrees(theta)

    return theta_deg


def compute_rotation_matrix_between_cameras(vanishing_points1, vanishing_points2, K):
    '''
    COMPUTE_K_FROM_VANISHING_POINTS
    Arguments:
        vanishing_points1 - a list of vanishing points in image 1
        vanishing_points2 - a list of vanishing points in image 2
        K - the camera matrix used to take both images

    Returns:
        R - the rotation matrix between camera 1 and camera 2
    '''
    # Unpack vanishing points
    pa1, pa2, pa3 = vanishing_points1[:, :, np.newaxis]
    pb1, pb2, pb3 = vanishing_points2[:, :, np.newaxis]

    # Group vanishing points by image
    va = np.concatenate([pa1, pa2, pa3], axis=1).T
    vb = np.concatenate([pb1, pb2, pb3], axis=1).T

    # Add ones to vanishing points 
    one_row = np.ones((va.shape[0], 1))
    va = np.concatenate([va, one_row], axis=1)
    vb = np.concatenate([vb, one_row], axis=1)

    # Compute directions of vanishing points for each image
    K_inv = np.linalg.pinv(K)
    dau = K_inv.dot(va.T)
    dbu = K_inv.dot(vb.T)

    # Normalize to unit vectors
    da = dau/np.linalg.norm(dau, axis=0)
    db = dbu/np.linalg.norm(dbu, axis=0)

    # Solve for rotation
    da_inv = np.linalg.pinv(da)
    R = np.dot(db, da_inv)

    # Check
    print('\nCheck R:')
    a1, a2, a3 = da.T
    b1, b2, b3 = db.T
    print(b1, np.dot(R, a1)) 
    print(b2, np.dot(R, a2)) 
    print(b3, np.dot(R, a3)) 

    return R

if __name__ == '__main__':
    # Part A: Compute vanishing points
    v1 = compute_vanishing_point(np.array([[674,1826],[2456,1060],[1094,1340],[1774,1086]]))
    v2 = compute_vanishing_point(np.array([[674,1826],[126,1056],[2456,1060],[1940,866]]))
    v3 = compute_vanishing_point(np.array([[1094,1340],[1080,598],[1774,1086],[1840,478]]))

    v1b = compute_vanishing_point(np.array([[314,1912],[2060,1040],[750,1378],[1438,1094]]))
    v2b = compute_vanishing_point(np.array([[314,1912],[36,1578],[2060,1040],[1598,882]]))
    v3b = compute_vanishing_point(np.array([[750,1378],[714,614],[1438,1094],[1474,494]]))

    # Part B: Compute the camera matrix
    vanishing_points = [v1, v2, v3]
    print("Intrinsic Matrix:\n",compute_K_from_vanishing_points(vanishing_points).astype(int))

    K_actual = np.array([[2448.0, 0, 1253.0],[0, 2438.0, 986.0],[0,0,1.0]])
    print()
    print("Actual Matrix:\n", K_actual.astype(int))

    # Part D: Estimate the angle between the box and floor
    floor_vanishing1 = v1
    floor_vanishing2 = v2
    box_vanishing1 = v3
    box_vanishing2 = compute_vanishing_point(np.array([[1094,1340],[1774,1086],[1080,598],[1840,478]]))
    angle = compute_angle_between_planes([floor_vanishing1, floor_vanishing2], [box_vanishing1, box_vanishing2], K_actual)
    print()
    print("Angle between floor and box:", angle)

    # Part E: Compute the rotation matrix between the two cameras
    rotation_matrix = compute_rotation_matrix_between_cameras(np.array([v1, v2, v3]), np.array([v1b, v2b, v3b]), K_actual)
    print()
    print("Rotation between two cameras:\n", rotation_matrix)
    z,y,x = mat2euler(rotation_matrix)
    print
    print("Angle around z-axis (pointing out of camera): %f degrees" % (z * 180 / math.pi))
    print("Angle around y-axis (pointing vertically): %f degrees" % (y * 180 / math.pi))
    print("Angle around x-axis (pointing horizontally): %f degrees" % (x * 180 / math.pi))
