#!/usr/bin/env python2

import numpy as np
import scipy.io as sio
import argparse
from camera import Camera
from plotting import *


# A very simple, but useful method to take the difference between the
# first and second element (usually for 2D vectors)
def diff(x):
    return x[1] - x[0]


def form_initial_voxels(xlim, ylim, zlim, num_voxels):
    '''
    FORM_INITIAL_VOXELS  create a basic grid of voxels ready for carving

    Arguments:
        xlim - The limits of the x dimension given as [xmin xmax]

        ylim - The limits of the y dimension given as [ymin ymax]

        zlim - The limits of the z dimension given as [zmin zmax]

        num_voxels - The approximate number of voxels we desire in our grid

    Returns:
        voxels - An ndarray of size (N, 3) where N is approximately equal the 
            num_voxels of voxel locations.

        voxel_size - The distance between the locations of adjacent voxels
            (a voxel is a cube)

    Our initial voxels will create a rectangular prism defined by the x,y,z
    limits. Each voxel will be a cube, so you'll have to compute the
    approximate side-length (voxel_size) of these cubes, as well as how many
    cubes you need to place in each dimension to get around the desired
    number of voxel. This can be accomplished by first finding the total volume of
    the voxel grid and dividing by the number of desired voxels. This will give an
    approximate volume for each cubic voxel, which you can then use to find the 
    side-length. The final "voxels" output should be a ndarray where every row is
    the location of a voxel in 3D space.
    '''
    # Get length of each side of space
    xsize, ysize, zsize = diff(xlim), diff(ylim), diff(zlim)
    # Compute total volume
    total_volume = xsize*ysize*zsize
    # Divid total volume by number of voxels to get volume per voxel
    voxel_volume = total_volume / num_voxels
    # SIde length of a voxel is cube root of volume
    voxel_size = np.cbrt(voxel_volume)
    # Compute position of each voxel
    voxels = np.mgrid[
        xlim[0]:xlim[1]:voxel_size,
        ylim[0]:ylim[1]:voxel_size,
        zlim[0]:zlim[1]:voxel_size].reshape(3, -1).T

    return voxels, voxel_size


def get_voxel_bounds(cameras, estimate_better_bounds = False, num_voxels = 4000):
    '''
    GET_VOXEL_BOUNDS: Gives a nice bounding box in which the object will be carved
    from. We feed these x/y/z limits into the construction of the inital voxel
    cuboid. 

    Arguments:
        cameras - The given data, which stores all the information
            associated with each camera (P, image, silhouettes, etc.)

        estimate_better_bounds - a flag that simply tells us whether to set tighter
            bounds. We can carve based on the silhouette we use.

        num_voxels - If estimating a better bound, the number of voxels needed for
            a quick carving.

    Returns:
        xlim - The limits of the x dimension given as [xmin xmax]

        ylim - The limits of the y dimension given as [ymin ymax]

        zlim - The limits of the z dimension given as [zmin zmax]

    The current method is to simply use the camera locations as the bounds. In the
    section underneath the TODO, please implement a method to find tigther bounds:
    One such approach would be to do a quick carving of the object on a grid with 
    very few voxels. From this coarse carving, we can determine tighter bounds. Of
    course, these bounds may be too strict, so we should have a buffer of one 
    voxel_size around the carved object. 
    '''
    camera_positions = np.vstack([c.T for c in cameras])
    xlim = [camera_positions[:,0].min(), camera_positions[:,0].max()]
    ylim = [camera_positions[:,1].min(), camera_positions[:,1].max()]
    zlim = [camera_positions[:,2].min(), camera_positions[:,2].max()]

    # For the zlim we need to see where each camera is looking. 
    camera_range = 0.6 * np.sqrt(diff( xlim )**2 + diff( ylim )**2)
    for c in cameras:
        viewpoint = c.T - camera_range * c.get_camera_direction()
        zlim[0] = min( zlim[0], viewpoint[2] )
        zlim[1] = max( zlim[1], viewpoint[2] )

    # Move the limits in a bit since the object must be inside the circle
    xlim = xlim + diff(xlim) / 4 * np.array([1, -1])
    ylim = ylim + diff(ylim) / 4 * np.array([1, -1])

    # Get better bounds by carving a small chunk of voxels
    if estimate_better_bounds:
        # Get initial bounds with few voxels
        voxels, voxel_size = form_initial_voxels(xlim, ylim, zlim, num_voxels)
        # Carve the voxel block
        for c in cameras:
            voxels = carve(voxels, c)  
        # Get the min and max positions in each dimension for the carved block
        xmin, ymin, zmin = voxels.min(axis=0)
        xmax, ymax, zmax = voxels.max(axis=0)
        xlim = np.array([xmin, xmax])
        ylim = np.array([ymin, ymax])
        zlim = np.array([zmin, zmax])
        # Make the minimum difference between min and max bounds
        # equal to a factor of the voxel size, 3 works well
        # for this example
        f = 3
        if diff(xlim) < f*voxel_size:
            xlim[0] -= f*voxel_size/2
            xlim[1] += f*voxel_size/2
        if diff(ylim) < f*voxel_size:
            ylim[0] -= f*voxel_size/2
            ylim[1] += f*voxel_size/2
        if diff(zlim) < f*voxel_size:
            zlim[0] -= f*voxel_size/2
            zlim[1] += f*voxel_size/2

    return xlim, ylim, zlim
    

def carve(voxels, camera):
    '''
    CARVE: carves away voxels that are not inside the silhouette contained in 
        the view of the camera. The resulting voxel array is returned.

    Arguments:
        voxels - an Nx3 matrix where each row is the location of a cubic voxel

        camera - The camera we are using to carve the voxels with. Useful data
            stored in here are the "silhouette" matrix, "image", and the
            projection matrix "P". 

    Returns:
        voxels - a subset of the argument passed that are inside the silhouette
    '''
    # Convert voxels to homogeneous coordinates
    hvoxels = np.concatenate([voxels, np.ones((voxels.shape[0], 1))], axis=1)
    # Project the voxels into pixels in the image
    hpixels = np.dot(camera.P, hvoxels.T)
    # Convert pixels from homogeneous to euclidean
    pixels = (hpixels[:2, :]/hpixels[2, :]).T
    # Get silhouette dimensions
    Y, X = camera.silhouette.shape
    # Round projected pixel positions to ints
    pixels = np.round(pixels).astype(int)
    # First, get indeces of pixels inside the bounds of the silhouette
    boundedx = np.logical_and(pixels[:, 0] > 0, pixels[:, 0] < X)
    boundedy = np.logical_and(pixels[:, 1] > 0, pixels[:, 1] < Y)
    bounded = np.logical_and(boundedx, boundedy)
    bounded_pixels = pixels[bounded]
    xidx = bounded_pixels[:, 0]
    yidx = bounded_pixels[:, 1]
    # Then, check indeces of these pixels which fall under the silhouette
    sil_idx = camera.silhouette[yidx, xidx].astype(bool)
    # Index voxels first by bounded indeces, then by silhouette indeces
    return voxels[bounded][sil_idx]

'''
ESTIMATE_SILHOUETTE: Uses a very naive and color-specific heuristic to generate
the silhouette of an object

Arguments:
    im - The image containing a known object. An ndarray of size (H, W, C).

Returns:
    silhouette - An ndarray of size (H, W), where each pixel location is 0 or 1.
        If the (i,j) value is 0, then that pixel location in the original image 
        does not correspond to the object. If the (i,j) value is 1, then that
        that pixel location in the original image does correspond to the object.
'''
def estimate_silhouette(im):
    return np.logical_and(im[:,:,0] > im[:,:,2], im[:,:,0] > im[:,:,1] )


if __name__ == '__main__':
    estimate_better_bounds = True
    use_true_silhouette = False
    frames = sio.loadmat('frames.mat')['frames'][0]
    cameras = [Camera(x) for x in frames]

    #for c in cameras:
    #    plt.imshow(c.image)
    #    plt.show()

    # Generate the silhouettes based on a color heuristic
    if not use_true_silhouette:
        for i, c in enumerate(cameras):
            c.true_silhouette = c.silhouette
            c.silhouette = estimate_silhouette(c.image)
            if i == 0:
                plt.figure()
                plt.subplot(121)
                plt.imshow(c.true_silhouette, cmap = 'gray')
                plt.title('True Silhouette')
                plt.subplot(122)
                plt.imshow(c.silhouette, cmap = 'gray')
                plt.title('Estimated Silhouette')
                plt.savefig('./plots/p1e1.png')

    # Generate the voxel grid
    # You can reduce the number of voxels for faster debugging, but
    # make sure you use the full amount for your final solution
    num_voxels = 6e6
    xlim, ylim, zlim = get_voxel_bounds(cameras, estimate_better_bounds)

    # This part is simply to test forming the initial voxel grid
    voxels, voxel_size = form_initial_voxels(xlim, ylim, zlim, 4000)
    #plot_surface(voxels, title='Initial Voxel Grid', 
    #    save=True, path='./plots/p1a.png')
    voxels, voxel_size = form_initial_voxels(xlim, ylim, zlim, num_voxels)

    # Test the initial carving
    #voxels = carve(voxels, cameras[0])
    #if use_true_silhouette:
    #    plot_surface(voxels, title='Voxels After One Iteration of Carving', 
    #        save=True, path='./plots/p1b.png')

    # Result after all carvings
    for i, c in enumerate(cameras):
        if i < 5: 
            voxels = carve(voxels, c)  
    plot_surface(voxels, voxel_size, title='Fully Carved Plot from Estimated Silhouette, with only 5 Views',
        save=True, path='./plots/p1e3.png')
