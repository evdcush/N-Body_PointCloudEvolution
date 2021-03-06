import mayavi
from mayavi import mlab
import numpy as np
import code
"""
Note: this script is in it's own directory because it uses a different
      virtualenv to manage mayavi/vtk dependencies
"""

def volumize_ptc(data_in, opacity=.5, labels=None, color=(1,0,0),frame=True,
                 row=0, col=0, show=True, figure=None, proj=(False,True,True),
                 shadow=(False,True,True), mode='point', scale_factor=.015,
                 filename=None,):
    if figure is None:
        figure = mlab.figure(1, size=(400, 400), bgcolor=(0, 0, 0), fgcolor=(1, 0, 0))

    data = data_in.copy()
    data -= np.min(data, keepdims=True)
    data /= np.max(data, keepdims=True)
    data[:,0] += np.float(col)
    data[:,1] += np.float(row)
    xproj, yproj, zproj = proj
    xshadow, yshadow, zshadow = shadow
    if labels is None:
        pts = mlab.points3d(data[:,0], data[:,1], data[:,2], mode=mode, color=color, opacity=opacity, figure=figure, scale_factor=scale_factor)
    else:
        for l in np.unique(labels):
            color = tuple(list(np.random.rand(3)))
            ind = (labels == l).nonzero()[0]
            mlab.pipeline.volume(mlab.points3d(data[ind,0], data[ind,1], data[ind,2],
                                               mode=mode, color=color, opacity=opacity))
    if frame:
        r_points = np.array([0,1,1,0,0,1,1,1,1,1,0,0,0,1,1,0,0,0]) + row + .0
        c_points = np.array([0,0,1,1,1,1,0,0,1,1,1,0,0,0,0,0,0,1]) + col +.0
        d_points = np.array([0,0,0,0,1,1,1,0,0,1,1,1,0,0,1,1,0,0]) +.0
        mlab.plot3d(c_points, r_points, d_points, representation='surface',
                    tube_radius=.003, line_width=1, figure=figure, opacity=.7, color=(1,1,1))

    #mlab.pipeline.volume(mlab.pipeline.gaussian_splatter(pts, figure=figure))
    mlab.view(azimuth=20, elevation=70, distance=3, focalpoint=None, roll=None, reset_roll=True, figure=figure)
    if filename is not None:
        mlab.savefig(filename, size=(500,500), figure=figure, magnification='auto')
        if not show: mlab.clf()
    if show: mlab.show()



def volumize_arrow(datain,# n x 3
                   arrow, #nx3
                 opacity=.5, labels=None,
                 color=(1,0,0),
                 frame=True, row=0, col=0,
                 show=True,
                 figure=None,
                 proj=(False,True,True),
                 shadow=(False,True,True),
                 mode='point',
                 scale_factor=.001,
                 filename=None,
                 normalize=False,
):
    if figure is None:
        figure = mlab.figure(1, size=(400, 400), bgcolor=(0, 0, 0), fgcolor=(1, 0, 0))

    data = datain.copy()
    if normalize:
        data -= np.min(data, keepdims=True)
        data /= np.max(data, keepdims=True)
    data[:,0] += np.float(col)
    data[:,1] += np.float(row)
    #xproj,yproj,zproj = proj
    #xshadow,yshadow,zshadow=shadow
    if labels is None:
        #pts = mlab.points3d(data[:,0], data[:,1], data[:,2], mode=mode, color=color, opacity=opacity, figure=figure, scale_factor=scale_factor)
        #mlab.pipeline.volume(mlab.pipeline.gaussian_splatter(pts, figure=figure))
        pts = mlab.quiver3d(data[:,0], data[:,1], data[:,2], arrow[:,0], arrow[:,1], arrow[:,2], color=color, opacity=opacity, figure=figure, mode=mode)
    else:
        for l in np.unique(labels):
            color = tuple(list(np.random.rand(3)))
            ind = (labels == l).nonzero()[0]
            mlab.pipeline.volume(mlab.points3d(data[ind,0], data[ind,1], data[ind,2], mode=mode, color=color, opacity=opacity))
    if frame:
        r_points = np.array([0,1,1,0,0,1,1,1,1,1,0,0,0,1,1,0,0,0]) + row + .0
        c_points = np.array([0,0,1,1,1,1,0,0,1,1,1,0,0,0,0,0,0,1]) + col +.0
        d_points = np.array([0,0,0,0,1,1,1,0,0,1,1,1,0,0,1,1,0,0]) +.0
        mlab.plot3d(c_points, r_points, d_points, representation='surface',tube_radius=.003, line_width=1, figure=figure, opacity=.7, color=(1,1,1))


    #mlab.pipeline.volume(mlab.pipeline.gaussian_splatter(pts, figure=figure))
    mlab.view(azimuth=20, elevation=70, distance=3, focalpoint=None, roll=None, reset_roll=True, figure=figure)
    if filename is not None:
        mlab.savefig(filename, size=(500,500), figure=figure, magnification='auto')
        if not show:
            mlab.clf()
    #if show: mlab.show()


j = 39 # sample

dpath = '../new_multi9k/X32_3-19_{}.npy'

#x_input = np.load('../X05.npy')
x_truth = np.load(dpath.format('true'))
x_input = x_truth[-2, j]
x_truth = x_truth[-1, j]
x_pred  = np.load(dpath.format('prediction'))[-1, j]


xtmp = x_input[:,:3]
bound = 0.1
lower, upper = bound, 1-bound
mask1 = np.logical_and(xtmp[:,0] < upper, xtmp[:,0] > lower)
mask2 = np.logical_and(xtmp[:,1] < upper, xtmp[:,1] > lower)
mask3 = np.logical_and(xtmp[:,2] < upper, xtmp[:,2] > lower)
mask = mask1 * mask2 * mask3
mask_nz = np.nonzero(mask)[0]

fig = mlab.figure(1, size=(400, 400), bgcolor=(1, 1, 1), fgcolor=(1, 0, 0))
red   = (1,0,0)
green = (0,1,0)
blue  = (0,0,1)
arrow_mode = 'arrow'
opacity = .8
sfactor = .005

#displacement = np.mean((x_truth[:,mask_nz,:3] - x_input[:,mask_nz,:3]),axis=(1,2))
#greatest = np.argmax(np.abs(displacement))
#least = np.argmin(np.abs(displacement))
#j = greatest
#j = least
#print('displacement: {} at {}'.format(displacement[j], j))

arrow_true  = (x_input[mask_nz,:3], x_truth[mask_nz,:3] - x_input[mask_nz,:3])
arrow_input = (x_input[mask_nz,:3], x_input[mask_nz,3:])
arrow_pred  = (x_input[mask_nz,:3], x_pred[ mask_nz,:3] - x_input[mask_nz,:3])
volumize_arrow(*arrow_true,  figure=fig, color=red,   opacity=opacity, mode=arrow_mode)
volumize_arrow(*arrow_input, figure=fig, color=green, opacity=opacity, mode=arrow_mode)
volumize_arrow(*arrow_pred,  figure=fig, color=blue,  opacity=opacity, mode=arrow_mode)
#volumize_ptc(x_truth[j,mask_nz,:3], show=False,figure=fig, opacity=.5, color=red,  mode='sphere', scale_factor=sfactor)
#volumize_ptc(x_input[j,mask_nz,:3], show=False,figure=fig, opacity=.9, color=green,mode='sphere', scale_factor=sfactor)
#volumize_ptc( x_pred[j,mask_nz,:3], show=True, figure=fig, opacity=.5, color=blue, mode='sphere', scale_factor=sfactor)
#mlab.savefig('test3.png', size=(3000,3000), figure=fig)

mlab.show()
