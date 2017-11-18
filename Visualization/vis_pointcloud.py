import mayavi
from mayavi import mlab
import numpy as np
"""
Note: this script is in it's own directory because it uses a different
      virtualenv to manage mayavi/vtk dependencies 
"""

data_dir = '../Evaluation/'
x_truth     = np.load(data_dir + 'GraphModel_beta_truth.npy')
x_hat_graph = np.load(data_dir + 'GraphModel_beta_pred.npy')
x_hat_set   = np.load(data_dir + 'SetModel_pred.npy')

x_truth_32     = np.load(data_dir + 'GraphModel_beta_32_truth.npy')
x_hat_32_graph = np.load(data_dir + 'GraphModel_beta_32_pred.npy')


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
        pts = mlab.points3d(data[:,0], data[:,1], data[:,2], 
                            mode=mode, color=color, opacity=opacity, figure=figure, scale_factor=scale_factor)
        #mlab.pipeline.volume(mlab.pipeline.gaussian_splatter(pts, figure=figure))
        #pts = mlab.points3d(data[:,0], data[:,1], data[:,2], mode=mode, color=color, opacity=opacity, figure=figure, scale_mode='none')
        #pts = mlab.points3d(data[:,0], data[:,1], data[:,2], color=color, opacity=opacity, figure=figure, scale_factor=scale_factor)
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



#j,k = 0,90
test0 = x_truth[j,:,:3]
testk = x_truth[k,:,:3]
test_32k = x_truth_32[k,:,:3]
hat32k_graph = x_hat_32_graph[k,:,:3]
hat0_graph   = x_hat_graph[j,:,:3]
hatk_graph = x_hat_graph[k,:,:3]
hat0_set   = x_hat_set[j,:,:3]
hatk_set = x_hat_set[k,:,:3]

#test1 = test[0][0,:,:3]
#test2 = test[1][0,:,:3]#l1 = np.array([0 for i in range(25)]+[1 for i in range(25)])
#test3 = out_test[1,:,:3]
fig = mlab.figure(1, size=(400, 400), bgcolor=(1, 1, 1), fgcolor=(1, 0, 0))

sfactor = .01
volumize_ptc(test_32k,figure=fig, opacity=.9, show=False, color=(0,0,1), mode='point', scale_factor=sfactor)
#volumize_ptc(test0,show = True, figure=fig, opacity=.9, color=(0,0,1), mode='point')
volumize_ptc(hat32k_graph,show = True, figure=fig, opacity=.9, color=(1,1,0), mode='point', scale_factor=sfactor)
