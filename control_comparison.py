#import mayavi
#from mayavi import mlab
import numpy as np
import pdb

REDSHIFTS = [9.0000, 4.7897, 3.2985, 2.4950, 1.9792, 1.6141, 1.3385,
             1.1212, 0.9438, 0.7955, 0.6688, 0.5588, 0.4620, 0.3758,
             0.2983, 0.2280, 0.1639, 0.1049, 0.0505, 0.0000]

"""
Note: this script is in it's own directory because it uses a different
      virtualenv to manage mayavi/vtk dependencies
"""

zx, zy = 15, 19
rsx, rsy = REDSHIFTS[zx], REDSHIFTS[zy]

mname = 'ShiftInv_corrected_nOrder_3K_ZG_{}-{}'.format(zx, zy)
dpath = './Model/' + mname + '/Cubes/X32_{}-{}_{}.npy'
T = 4


def get_mask(x, bound=0.1):
    xtmp = x[...,:3]
    #lower, upper = bound, 1-bound
    lower = bound
    upper = 1-bound
    mask1 = np.logical_and(xtmp[:,0] < upper, xtmp[:,0] > lower)
    mask2 = np.logical_and(xtmp[:,1] < upper, xtmp[:,1] > lower)
    mask3 = np.logical_and(xtmp[:,2] < upper, xtmp[:,2] > lower)
    mask = mask1 * mask2 * mask3
    mask_nz = np.nonzero(mask)[0]
    return mask_nz



'''
#x_preds  = np.load(dpath.format('prediction'))[:, j, ...]
#x_truths = np.load(dpath.format('true'))[:, j, ...]
#x_preds = np.concatenate( (x_truths[[0],...],x_preds), axis=0)

mask = np.ones(x_truths.shape[1], dtype='bool')
bound = .9
for i in range(T):
    mask = np.logical_and(mask, np.max(np.abs(x_truths[i+1] - x_truths[i]), axis=-1) < bound)
    mask = np.logical_and(mask, np.max(np.abs(x_preds[i+1] - x_preds[i]), axis=-1) < bound)

mask_nz = np.nonzero(mask)[0]
#mask_nz = np.arange(x_input.shape[0])
#mask_nz = np.random.choice(mask_nz, 1000)



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
                   scale_mode='none',
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
        pts = mlab.quiver3d(data[:,0], data[:,1], data[:,2], arrow[:,0], arrow[:,1], arrow[:,2], color=color, opacity=opacity, figure=figure, mode=mode, scale_factor=1.)
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







red   = (1,0,0)
green = (0,1,0)
blue  = (0,0,1)
arrow_mode = 'arrow'
opacity = .3
sfactor = .005

#displacement = np.mean((x_truth[:,mask_nz,:3] - x_input[:,mask_nz,:3]),axis=(1,2))
#greatest = np.argmax(np.abs(displacement))
#least = np.argmin(np.abs(displacement))
#j = greatest
#j = least
#print('displacement: {} at {}'.format(displacement[j], j))

#arrow_true  = (x_input[mask_nz,:3], x_truth[mask_nz,:3] - x_input[mask_nz,:3])
#arrow_input = (x_input[mask_nz,:3], x_input[mask_nz,3:])
#arrow_pred  = (x_input[mask_nz,:3], x_pred[ mask_nz,:3] - x_input[mask_nz,:3])
#volumize_arrow(*arrow_true,  figure=fig, color=red,   opacity=opacity, mode=arrow_mode)
#volumize_arrow(*arrow_input, figure=fig, color=green, opacity=opacity, mode=arrow_mode)
#volumize_arrow(*arrow_pred,  figure=fig, color=blue,  opacity=opacity, mode=arrow_mode)

###################3

#fig = mlab.figure(1, size=(400, 400), bgcolor=(1, 1, 1), fgcolor=(1, 0, 0))
# c3   = (1,0,0)
# c2 = (0,1,0)
# c1  = (0,0,1)

# arrow_pred1  = (x_preds[0,mask_nz,:3], x_preds[0,mask_nz,3:])
# arrow_pred2  = (x_preds[1,mask_nz,:3], x_preds[1,mask_nz,3:])
# arrow_pred3  = (x_preds[2,mask_nz,:3], x_preds[2,mask_nz,3:])
# volumize_arrow(*arrow_pred1,  figure=fig, color=c1,   opacity=opacity, mode=arrow_mode)
# volumize_arrow(*arrow_pred2, figure=fig, color=c2, opacity=opacity, mode=arrow_mode)
# volumize_arrow(*arrow_pred3,  figure=fig, color=c3,  opacity=opacity, mode=arrow_mode)


# fig2 = mlab.figure(2, size=(400, 400), bgcolor=(1, 1, 1), fgcolor=(1, 0, 0))
# arrow_truth1  = (x_truths[0,mask_nz,:3], x_truths[0,mask_nz,3:])
# arrow_truth2  = (x_truths[1,mask_nz,:3], x_truths[1,mask_nz,3:])
# arrow_truth3  = (x_truths[2,mask_nz,:3], x_truths[2,mask_nz,3:])
# volumize_arrow(*arrow_truth1,  figure=fig2, color=c1,   opacity=opacity, mode=arrow_mode)
# volumize_arrow(*arrow_truth2, figure=fig2, color=c2, opacity=opacity, mode=arrow_mode)
# volumize_arrow(*arrow_truth3,  figure=fig2, color=c3,  opacity=opacity, mode=arrow_mode)
# mlab.show()
#######################

#change the scale_factor for quiver3d to 1.
# fig = mlab.figure(1, size=(400, 400), bgcolor=(1, 1, 1), fgcolor=(1, 0, 0))
# scale_mode = 'none'
# scale_factor = 1
# arrow_mode = 'arrow'
# c2 = (1,0,0)
# c1  = (0,1,0)

# arrow_pred1  = (x_preds[0,mask_nz,:3], x_preds[1,mask_nz,:3] - x_preds[0,mask_nz,:3])
# arrow_pred2  = (x_preds[1,mask_nz,:3], x_preds[2,mask_nz,:3] - x_preds[1,mask_nz,:3])
# volumize_arrow(*arrow_pred1,  figure=fig, color=c1, opacity=opacity, mode=arrow_mode)
# volumize_arrow(*arrow_pred2, figure=fig, color=c1, opacity=opacity, mode=arrow_mode)

# arrow_truth1  = (x_truths[0,mask_nz,:3], x_truths[1,mask_nz,:3] - x_truths[0,mask_nz,:3])
# arrow_truth2  = (x_truths[1,mask_nz,:3], x_truths[2,mask_nz,:3] - x_truths[1,mask_nz,:3])
# volumize_arrow(*arrow_truth1,  figure=fig, color=c2,   opacity=opacity, mode=arrow_mode)
# volumize_arrow(*arrow_truth2, figure=fig, color=c2, opacity=opacity, mode=arrow_mode)
# mlab.show()
###################### multi-step

#change the scale_factor for quiver3d to 1.
# fig1 = mlab.figure(1, size=(400, 400), bgcolor=(1, 1, 1), fgcolor=(1, 0, 0))
# fig2 = mlab.figure(2, size=(400, 400), bgcolor=(1, 1, 1), fgcolor=(1, 0, 0))
# scale_mode = 'none'
# scale_factor = 1
# arrow_mode = 'arrow'
# opacity = .3

# arrow_preds = []
# arrow_truths = []
# arrow_diff = []
# for t in range(T):
#     arrow_preds.append((x_preds[t,mask_nz,:3], x_preds[t+1,mask_nz,:3] - x_preds[t,mask_nz,:3]))
#     c1 = (t/4.,0,0)
#     volumize_arrow(*arrow_preds[-1],  figure=fig1, color=c1, opacity=opacity, mode=arrow_mode)

#     c2 = (0, t/4.,0)
#     arrow_truths.append((x_truths[t,mask_nz,:3], x_truths[t+1,mask_nz,:3] - x_truths[t,mask_nz,:3]))
#     volumize_arrow(*arrow_truths[-1],  figure=fig2, color=c2,   opacity=opacity, mode=arrow_mode)

# # error in final step
# #arrow_diff = (x_preds[-1, mask_nz, :3], x_truths[-1, mask_nz, :3] - x_preds[-1, mask_nz, :3])
# #volumize_arrow(*arrow_diff,  figure=fig, color=c2,   opacity=opacity, mode=arrow_mode)

# mlab.show()
'''


#####################
import pylab as plt
plt.style.use('ggplot')
plt.ion()


x_truth_cube = np.load(dpath.format(zx, zy, 'true'))
x_input_full = x_truth_cube[0]
x_truth_full = x_truth_cube[1]
x_pred_full  = np.load(dpath.format(zx, zy, 'prediction'))

def angle(v1, v2):
# v1 is your firsr vector
# v2 is your second vector
    angle = np.degrees(np.arccos(np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))))
    return angle


j = 39 # sample
x_input = x_input_full[j]
x_truth = x_truth_full[j]
x_pred  =  x_pred_full[j]

mask_nz = get_mask(x_input)
#i = 1
#v_truth = x_truths[i, mask_nz, :3] - x_truths[0, mask_nz, :3]
#v_pred = x_preds[i, mask_nz, :3] - x_preds[0, mask_nz, :3]
#v_vel = x_truths[0, mask_nz, 3:]
loc_truth = np.copy(x_truth[mask_nz,  :3])
loc_pred  = np.copy(x_pred[ mask_nz,  :3])
loc_input = np.copy(x_input[mask_nz,  :3])
vel_input = np.copy(x_input[mask_nz, 3:])

alpha = .5
label1 = 'using velocity'
label2 = 'deep model'
c1 = 'r'
c2 = 'b'
plt.clf()

'''
bins = np.linspace(0,90,200)
plt.hist(angle( v_vel, v_truth), bins= bins, label='using velocity', alpha=alpha)
plt.hist(angle(v_pred, v_truth), bins= bins, label='deep model',     alpha=alpha)
plt.xlabel('angle (degrees)')
#plt.title('error after {0:2d} steps'.format(i))
plt.title('Error, single-step: {:.4f} --> {:.4f}'.format(rsx, rsy))
plt.legend()
plt.show()
'''
#plt.figure()
#v_vel0 = x_truths[0, mask_nz, 3:]
#t = np.linalg.lstsq(v_vel0.ravel()[:,None], v_truth.ravel())[0]
#t = np.linalg.lstsq(v_vel.ravel()[:,None], v_truth.ravel())[0]
#t = np.linalg.lstsq(vel_input.ravel()[:,None], loc_truth.ravel())[0]
true_diff = loc_truth - loc_input
timestep = np.linalg.lstsq(vel_input.ravel()[:,None], true_diff.ravel())[0]
#v_vel = v_vel0 * t
#vel_input_t = vel_input * t
displacement = vel_input * timestep
bins = np.linspace(-.01,.05,200)
#bins = None
#l2_dist_vel  = np.linalg.norm(v_truth - v_vel_t, axis=-1)
#l2_dist_pred = np.linalg.norm(v_truth - v_pred, axis=-1)
#l2_dist_vel  = np.linalg.norm(loc_truth - displacement, axis=-1)
vel_model_loc = loc_input + displacement
l2_dist_vel  = np.linalg.norm(loc_truth - vel_model_loc, axis=-1)
l2_dist_pred = np.linalg.norm(loc_truth - loc_pred,    axis=-1)

plt.hist(l2_dist_vel,  bins= bins, label=label1, color=c1,alpha=alpha)
plt.hist(l2_dist_pred, bins= bins, label=label2, color=c2,alpha=alpha)

#plt.title('error after {0:2d} steps'.format(i))
plt.title('Error, single-step {:>2}, {:>2}: {:.4f} --> {:.4f}'.format(zx, zy, rsx, rsy))
plt.xlabel('distance (L2)')
plt.legend()
plt.tight_layout()
#plt.show()



#######################
#volumize_ptc(x_truth[j,mask_nz,:3], show=False,figure=fig, opacity=.5, color=red,  mode='sphere', scale_factor=sfactor)
#volumize_ptc(x_input[j,mask_nz,:3], show=False,figure=fig, opacity=.9, color=green,mode='sphere', scale_factor=sfactor)
#volumize_ptc( x_pred[j,mask_nz,:3], show=True, figure=fig, opacity=.5, color=blue, mode='sphere', scale_factor=sfactor)
#mlab.savefig('test3.png', size=(3000,3000), figure=fig)


