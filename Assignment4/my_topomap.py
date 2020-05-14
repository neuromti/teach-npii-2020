# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 16:50:04 2020

@author: Messung
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
from os import path
from pylab import figure, show
import pickle
with open('channel_info.p', 'rb') as pf:
    channel_info = pickle.load(pf)
    
pos = channel_info['positions']
labels = channel_info['labels']

def multichannel_plot(t, datarray):    
    fig = figure()
    
    yprops = dict(rotation=0,
                  horizontalalignment='right',
                  verticalalignment='center')
    
    axprops = dict(yticks=[])
    nchans = np.shape(datarray)[1]
    ht = .9/nchans
    ypos = [0.7, 0.5, 0.3, 0.1]
    ypos = np.linspace(1-ht, .1, nchans)
    
    for ch in range(nchans):
        ax1 =fig.add_axes([0.1, ypos[ch], 0.8, ht], **axprops)
        ax1.plot(t, datarray[:,ch])
        ax1.set_ylabel('comp'+str(ch), **yprops)
        
        axprops['sharex'] = ax1
        axprops['sharey'] = ax1
    ax1.set_xlabel('time [s]')
    show()

def find_muscular_components(comps_epoch):
    dmy = np.zeros(np.shape(comps_epoch)[1])
    timewin = np.arange(488,530)
    for c in range(len(dmy)):
        excessratio = np.max(np.abs(comps_epoch[timewin,c]))/np.mean(np.abs(comps_epoch[:,c]))
        if excessratio >= 8:
            dmy[c] = 1
    return np.where(dmy)[0]


#function copied from mne.viz.topomap
def _check_outlines(pos, outlines, head_pos=None):
   # """Check or create outlines for topoplot."""
    pos = np.array(pos, float)[:, :2]  # ensure we have a copy
    head_pos = dict() if head_pos is None else head_pos
    if not isinstance(head_pos, dict):
        raise TypeError('head_pos must be dict or None')
    head_pos = copy.deepcopy(head_pos)
    for key in head_pos.keys():
        if key not in ('center', 'scale'):
            raise KeyError('head_pos must only contain "center" and '
                           '"scale"')
        head_pos[key] = np.array(head_pos[key], float)
        if head_pos[key].shape != (2,):
            raise ValueError('head_pos["%s"] must have shape (2,), not '
                             '%s' % (key, head_pos[key].shape))

    if isinstance(outlines, np.ndarray) or outlines in ('head', 'skirt', None):
        radius = 0.5
        ll = np.linspace(0, 2 * np.pi, 101)
        head_x = np.cos(ll) * radius
        head_y = np.sin(ll) * radius
        nose_x = np.array([0.18, 0, -0.18]) * radius
        nose_y = np.array([radius - .004, radius * 1.15, radius - .004])
        ear_x = np.array([.497, .510, .518, .5299, .5419, .54, .547,
                          .532, .510, .489])
        ear_y = np.array([.0555, .0775, .0783, .0746, .0555, -.0055, -.0932,
                          -.1313, -.1384, -.1199])

        # shift and scale the electrode positions
        if 'center' not in head_pos:
            head_pos['center'] = 0.5 * (pos.max(axis=0) + pos.min(axis=0))
        pos -= head_pos['center']

        if outlines is not None:
            # Define the outline of the head, ears and nose
            outlines_dict = dict(head=(head_x, head_y), nose=(nose_x, nose_y),
                                 ear_left=(ear_x, ear_y),
                                 ear_right=(-ear_x, ear_y))
        else:
            outlines_dict = dict()

        if isinstance(outlines, str) and outlines == 'skirt':
            if 'scale' not in head_pos:
                # By default, fit electrodes inside the head circle
                head_pos['scale'] = 1.0 / (pos.max(axis=0) - pos.min(axis=0))
            pos *= head_pos['scale']

            # Make the figure encompass slightly more than all points
            mask_scale = 1.25 * (pos.max(axis=0) - pos.min(axis=0))

            outlines_dict['autoshrink'] = False
            outlines_dict['mask_pos'] = (mask_scale[0] * head_x,
                                         mask_scale[1] * head_y)
            outlines_dict['clip_radius'] = (mask_scale / 2.)
        else:
            if 'scale' not in head_pos:
                # The default is to make the points occupy a slightly smaller
                # proportion (0.85) of the total width and height
                # this number was empirically determined (seems to work well)
                head_pos['scale'] = 0.85 / (pos.max(axis=0) - pos.min(axis=0))
            pos *= head_pos['scale']
            outlines_dict['mask_pos'] = head_x, head_y
            if isinstance(outlines, np.ndarray):
                outlines_dict['autoshrink'] = False
                outlines_dict['clip_radius'] = outlines
                x_scale = np.max(outlines_dict['head'][0]) / outlines[0]
                y_scale = np.max(outlines_dict['head'][1]) / outlines[1]
                for key in ['head', 'nose', 'ear_left', 'ear_right']:
                    value = outlines_dict[key]
                    value = (value[0] / x_scale, value[1] / y_scale)
                    outlines_dict[key] = value
            else:
                outlines_dict['autoshrink'] = True
                outlines_dict['clip_radius'] = (0.5, 0.5)

        outlines = outlines_dict

    elif isinstance(outlines, dict):
        if 'mask_pos' not in outlines:
            raise ValueError('You must specify the coordinates of the image '
                             'mask.')
    else:
        raise ValueError('Invalid value for `outlines`.')

    return pos, outlines

#function copied from mne.viz.topomap
def _draw_outlines(ax, outlines):
    #"""Draw the outlines for a topomap."""
    outlines_ = {k: v for k, v in outlines.items()
                 if k not in ['patch', 'autoshrink']}
    for key, (x_coord, y_coord) in outlines_.items():
        if 'mask' in key:
            continue
        ax.plot(x_coord, y_coord, color='k', linewidth=1, clip_on=False)
    return outlines_



def scalefunc(label):
    scaledict = {'10 kOhm': 10, '20 kOhm': 20, '50 kOhm': 50}
    cmax = scaledict[label]
    print(cmax)

    return cmax


def imp2col(Imp, cmax):
    if cmax == 0:
        cmax = 50
    cmap = plt.cm.get_cmap('RdYlGn_r')
    imp = min(Imp, cmax)
    col = cmap(float(imp/cmax))
    return col    


def plot_topomap(x, pos = pos, labels = labels, cmax = 50):
    x = x[:64]


    if not plt.fignum_exists(1):
        plt.figure(1)
    else:
        plt.figure(1).clear()
        
    plt.axis('off')
    pos, outlines = _check_outlines(pos, outlines='head', head_pos=None)
    ax = plt.gca()
    _draw_outlines(ax, outlines)
#    rax = plt.axes([0.05, 0.7, 0.15, 0.15])
#    radio = RadioButtons(rax, ('10 kOhm', '20 kOhm', '50 kOhm'))
#    rax.set_title('select scale')
#    
#    

    norm = mpl.colors.Normalize(vmin = 0, vmax = 1)
    sm = plt.cm.ScalarMappable(norm = norm,cmap = 'RdYlGn_r')
    sm.set_array([])
    cb = plt.colorbar(sm, ax = ax)
    cb.set_ticks((0,0.5,1))
    cb.set_ticklabels((0,int(cmax/2),int(cmax)))
    for c in range(len(pos)):
        ax.text(pos[c,0],pos[c,1],'{}\n{}'.format(labels[c], min(int(x[c]), int(cmax))), size = 12, bbox=dict(facecolor=imp2col(x[c], cmax), alpha=1))
        

    

