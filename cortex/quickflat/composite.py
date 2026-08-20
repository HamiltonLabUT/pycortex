import copy
from typing import TYPE_CHECKING, Any, Optional, Union, Sequence

from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.typing import ColorType
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
import numpy as np
import numpy.typing as npt

from .utils import _get_height, _get_extents, _convert_svg_kwargs, _get_images, _parse_defaults
from .utils import make_flatmap_image, _make_hatch_image, _get_fig_and_ax, get_flatmask, get_flatcache
from .. import dataset
from ..dataset.views import HasSubject, as_renderable
from ..dataset.views import VolumetricView
from ..database import db
from ..options import config

if TYPE_CHECKING:
    from ..electrodes import ElectrodeSet


""" --- Individual compositing functions --- """


def add_curvature(fig: Axes, dataview: HasSubject, extents: Optional[tuple[float, float, float, float]]=None, height: Optional[int]=None, threshold: Optional[bool]=True, contrast: Optional[float]=None,
                  brightness: Optional[float]=None, smooth: Optional[float]=None, cmap: str='gray', recache: bool=False, curvature_lims: float=0.5,
                  legacy_mode: bool=False) -> AxesImage:
    """Add curvature layer to figure

    Parameters
    ----------
    fig : figure or ax
        figure into which to plot image of curvature
    dataview : cortex.Dataview object
        dataview containing data to be plotted, subject (surface identifier), and transform.
    extents : array-like TODO: fix
        4 values for [Left, Right, Top, Bottom] extents of image plotted. None defaults to 
        extents of images already present in figure.
    height : scalar
        Height of image. None defaults to height of images already present in figure. 
        TODO: what units?
    threshold : boolean
        Whether to apply a threshold to the curvature values to create a binary curvature image
        (one shade for positive curvature, one shade for negative). `None` defaults to value 
        specified in the config file
    contrast : float, [0-1] or None
        Contrast of curvature image. 1 is maximal contrast (given brightness). If brightness is 0.5
        and contrast is 1, and cmap is 'gray', curvature will be black and white. None defaults
        to value in config file.
    brightness : float, [0-1] or None
        How bright to make average value of curvature (0=black, 1=white in gray cmap). None
        defaults to the value in config file.
    curvature_lims : float
        Limits for real curvature values (actual values for cortical curvature are normalized
        within [-`curvature_lims`, +`curvature_lims`] before scaling by `contrast` and shifting
        by `brightness`).
    smooth : scalar or None
        Width of smoothing to apply to surface curvature. None defaults to no smoothing, or
        whatever the default value for curvature is that is stored in
        <filestore>/<subject>/surface-info/curvature.npz (for some subjects initiated in old
        versions of pycortex, this may be smoothed too!)
    cmap : string
        name for colormap of curvature
    recache : boolean
        Whether or not to recache intermediate files. Takes longer to plot this way, potentially
        resolves some errors.

    Returns
    -------
    img : matplotlib.image.AxesImage
        matplotlib axes image object for plotted data

    """
    from matplotlib.colors import Normalize
    if height is None:
        height = _get_height(fig)
    # Get curvature map as image
    default_smoothing = config.get('curvature', 'smooth')
    if default_smoothing.lower()=='none':
        default_smoothing = None
    else:
        default_smoothing = np.float64(default_smoothing)
    if smooth is None:
        # (Might still be None!)
        smooth = default_smoothing
    if smooth is None:
        # If no value for 'smooth' is given in kwargs, db.get_surfinfo returns
        # the default curvature value, whatever that may be. This is the behavior
        # that we want a None in the code to invoke. This is silly and complicated
        # due to backward compatibility issues with some old subjects.
        curv_vertices = db.get_surfinfo(dataview.subject)
    else:
        curv_vertices = db.get_surfinfo(dataview.subject, smooth=smooth)
    curv, _ = make_flatmap_image(curv_vertices, recache=recache, height=height)
    # First, limit to sensible range for flatmap curvature
    norm = Normalize(vmin=-0.5, vmax=0.5)
    curv_im = norm(curv)
    # Option to use thresholded curvature
    default_threshold = config.get('curvature','threshold').lower() in ('true', 't', '1', 'y', 'yes')
    use_threshold_curvature = default_threshold if threshold is None else threshold
    if legacy_mode and use_threshold_curvature:
        curvT = (curv>0).astype(np.float32)
        curvT[np.isnan(curv)] = np.nan
        curv = curvT
    if isinstance(curvature_lims, (list, tuple)):
        vmin, vmax = curvature_lims
    else:
        vmin, vmax = -curvature_lims, curvature_lims
    norm = Normalize(vmin=vmin, vmax=vmax)
    curv_im = norm(curv)
    if not legacy_mode:
        if use_threshold_curvature:
            # Assumes symmetrical curvature_lims
            curv_im = (np.nan_to_num(curv_im) > 0.5).astype(float)
            curv_im[np.isnan(curv)] = np.nan
        # Get defaults for brightness, contrast
        if brightness is None:
            brightness = float(config.get('curvature', 'brightness'))
        if contrast is None:
            contrast = float(config.get('curvature', 'contrast'))
        # Scale and shift curvature image
        curv_im = (curv_im - 0.5) * contrast + brightness
    if extents is None:
        extents = _get_extents(fig)
    _, ax = _get_fig_and_ax(fig)
    cvimg = ax.imshow(curv_im,
                      aspect='equal',
                      extent=extents,
                      cmap=cmap,
                      vmin=0,
                      vmax=1,
                      label='curvature',
                      zorder=0)
    return cvimg

def add_data(fig: Figure, braindata: Union[dataset.Dataview, tuple], height: int=1024, thick: int=32, depth: float=0.5, pixelwise: bool=True,
             sampler: str='nearest', recache: bool=False, nanmean: bool=False) -> tuple[AxesImage, npt.NDArray]:
    """Add data to quickflat plot

    Parameters
    ----------
    fig : figure or ax
        Figure into which to plot image of curvature
    braindata : cortex.Dataview, or a tuple that `normalize` accepts
        Object containing containing data to be plotted, subject (surface identifier),
        and transform.
    height : scalar
        Height of image. None defaults to height of images already present in figure.
    recache : boolean
        Whether or not to recache intermediate files. Takes longer to plot this way, potentially
        resolves some errors. Useful if you've made changes to the alignment
    pixelwise : bool
        Use pixel-wise mapping
    thick : int
        Number of layers through the cortical sheet to sample. Only applies for pixelwise = True
    sampler : str
        Name of sampling function used to sample underlying volume data. Options include
        'trilinear','nearest','lanczos'; see functions in cortex.mapper.samplers.py for all options
    nanmean : bool, optional (default = False)
        If True, NaNs in the data will be ignored when averaging across layers.

    Returns
    -------
    img : matplotlib.image.AxesImage
        matplotlib axes image object for plotted data
    extents : list
        Extents of image [left, right, top, bottom] in figure coordinates
    """
    normalized = dataset.normalize(braindata)
    if not isinstance(normalized, dataset.Dataview):
        # Unclear what this means. Clarify error in terms of pycortex classes
        # (please provide a [cortex.dataset.Dataview or whatever] instance)
        raise TypeError('Please provide a Dataview, not a Dataset')
    dataview = normalized
    # as_renderable is the one boundary between "some view" and "a view the
    # flatmap renderer can draw"; it is applied here rather than to `dataview`
    # so the Dataview members below (get_cmapdict) stay available.
    im, extents = make_flatmap_image(as_renderable(dataview), recache=recache, pixelwise=pixelwise, sampler=sampler,
                                     height=height, thick=thick, depth=depth, nanmean=nanmean)
    # Check whether dataview has a cmap instance
    cmapdict = dataview.get_cmapdict()
    # Plot
    _, ax = _get_fig_and_ax(fig)
    img = ax.imshow(im,
                    aspect='equal',
                    extent=extents,
                    label='data',
                    zorder=1,
                    interpolation="nearest",
                    **cmapdict)
    return img, extents

def add_rois(fig: Figure, dataview: HasSubject, extents: Optional[tuple[float, float, float, float]]=None, height: Optional[int]=None, with_labels: bool=True, roi_list: Optional[Sequence[str]]=None, overlay_file: Optional[str]=None, **kwargs) -> AxesImage:
    """Add ROIs layer to a figure

    NOTE: zorder for rois is 3

    Parameters
    ----------
    fig : figure or ax
        figure into which to plot image of curvature
    dataview : cortex.Dataview object
        dataview containing data to be plotted, subject (surface identifier), and transform.
    extents : array-like
        4 values for [Left, Right, Top, Bottom] extents of image plotted. None defaults to 
        extents of images already present in figure.
    height : scalar 
        Height of image. None defaults to height of images already present in figure. 
    with_labels : bool
        Whether to display text labels on ROIs
    roi_list : list of str, optional
        List of ROIs to include

    kwargs : 

    Returns
    -------
    img : matplotlib.image.AxesImage
        matplotlib axes image object for plotted data
    """
    if extents is None:
        extents = _get_extents(fig)
    if height is None:
        height = _get_height(fig)        
    svgobject = db.get_overlay(dataview.subject, overlay_file=overlay_file)
    svg_kws = _convert_svg_kwargs(kwargs)
    layer_kws = _parse_defaults('rois_paths')
    layer_kws.update(svg_kws)
    im = svgobject.get_texture('rois', height, labels=with_labels, shape_list=roi_list, **layer_kws)
    _, ax = _get_fig_and_ax(fig)
    img = ax.imshow(im,
                    aspect='equal',
                    interpolation='bicubic',
                    extent=extents,
                    label='rois',
                    zorder=1000)
    return img


def add_sulci(fig: Figure, dataview: HasSubject, extents: Optional[tuple[float, float, float, float]]=None, height: Optional[int]=None, with_labels: bool=True, sulci_list: Optional[Sequence[str]]=None, overlay_file: Optional[str]=None, **kwargs) -> AxesImage:
    """Add sulci layer to figure

    Parameters
    ----------
    fig : figure or ax
        figure into which to plot image of curvature
    dataview : cortex.Dataview object
        dataview containing data to be plotted, subject (surface identifier), and transform.
    extents : array-like
        4 values for [Left, Right, Top, Bottom] extents of image plotted. None defaults to 
        extents of images already present in figure.
    height : scalar
        Height of image. None defaults to height of images already present in figure. 
    with_labels : bool
        Whether to display text labels for sulci
    sulci_list : list[str]
        List of sulci to include

    Other Parameters
    ----------------
    kwargs : keyword arguments
        Keywords args govern line appearance in final plot. Allowable kwargs are : linewidth,
        linecolor

    Returns
    -------
    img : matplotlib.image.AxesImage
        matplotlib axes image object for plotted data
    """
    svgobject = db.get_overlay(dataview.subject, overlay_file=overlay_file)
    svg_kws = _convert_svg_kwargs(kwargs)
    layer_kws = _parse_defaults('sulci_paths')
    layer_kws.update(svg_kws)
    sulc = svgobject.get_texture('sulci', height, labels=with_labels, shape_list=sulci_list, **layer_kws)
    if extents is None:
        extents = _get_extents(fig)
    _, ax = _get_fig_and_ax(fig)
    img = ax.imshow(sulc,
                    aspect='equal',
                    interpolation='bicubic',
                    extent=extents,
                    label='sulci',
                    zorder=5)
    return img


def add_hatch(fig: Axes, hatch_data: dataset.Dataview, extents: Optional[tuple[float, float, float, float]]=None, height: Optional[int]=None, hatch_space: int=4,
              hatch_color: tuple[int, int, int]=(0, 0, 0), sampler: str='nearest', recache: bool=False) -> AxesImage:
    """Add hatching to figure at locations specified in hatch_data

    Parameters
    ----------
    fig : matplotlib figure
        Figure into which to plot the hatches. Should have pycortex flatmap image in it already.
    hatch_data : cortex.Dataview
        Rendered through the flatmap, so it must be a renderable kind.
        cortex.Volume object created from data scaled from 0-1; locations with values of 1 will
        have hatching overlaid on them in the resulting image.
    extents : array-like
        4 values for [Left, Right, Top, Bottom] extents of image plotted. If None, defaults to 
        extents of images already present in figure.
    height : scalar 
        Height of image. if None, defaults to height of images already present in figure. 
    hatch_space : scalar 
        Spacing between hatch lines, in pixels
    hatch_color : 3-tuple
        (R, G, B) tuple for color of hatching. Values for R,G,B should be 0-1
    sampler : str
        Name of sampling function used to sample underlying volume data. Options include 
        'trilinear','nearest','lanczos'; see functions in cortex.mapper.samplers.py for all options
    recache : boolean
        Whether or not to recache intermediate files. Takes longer to plot this way, potentially
        resolves some errors. 

    Returns
    -------
    img : matplotlib.image.AxesImage
        matplotlib axes image object for plotted hatch image

    Notes
    -----
    Possibly to add: add hatch_width, hatch_offset arguments.
    """
    if extents is None:
        extents = _get_extents(fig)
    if height is None:
        height = _get_height(fig)
    hatchim = _make_hatch_image(as_renderable(hatch_data), height, sampler, recache=recache, 
                                hatch_space=hatch_space)
    hatchim[:,:,0] = hatch_color[0]
    hatchim[:,:,1] = hatch_color[1]
    hatchim[:,:,2] = hatch_color[2]

    _, ax = _get_fig_and_ax(fig)
    img = ax.imshow(hatchim, 
                    aspect="equal", 
                    interpolation="bicubic", 
                    extent=extents, 
                    label='hatch',
                    zorder=2)
    return img


def add_colorbar(fig: Figure, cimg: AxesImage, colorbar_ticks: Optional[npt.ArrayLike]=None, colorbar_location: tuple[float, float, float, float]=(0.4, 0.07, 0.2, 0.04),
                 orientation: str='horizontal') -> Axes:
    """Add a colorbar to a flatmap plot

    Parameters
    ----------
    fig : matplotlib Figure object
        Figure into which to insert colormap
    cimg : matplotlib.image.AxesImage object
        Image for which to create colorbar. For reference, matplotlib.image.AxesImage 
        is the output of imshow()
    colorbar_ticks : array-like
        values for colorbar ticks
    colorbar_location : array-like
        Four-long list, tuple, or array that specifies location for colorbar axes 
        [left, top, width, height] (?)
    orientation : string
        'vertical' or 'horizontal'
    """
    fig, _ = _get_fig_and_ax(fig)
    cbar = fig.add_axes(colorbar_location)
    fig.colorbar(cimg, cax=cbar, orientation=orientation, ticks=colorbar_ticks)
    return cbar


def add_colorbar_2d(fig: Figure, cmap_name: str, colorbar_ticks: tuple[float, float, float, float],
                    colorbar_location: tuple[float, float, float, float]=(0.425, 0.02, 0.15, 0.15), fontsize: int=12) -> AxesImage:
    """Add a 2D colorbar to a flatmap plot

    Parameters
    ----------
    fig : matplotlib Figure object
    cimg : matplotlib.image.AxesImage object
        Image for which to create colorbar. For reference, matplotlib.image.AxesImage 
        is the output of imshow()
    colorbar_ticks : tuple[float, float, float, float]
        Values for colorbar *extents*, in order [xmin, xmax, ymin, ymax]. The colorbar will be plotted with these values as the limits of the colorbar axes, and the ticks will be placed at the values specified in the first two and last two entries of this tuple.
    colorbar_location : array-like
        Four-long list, tuple, or array that specifies location for colorbar axes 
        [left, top, width, height] (?)
    orientation : string
        'vertical' or 'horizontal'
        TODO: unused
    """
    # a bit sketchy - lazy imports
    import matplotlib.pyplot as plt
    import os
    cmap_dir = config.get('webgl', 'colormaps')
    cim = plt.imread(os.path.join(cmap_dir, cmap_name + '.png'))
    fig, _ = _get_fig_and_ax(fig)
    fig.add_axes(colorbar_location)
    cbar = plt.imshow(cim, extent=colorbar_ticks, interpolation='bilinear')
    cbar.axes.set_xticks(colorbar_ticks[:2])
    cbar.axes.set_xticklabels(colorbar_ticks[:2], fontdict=dict(size=fontsize))
    cbar.axes.set_yticks(colorbar_ticks[2:])
    cbar.axes.set_yticklabels(colorbar_ticks[2:], fontdict=dict(size=fontsize))

    return cbar

def add_custom(fig: Figure, dataview: HasSubject, svgfile: str, layer: str, extents: Optional[tuple[float, float, float, float]]=None, height: Optional[int]=None, with_labels: bool=False, 
               shape_list: Optional[Sequence[str]]=None, **kwargs):
    """Add a custom data layer

    Parameters
    ----------
    fig : matplotlib figure
        Figure into which to plot the hatches. Should have pycortex flatmap image in it already.
    dataview : cortex.Volume
        cortex.Volume object containing
    svgfile : string
        Filepath for custom svg file to use. Must be formatted identically to overlays.svg
        file for subject in `dataview`
    layer : string
        Layer name within custom svg file to display
    extents : array-like
        4 values for [Left, Right, Bottom, Top] extents of image plotted. If None, defaults to 
        extents of images already present in figure.
    height : scalar
        Height of image. if None, defaults to height of images already present in figure. 
    with_labels : bool
        Whether to display text labels on ROIs
    shape_list : list of str, optional
        list of paths/shapes within svg layer to render, if only a subset of
        the paths/shapes within the layer are desired.

    Other Parameters
    ----------------
    kwargs : dict
        maps to svg keyword arguments for e.g. line width, color, etc

    Returns
    -------
    img : matplotlib.image.AxesImage
        matplotlib axes image object for plotted data

    """
    from ..svgoverlay import get_overlay
    if height is None:
        height = _get_height(fig)
    if extents is None:
        extents = _get_extents(fig)
    pts_, polys_ = db.get_surf(dataview.subject, "flat", merge=True, nudge=True)
    extra_svg = get_overlay(dataview.subject, svgfile, pts_, polys_)
    svg_kws = _convert_svg_kwargs(kwargs)
    try:
        # Check for layer if it exists
        layer_kws = _parse_defaults(layer+'_paths')
        layer_kws.update(svg_kws)
    except:
        layer_kws = svg_kws
    im = extra_svg.get_texture(layer, height, 
                               labels=with_labels, 
                               shape_list=shape_list, 
                               **layer_kws)
    _, ax = _get_fig_and_ax(fig)
    img = ax.imshow(im, 
                    aspect="equal", 
                    interpolation="nearest", 
                    extent=extents,  
                    label='custom',
                    zorder=6)
    return img

def add_connected_vertices(fig: Axes, dataview: VolumetricView, exclude_border_width: Optional[int]=None,
                           height: Optional[int]=None, extents: Optional[tuple[float, float, float, float]]=None, recache: bool=False,
                           color: tuple[float, float, float, float]=(1.0, 0.5, 0.1, 0.6), linewidth: float=0.75,
                           alpha: float=1.0, **kwargs) -> LineCollection:
    """Plot lines btw distant vertices that are within the same voxel

    Parameters
    ----------
    fig : matplotlib figure
        Figure into which to plot the hatches. Should have pycortex flatmap
        image in it already.
    dataview : cortex.Volume
        cortex.Volume object containing data used to determine which vertices
        are connected.
    exclude_border_width : scalar or None
        if not None, width from edge of flatmap for which crossover lines are
        not computed.
    height : scalar or None
        Height of image. if None, defaults to height of images already present
        in figure.
    extents : array-like or None
        4 values for [Left, Right, Bottom, Top] extents of image plotted. If
        None, defaults to extents of images already present in figure.
    color : rgba tuple
        color of lines
    linewidth : scalar
        width of plotted lines
    alpha : scalar, [0-1]
        alpha value for plotted lines
    kwargs are mapped to cortex.db.get_shared_voxels

    Notes
    -----
    The process of drawing all the connected vertices is graphically intensive
    because of the sheer number of lines to draw. This is already partly sped
    up by using a LineCollection object instead of plotting each line,
    but it's still an expensive step, and takes quite a while on some systems.

    `extents` is currently unused, but probably should be to scale pix_array
    As a result, this may be brittle to some figure transformations.
    """
    from matplotlib.collections import LineCollection
    from scipy.ndimage import binary_dilation

    if extents is None:
        extents = _get_extents(fig)
    if height is None:
        height = _get_height(fig)            
    subject = dataview.subject
    xfmname = dataview.xfmname
    if xfmname is None:
        raise ValueError("Dataview for add_connected_vertices must be a Volume! You seem to have provided vertex data.")
    # print('computing shared voxels')
    shared_voxels = db.get_shared_voxels(subject, xfmname, recache=recache, **kwargs)
    # print('Finished computing shared voxels')
    mask, extents = get_flatmask(subject)
    pixmap = get_flatcache(subject, None)
    n_pixels, n_verts = pixmap.shape

    if exclude_border_width:
        # Finding vertices that map to the border of the flatmap
        img = np.nan * np.ones(mask.shape) 
        img[mask] = pixmap * np.arange(n_verts) # mapper.nverts
        border_mask = binary_dilation(~mask, iterations=exclude_border_width) ^ (~mask)
        border_vertices = set(img[border_mask].astype(int))
        shared_voxels = np.array([a for a in shared_voxels if ((a[1] not in border_vertices) and (a[2] not in border_vertices))])

    valid_vert_mask = np.array(pixmap.sum(0) > 0).flatten()
    valid_verts = np.arange(n_verts)[valid_vert_mask] # mapper.nverts
    # Assure both vertices in each pair are not in the medial wall
    vtx1valid = np.isin(shared_voxels[:, 1], valid_verts)
    vtx2valid = np.isin(shared_voxels[:, 2], valid_verts)
    va, vb = shared_voxels[vtx1valid & vtx2valid, 1:].T
    # Get X, Y coordinates per vertex, scale to 0-1 range
    [lpt, lpoly], [rpt, rpoly] = db.get_surf(subject, "flat", nudge=True)
    vert_xyz = np.vstack([lpt, rpt])
    vert_xyz -= vert_xyz.min(0)
    vert_xyz /= vert_xyz.max(0)
    x, y = vert_xyz[:, :2].T
    # Map vertices to X, Y coordinates suitable for LineCollection input
    pix_array_x = np.vstack([x[va], x[vb]]).T
    pix_array_y = np.vstack([y[va], y[vb]]).T
    pix_array_scaled = np.dstack([pix_array_x, pix_array_y])
    # Add line collection
    # (This is the most time consuming step, as it draws many lines)
    # print('plotting lines...')
    fig, ax = _get_fig_and_ax(fig)
    lc = LineCollection(pix_array_scaled,
                        transform=fig.transFigure,
                        figure=fig,
                        colors=color,
                        alpha=alpha,
                        linewidths=linewidth)
    lc_object = ax.add_collection(lc)
    return lc_object

def add_cutout(fig, name: str, dataview: HasSubject, layers=None, height=None,
               extents=None, overlay_file=None) -> None:
    """Apply a cutout mask to extant layers in flatmap figure

    Parameters
    ----------
    fig : figure or ax
        figure to which to add cutouts
    name : str
        name of cutout shape within cutouts layer to use to crop the rest of the figure
    dataview : cortex.dataset.HasSubject
        Any view; only its ``subject`` is read.
    layers : list of layers in svg object
        layers to which the cutout will be applied. None defaults to all.
        [unclear if it's worth it to keep this input.]
    height : int
        height of resulting figure. None defaults to height specified by other
        previous compositing functions. [unclear if it's worth it to keep this
        input.]
    extents : tuple | list
        extents of figure. None defaults to previously specified extents.
        [unclear if it's worth it to keep this input.]
    """
    if layers is None:
        layers = _get_images(fig)
    if height is None:
        height = _get_height(fig)
    if extents is None:
        extents = _get_extents(fig)
    svgobject = db.get_overlay(dataview.subject, overlay_file=overlay_file)
    # Set other cutouts to be invisible
    for co_name, co_shape in svgobject.cutouts.shapes.items():
        co_shape.visible = co_name == name
    # Get cutout image (now all white = 1, black = 0)
    svg_kws = _convert_svg_kwargs(dict(fillcolor="white", 
                                       fillalpha=1.0,
                                       linecolor="white", 
                                       linewidth=2))
    co = svgobject.get_texture('cutouts', height, labels=False, **svg_kws)[..., 0]
    if not np.any(co):
        raise Exception(f'No pixels in cutout region {name}!')

    # Bounding box indices
    LL, RR, BB, TT = np.nan, np.nan, np.nan, np.nan
    # Clip each layer to this cutout
    for layer_name, im_layer in layers.items():
        im = im_layer.get_array()

        # Reconcile occasional 1-pixel difference between flatmap image layers 
        # that are generated by different functions
        if not all([np.abs(aa - bb) <= 1 for aa, bb in zip(im.shape, co.shape)]):
            raise Exception("Shape mismatch btw cutout and data!")
        if any([np.abs(aa - bb) > 0 and np.abs(aa - bb) < 2 for aa, bb in zip(im.shape, co.shape)]):
            from PIL import Image

            # `co` is float32 in [0, 1] and stays that way. PIL's "F" mode resizes
            # 32-bit float directly, so the mask is not quantised to 1/255 steps on
            # the way through. Note the axis order: PIL takes (width, height) where
            # `im.shape[:2]` is (height, width).
            target = (int(im.shape[1]), int(im.shape[0]))
            layer_cutout = np.asarray(
                Image.fromarray(
                    np.ascontiguousarray(co, dtype=np.float32), mode="F"
                ).resize(target, Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
        else:
            layer_cutout = copy.copy(co)

        # Handle different types of alpha layers. Useful for RGBVolumes if nothing else.
        if im.dtype == np.uint8:
            # `np.asarray` before `astype`, so a masked layer degrades to its data
            # the way it always has rather than staying masked through the
            # multiplication below.
            im = np.asarray(im).astype(np.float32) / 255.0
            im[:,:,3] *= layer_cutout
            h, w, cdim = [float(v) for v in im.shape]
        else:
            if np.ndim(im)==3:
                im[:,:,3] *= layer_cutout
                h, w, cdim = [float(v) for v in im.shape]
            elif np.ndim(im)==2:
                im[layer_cutout==0] = np.nan
                h, w = [float(v) for v in im.shape]
        y, x = np.nonzero(layer_cutout)
        l, r, b, t = extents
        x_span = np.abs(r-l)
        y_span = np.abs(t-b)
        extents_new = [l + x.min() / w * x_span,
                       l + x.max() / w * x_span,
                       t + y.min() / h * y_span,
                       t + y.max() / h * y_span]

        # Bounding box indices
        iy, ix = ((y.min(), y.max()), (x.min(), x.max()))
        tmp = im[iy[0]:iy[1], ix[0]:ix[1]]
        im_layer.set_array(tmp)
        im_layer.set_extent(extents_new)

        # Track maxima / minima for figure
        LL = np.nanmin([extents_new[0], LL])
        RR = np.nanmax([extents_new[1], RR])
        BB = np.nanmin([extents_new[2], BB])
        TT = np.nanmax([extents_new[3], TT])
        imsize = (np.abs(np.diff(iy))[0], np.abs(np.diff(ix))[0])

    # Re-set figure limits
    fig, ax = _get_fig_and_ax(fig)
    ax.set_xlim(LL, RR)
    ax.set_ylim(BB, TT)
    inch_size = np.array(imsize)[::-1] / float(fig.dpi)
    fig.set_size_inches(inch_size[0], inch_size[1])

    return


#: Which matplotlib marker each electrode group type gets when none is named.
#: Shape carries the device -- a reader should not have to consult a legend to
#: tell a grid from a depth electrode, since the two mean quite different things
#: about how far to trust the position drawn.
MARKER_BY_GROUP_TYPE = {
    "grid": "o",
    "strip": "s",
    "seeg": "D",
    "depth": "D",
}

DEFAULT_MARKER = "o"


def add_electrodes(fig, electrodes: "ElectrodeSet", values: Optional[npt.ArrayLike]=None,
                   subject: Optional[str]=None,
                   depth: Optional[Union[float, tuple[float, float]]]=None,
                   depth_tol: float=0.25, placeable_only: bool=True,
                   cmap: Optional[str]=None, vmin: Optional[float]=None, vmax: Optional[float]=None,
                   color: Optional[ColorType]=None,
                   size: float=36.0, size_by: Optional[str]=None,
                   size_range: tuple[float, float]=(12.0, 110.0),
                   marker: Optional[Union[str, dict]]=None,
                   edgecolor: ColorType="k", linewidth: float=0.5, alpha: Optional[float]=None,
                   with_labels: bool=False, labelsize: float=6.0, labelcolor: ColorType="k",
                   zorder: int=1001, **kwargs: Any) -> dict[str, PathCollection]:
    """Add intracranial electrode markers to a flatmap figure.

    NOTE: zorder for electrodes is 1001, above the ROI layer, so contacts stay
    visible wherever they land.

    Parameters
    ----------
    fig : figure or ax
        figure into which to plot the electrodes
    electrodes : cortex.electrodes.ElectrodeSet
        The electrodes to draw. Anchored in place if it is not already, which
        needs the set to know its subject.
    values : array-like, optional
        One value per electrode, colormapped through ``cmap``. Length must match
        the *whole* set, before any filtering below. None draws every marker in
        ``color``.
    subject : str, optional
        The subject the figure belongs to. When given, it must match the
        electrode set's own subject. Worth passing: electrodes from the wrong
        subject land on plausible-looking cortex and produce a figure that is
        wrong rather than empty.
    depth : float or (float, float), optional
        Select on normalised cortical depth -- 0 at the pia, 1 at the white
        matter, negative outside. A float keeps electrodes within ``depth_tol``
        of it, a pair keeps a closed range, and None (the default) keeps
        everything. This is the same coordinate as the webgl viewer's depth
        slider, so a figure and a viewer can be made to agree.
    depth_tol : float
        Half-width of the band kept when ``depth`` is a single value.
    placeable_only : bool
        Drop electrodes the placement policy rejected -- those too far from any
        cortical column to have an honest surface position. On by default; the
        rejected ones are still in the set, so turning this off shows what was
        excluded rather than resurrecting anything.
    cmap : str, optional
        Colormap for ``values``. Defaults to matplotlib's current default.
    vmin, vmax : float, optional
        Colormap limits. Default to the range of ``values`` after filtering.
    color : matplotlib colorspec, optional
        A single colour for every marker, used when ``values`` is None.
        Defaults to white with a dark edge, which reads against both the light
        and dark bands of a curvature background.
    size : float
        Marker area in points squared.
    size_by : str, optional
        Name of a numeric electrode field -- in practice ``"size"``, the contact
        diameter in millimetres -- to scale markers by, mapped onto
        ``size_range``. Overrides ``size``.
    size_range : (float, float)
        Smallest and largest marker area when ``size_by`` is used.
    marker : str or dict, optional
        A matplotlib marker for every electrode, or a dict from group type to
        marker. None uses :data:`MARKER_BY_GROUP_TYPE`.
    edgecolor, linewidth, alpha : optional
        Passed through to ``scatter``.
    with_labels : bool
        Annotate each marker with its channel name.
    labelsize, labelcolor : optional
        Font size and colour for those labels.
    zorder : int
        Drawing order.

    Returns
    -------
    dict of str to matplotlib.collections.PathCollection
        One collection per marker shape drawn, keyed by the marker. They share a
        colormap and norm, so any one of them serves as the mappable for a
        colorbar.

    Notes
    -----
    Markers are a constant size in figure units; they are *not* scaled by local
    areal distortion. Flattening stretches cortex unevenly, by a factor of two
    or three in places, so a marker drawn to cover a fixed cortical area would
    change size across the map and invite the reader to compare sizes that mean
    nothing. Constant size says "a contact is here" and no more, which is what a
    contact position supports. Pass ``size_by="size"`` to encode the real
    contact diameter deliberately.

    Call this after :func:`add_cutout` if you use one. A cutout clips image
    layers and resets the axis limits; markers outside those limits vanish with
    it, but a marker inside the cutout's bounding box and outside its outline
    will still be drawn.
    """
    from ..electrodes import ElectrodeSet  # noqa: F401  (runtime type check below)

    if not isinstance(electrodes, ElectrodeSet):
        raise TypeError(
            "add_electrodes needs a cortex.electrodes.ElectrodeSet, got %s"
            % type(electrodes).__name__
        )
    if subject is not None and electrodes.subject not in (None, subject):
        raise ValueError(
            "these electrodes belong to subject %r but the figure is subject %r"
            % (electrodes.subject, subject)
        )
    if electrodes.anchors is None:
        electrodes.anchor(subject=subject)
    anchors = electrodes.anchors
    assert anchors is not None                       # anchor() just set it

    values_arr = None
    if values is not None:
        values_arr = np.asarray(values, dtype=np.float64).ravel()
        if len(values_arr) != len(electrodes):
            raise ValueError(
                "values has %d entries but there are %d electrodes"
                % (len(values_arr), len(electrodes))
            )

    keep = np.ones(len(electrodes), dtype=bool)
    if placeable_only:
        keep &= anchors.placeable
    if depth is not None:
        d = electrodes.depth
        if isinstance(depth, (int, float, np.floating, np.integer)):
            in_band = np.abs(d - float(depth)) <= depth_tol
        else:
            lo, hi = depth
            in_band = (d >= lo) & (d <= hi)
        # An electrode whose depth is unknown -- a subject with no white-matter
        # surface -- cannot fail a depth test, so it is kept rather than
        # silently disappearing when a depth argument is added.
        keep &= in_band | ~np.isfinite(d)

    if not keep.any():
        return {}
    # `select` rather than `electrodes[keep]`: indexing is overloaded to return a
    # single ElectrodeInfo for a scalar index, so only this spelling is a set.
    selected = electrodes.select(where=keep)
    if values_arr is not None:
        values_arr = values_arr[keep]

    positions = selected.positions("flat", subject=subject, nudge=True)

    # Colours
    norm = None
    if values_arr is not None:
        from matplotlib.colors import Normalize
        norm = Normalize(
            vmin=np.nanmin(values_arr) if vmin is None else vmin,
            vmax=np.nanmax(values_arr) if vmax is None else vmax,
        )
    elif color is None:
        color = "white"

    # Sizes
    if size_by is not None:
        raw = np.asarray(selected._field(size_by), dtype=np.float64)
        finite = np.isfinite(raw)
        sizes = np.full(len(selected), float(np.mean(size_range)))
        if finite.any():
            lo, hi = np.nanmin(raw[finite]), np.nanmax(raw[finite])
            frac = np.zeros_like(raw) if hi == lo else (raw - lo) / (hi - lo)
            sizes[finite] = size_range[0] + frac[finite] * (size_range[1] - size_range[0])
    else:
        sizes = np.full(len(selected), float(size))

    # One scatter call per marker shape, since matplotlib takes one marker each.
    marker_of = _electrode_markers(selected, marker)
    _, ax = _get_fig_and_ax(fig)
    collections: dict[str, PathCollection] = {}
    for shape in sorted(set(marker_of)):
        sel = marker_of == shape
        scatter_kws = dict(
            s=sizes[sel], marker=shape, edgecolors=edgecolor, linewidths=linewidth,
            alpha=alpha, zorder=zorder, label="electrodes", **kwargs,
        )
        if values_arr is None:
            scatter_kws["c"] = [color] * int(sel.sum())
        else:
            scatter_kws.update(c=values_arr[sel], cmap=cmap, norm=norm)
        collections[shape] = ax.scatter(positions[sel, 0], positions[sel, 1], **scatter_kws)

    if with_labels:
        for i, name in enumerate(selected.names):
            ax.annotate(str(name), (positions[i, 0], positions[i, 1]),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=labelsize, color=labelcolor, zorder=zorder + 1)

    return collections


def _electrode_markers(electrodes: "ElectrodeSet",
                       marker: Optional[Union[str, dict]]) -> npt.NDArray[np.str_]:
    """One matplotlib marker per electrode."""
    if isinstance(marker, str):
        return np.array([marker] * len(electrodes), dtype=object)
    table = MARKER_BY_GROUP_TYPE if marker is None else marker
    return np.array(
        [table.get(str(g).lower(), DEFAULT_MARKER) for g in electrodes.group_type],
        dtype=object,
    )
