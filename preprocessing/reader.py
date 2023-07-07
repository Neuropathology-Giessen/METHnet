import openslide as slide

def read_wsi(file_name):
    """ Open WSI choice of methods according to file ending
    
    Parameters
    ----------
    file_name : string
        Path to WSI
    
    Returns
    -------
    OpenSlideObject
        The WSI
    """
    # Read WSI according to file type
    file_ending = file_name.split('.')[-1]

    return file_reader[file_ending](file_name)

def read_openslide(file_name):
    """ Open WSI .ndpi

    Parameters
    ----------
    file_name : string
        Path to .ndpi

    Returns
    -------
    OpenSlideObject
        The WSI
    """
    wsi = slide.open_slide(file_name)

    return wsi

def read_region(wsi, corner=(0, 0), size=(256, 256), level_downsamples=0):
    """ Read a region from a WSI.
    
    Parameters
    ----------
    wsi : OpenSlideObject
        The WSI
    corner : tuple (int, int)
        The left upper corner of the region to read in pixel (x, y)
    size : tuple (int, int)
        The size of the region to read in pixel (x, y)
    level_downsamples : int
        The level of the image pyramid to read the region from

    Returns
    -------
    Image
        The image data for this region
    """
    # Read Image region 
    region = wsi.read_region(corner, level_downsamples, size)

    return region

def get_overview(wsi):
    """ Read the overview image lowest resolution image pyramid level

    Parameters
    ----------
    wsi : OpenSlideObject
        The WSI

    Returns
    -------
    Image
        The image data for the overview
    float
        The factor at which overview is downsampled compared to highest resolution
    """
    # Get Lowest magnification in Image Pyramid
    lowest_level = len(wsi.level_dimensions) - 1
    # Read Overview Image
    overview_image = wsi.read_region((0, 0), lowest_level, wsi.level_dimensions[lowest_level])
    # Get Downsampling factor compared to native
    downsampling_factor = wsi.level_downsamples[lowest_level]

    return overview_image, downsampling_factor

file_reader = {'svs':read_openslide, 'ndpi':read_openslide}
scanner = {'svs':'Leica', 'ndpi':'Hamamatsu'}

