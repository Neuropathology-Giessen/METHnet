import cv2
import numpy as np

from datastructure.tile import Tile



def filter_overview_image(img):
    """ Filter the background of the overview image according to Otus's Thresholding

    Parameters
    ----------
    img : Image
        The overview image

    Returns
    -------
    numpy array 
        The mask for the overview image tissue is 1
    float
        The otsu threshold value used
    """
    # Cast overview image to Numpy array
    img = np.array(img)
    # To grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

    # Threshold defined by Otsu's method
    otsu_value, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Negate as we want tissue
    img = cv2.bitwise_not(img)

    kernel_size = 7
    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Close gaps in tissue
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, structuring_element)
    # Ignore small isolated parts
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, structuring_element)

    return img, otsu_value


def apply_filter(img):
    pass

def filtering_blood(img):
    """ Filter blood from Image according to fixed red values

    Parameters
    ----------
    img : Image
        The image to filter

    Returns
    -------
    numpy array
        The mask 1 is blood
    """
    # Convert image to HSV
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Compute Red values
    lower_red = np.array([0, 110, 70])
    upper_red = np.array([20, 255, 255])
    
    # Get mask for blood
    mask_1 = cv2.inRange(img, lower_red, upper_red)

    # Compute second Red values
    lower_red = np.array([170, 120, 70])
    upper_red = np.array([200, 255, 255])
    
    # Get second blood mask
    mask_2 = cv2.inRange(img, lower_red, upper_red)
    
    # Get complete blood mask
    mask = mask_1 + mask_2
    mask = np.array(mask)

    return mask

def filtering_tissue(img, otsu_value):
    """ Filter image according to fixed threshold
    
    Parameters
    ----------
    img : Image
        The image to filter 
    otsu_value : float
        The threshold value

    Returns
    -------
        The mask 1 is tissue
    """
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2GRAY)
    # get background mask using precomputed otsu (whole image)
    _, background_mask = cv2.threshold(img, otsu_value, 255, cv2.THRESH_BINARY)

    # tissue is not background
    tissue_mask = cv2.bitwise_not(background_mask)

    kernel_size = 5
    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    iterations = 2
    # Close mask to take account from small non tissue parts
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, structuring_element, iterations=iterations)
    # Nonetheless eliminate small e.g. dust parts
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, structuring_element, iterations=iterations)

    return tissue_mask,


def check_tissue(overview_image, position, size, downsampling_factor, factor, percentage=0.1):
    """ Check wheter tissue mask is more than percent in overview mask

    Parameters
    ----------
    overview_image : numpy array
        The tissue mask for the overview image
    position : tuple (int, int)
        The position of the region to check in pixel (x, y)
    size : tuple (int, int)
        The size of the region to check in pixel (x, y)
    downsampling_factor : float
        The factor between the used image level and the overview image
    factor : float
        The factor between the highest and lowest image pyramd level
    percentage : float
        The minimum tissue percentage to accept the region

    Returns 
    -------
    bool
        True if tissue percentage is over percentage
    """
    # Check if over percentage according to overview image
    overview_image = np.array(overview_image)
    # compute region in overview

    x = int(position[0] / factor)
    x_end = x + int(size[0] / downsampling_factor)

    y = int(position[1] / factor)
    y_end = y + int(size[1] / downsampling_factor)

    region = overview_image[y:y_end, x:x_end]
    # if over percentage non background return true
    return np.mean(region / 255.) > percentage

def check_tissue_tile(region, otsu_value, percentage=0.1, filter_background=True, filter_blood=True):
    """ Check wheter tissue percentage is over minimum percentage without blood and background

    Parameters
    ----------
    region : Image
        The region to check
    otsu_value : float
        The original Otsu value used to threshold the overview image
    percentage : float
        The minimum percentage to accept the region
    filter_background : bool
        True if want to substract background from percentage
    filter_blood : bool
        True if want to substract blood from percentage

    Returns
    -------
    bool
        True if valid region
    """
    # get tissue and blood masks
    tissue_mask = filtering_tissue(region, otsu_value)
    blood_mask = filtering_blood(region)

    tissue_percentage = np.mean(tissue_mask) / 255.
    blood_percentage = np.mean(blood_mask) / 255. 

    # Initial percentage is 100%
    score = 1. 

    # If filtering background tissue percentage is score
    if filter_background:
        score = tissue_percentage
    
    # If filtering blood total score is minus blood
    if filter_blood:
        score = score - blood_percentage
        
    # if more tissue than blood and over percentage tissue return true
    return score >= percentage

def multiprocess_filtering(x, otsu_value, tile_property, level):
    """ Check wheter at a position a valid is tile is possible. Filtering may be applied to check for background and blood.

    Parameters
    ----------
    x : tuple
        (Image, Position) where Image is a pillow Image of the Tile and Position is a tuple of its location in pixel (x,y) 
    otsu_value : int
        The original otsu value as obtained by filtering the overview image
    tile_property : TileProperty
        Properties of the tile to generate
    level : int
        Level of the image pyramid where tile was extracted

    Returns
    -------
    Tile
        Tile object if minimum percentage of tissue is in image else None
    """
    # Get image region
    region = x[0]
    # Get position of image region
    position = x[1]
    # If want to filter for background or blood
    if tile_property.get_filter_background() or tile_property.get_filter_blood():
        # Check single tile if it contains enough tissue
        is_tissue = check_tissue_tile(region, otsu_value, tile_property.get_min_tissue_percentage(),
                                       tile_property.get_filter_background(), tile_property.get_filter_blood())
    else:
        # Else it is tissue
        is_tissue = True
    
    # If valid Tile create it
    if is_tissue:
        # Create Tile object
        tile_obj = Tile(position[0], position[1], tile_property.get_tile_size(),
        tile_property.get_input_size(), tile_property.get_augmentations(), level)
        # Return tile object
        return tile_obj
    # Else return None
    return None