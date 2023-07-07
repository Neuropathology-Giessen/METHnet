import os

class TileProperty(object):
    """ Class to store parameters for Tiles to generate

    Attributes
    ----------
    tile_size : tuple (int, int)
        Size of Tile in pixel (x, y)
    tile_overlap : tuple (int, int)
        Overlap between two adjacent tiles in pixel (x, y)
    input_size : tuple (int, int)
        Expected size of image for Encoder in pixel as tuple (x, y) should be smaller or equal to tile_size
    augmentations : Albumentations object
        Augmentation sequence to apply to Tiles
    filter_mask : bool
        True if filter Tiles outside corresponding mask
    mask_folder : string
        Folder where masks for filtering are stored
    filter_background : bool
        True if filter Tiles that contain too much background
    filter_blood : bool
        True if filter Tiles that contain too much blood
    min_tissue_percentage : float
        Minimum percentage of tissue excluded filtered content

    Methods
    -------
    get_tile_size()
        Return tile_size
    get_tile_overlap()
        Return tile_overlap
    get_input_size()
        Return input_size
    get_augmentations()
        Return augmentations
    get_filter_mask()
        Return filter_mask
    get_mask_folder()
        Return mask_folder
    get_filter_background()
        Return filter_background
    get_filter_blood()
        Return filter_blood
    get_min_tissue_percentage()
        Return min_tissue_percentage
    """
    def __init__(self, tile_size, tile_overlap, input_size, augmentations, filter_mask=False, mask_folder='',
                 filter_background=True, filter_blood=True, min_tissue_percentage=0.75):
        """
        Parameters
        ----------
        tile_size : tuple (int, int)
            Size of tile in pixel (x, y)
        tile_overlap : tuple (int, int)
            Overlap between two adjacent tiles in pixel (x, y)
        input_size : tuple (int, int)
            Expected size for Encoder in pixel (x, y)
        augmentations : Albumentations object
            Augmentation sequence to apply to Tiles
        filter_mask : bool
            True if filter Tiles outside corresponding mask
        mask_folder : string
            Folder where masks for filtering are stored
        filter_background : bool
            True if filter Tiles that contain too much background
        filter_blood : bool
            True if filter Tiles that contain too much blood
        min_tissue_percentage : float
            Minimum percentage of tissue excluded filtered content
        """
        # Size of tile in pixel as tuple
        self.tile_size = tile_size
        # Overlap of tiles in pixel
        self.tile_overlap = tile_overlap
        # Size of input for Encoder network in pixel as tuple <= tile_size
        self.input_size = input_size
        # Augmentation sequence to use - see albumentations
        self.augmentations = augmentations
        # If True filter Tiles outside corresponding mask
        self.filter_mask = filter_mask
        # Folder where corresponding masks are stored
        self.mask_folder = mask_folder

        # Check if mask folder is valid otherwise can not filter
        if self.filter_mask and ((self.mask_folder == '') or (not os.path.exists(mask_folder))):
            print("Label Maps not available can not filter according to mask")
            self.filter_mask = False

        # If True Tiles that contain background too much are filtered
        self.filter_background = filter_background
        # If True Tiles that contain blood too much are filtered
        self.filter_blood = filter_blood
        # Minimum percentage of tissue excluded filtered content for valid tiles
        self.min_tissue_percentage = min_tissue_percentage

    def get_tile_size(self):
        """ Return tile_size
        Returns
        -------
        tuple (int, int)
            tile_size
        """
        return self.tile_size

    def get_tile_overlap(self):
        """ Return tile_overlap
        Returns
        -------
        tuple (int, int)
            tile_overlap
        """
        return self.tile_overlap

    def get_input_size(self):
        """ Return input_size
        Returns
        -------
        tuple (int, int)
            input_size
        """
        return self.input_size

    def get_augmentations(self):
        """ Return augmentations
        Returns
        -------
        Albumentations object
            augmentations
        """
        return self.augmentations

    def get_filter_mask(self):
        """ Return filter_mask 
        Returns
        -------
        bool
            filter_mask
        """
        return self.filter_mask
    
    def get_mask_folder(self):
        """ Return mask_folder
        Returns
        -------
        string
            mask_folder
        """
        return self.mask_folder
    
    def get_filter_background(self):
        """ Return filter_background
        Returns
        -------
        bool
            filter_background
        """
        return self.filter_background
    
    def get_filter_blood(self):
        """ Return filter_blood
        Returns
        -------
        bool
            filter_blood
        """
        return self.filter_blood
    
    def get_min_tissue_percentage(self):
        """ Return min_tissue_percentage
        Returns
        -------
        float
            min_tissue_percentage
        """
        return self.min_tissue_percentage

    def get_folder_name(self):
        return str(self.tile_size[0]) + '_' + str(self.tile_size[1]) + '_' + \
            str(self.tile_overlap[0]) + '_' + str(self.tile_overlap[1]) + '_' + \
            str(self.input_size[0]) + '_' + str(self.input_size[1]) + '_' + \
            str(self.filter_mask) + '_' + str(self.filter_background) + '_' + \
            str(self.filter_blood) + '_' + str(self.min_tissue_percentage)