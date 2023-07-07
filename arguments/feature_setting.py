class FeatureSetting(object):
    """ Class to hold parameters for encoded features

    Attributes
    ----------
    skip_existing : bool
        True if not want to encode already existing features
    augment_tiles : bool
        True if apply augmentation sequence to tiles
    batch_size : int
        Size of batch for feature encoding should be chosen according to Memory and GPUs
    feature_dimension : int
        Size of feature vector
    
    Methods
    -------
    get_skip_existing()
        Return skip_existing
    get_augment_tiles()
        Return augment_tiles
    get_batch_size()
        Return batch_size
    get_feature_dimension()
        Return feature_dimension
    """
    def __init__(self):
        """
        """
        # Set True if want to skip already encoded features
        self.skip_existing = True   #TODO automatize check
        # Set True if want to augment tiles according to augmentation sequence during training
        self.augment_tiles = True   #TODO split in augmentation and normalization sequence?
        # Batch Size for feature encoding
        self.batch_size = 256
        # Feature Dimension to use
        self.feature_dimension = 1024

    def get_skip_existing(self):
        """ Return skip_existing
        Returns
        -------
        bool
            skip_exisiting attribute
        """
        return self.skip_existing

    def get_augment_tiles(self):
        """ Return augment_tiles
        Returns
        -------
        bool 
            augment_tiles attribute
        """
        return self.augment_tiles

    def get_batch_size(self):
        """ Return batch_size
        Returns
        -------
        int
            batch_size attribute
        """
        return self.batch_size

    def get_feature_dimension(self):
        """ Return feature_dimension
        Returns
        -------
        int
            feature_dimension attribute
        """
        return self.feature_dimension