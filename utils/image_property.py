class ImageProperty(object):
    """ Class to store parameters of one WholeSlideImage type to use

    Attributes
    ----------
    staining : string
        staining to use same as in WSI file name
    scanner : string
        scanner to use same as in WSI file name
    magnification : int
        magnification to use same as in WSI file name
    preparation : string
        Preparation type to use currently not considered
    used_magnification : int
        magnification to use from image pyramid - closest will be used
    use_only_stamp : bool
        Filter for WSIs with Block/Subblock combination as mentioned in csv
        
    Methods
    -------
    get_staining()
        Return staining
    get_scanner()
        Return scanner
    get_magnification()
        Return magnification
    get_preparation()
        Return preparation
    get_file_ending()
        Return fie type of WSI of used scanner
    get_used_magnification()
        Return used_magnification
    get_use_only_stamp()
        Return use_only_stamp
    """
    def __init__(self, staining, scanner, magnification, preparation, used_magnification, use_only_stamp=False):
        """
        Parameters
        ----------
        staining : string
            Staining to use same as in WSI file name
        scanner : string
            Scanner to use same as in WSI file name
        magnification : int
            Magnification of WSI to use same as in WSI file name
        preparation : string
            Preparation type to use - currently unused
        used_magnification : int
            magnification to use from image pyramid - closest will be used
        use_only_stamp : bool
            Filter for WSIs with Block/Subblock combination as mentioned in csv
        """
        # Staining of WSI to use
        self.staining = staining
        # Scanner to use
        self.scanner = scanner
        # Magnification to use
        self.magnification = magnification
        # Fixation to use
        self.preparation = preparation
        # Magnification factor in image pyramid to use <= magnification
        self.used_magnification = used_magnification
        # Filter for WSIs with Block/Subblock combination as mentioned in csv
        self.use_only_stamp = use_only_stamp


    def get_staining(self):
        """ Return staining
        Returns
        -------
        string
            staining
        """
        return self.staining

    def get_scanner(self):
        """ Return scanner
        Returns
        -------
        string
            scanner
        """
        return self.scanner

    def get_magnification(self):
        """ Return magnification
        Returns
        -------
        int
            magnification
        """
        return self.magnification

    def get_preparation(self):
        """ Return preparation
        Returns
        -------
        string
            preparation
        """
        return self.preparation

    def get_file_ending(self):
        """ Return file type for used scanner
        Returns
        -------
        string
            file type of scanner extend for new scanners
        """
        # Extend for new Scanners
        file_endings = {
            'Leica':'.svs',
            'Hamamatsu':'.ndpi'
        }

        return file_endings[self.scanner]

    def get_used_magnification(self):
        """ Return used_magnification
        Returns
        -------
        int
            used_magnification
        """
        return self.used_magnification
    
    def get_use_only_stamp(self):
        """ Return use_only_stamp
        Returns
        -------
        bool
            use_only_stamp
        """
        return self.use_only_stamp
    
    def get_folder_name(self):
        return self.staining + '_' + self.scanner + '_' + str(self.magnification) + \
              '_' + self.preparation + '_' + str(self.used_magnification) + \
              '_' + str(self.use_only_stamp)