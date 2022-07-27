import qupath.lib.roi.ROIs
import qupath.lib.regions.ImagePlane
import qupath.lib.objects.PathObjects
// Parameters
def rel ='_relative'
def magnification = 40
def magnification_used = 20
def tile_size_x = 256
def tile_size_y = 256
def tile_overlap_x = 0
def tile_overlap_y = 0



def filename = folder + '/' +  name + '_' + magnification_used + '_256_256_0_0'+'.json'
def factor = magnification / magnification_used

// Get Image
def plane = ImagePlane.getPlane(0, 0)
def imageData = getCurrentImageData()
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())

def folder =buildFilePath(PROJECT_BASE_DIR,'../Attention/')ö

import qupath.lib.io.GsonTools
import qupath.lib.objects.classes.PathClassFactory
// get json
def gson = GsonTools.getInstance(true)
def json = new File(filename).text
json = json[1..(json.length()-2)]
json = json.replace("\\","")
def mapping = gson.fromJson(json, Map.class)
def annotations = []
def n_tiles = mapping["tiles"].keys.size()
// Clear current annotation to blank
clearAllObjects()
// Iterate Tiles
for (i = 0; i < n_tiles; i++) {
    // Create tile annotation
    def roi_annotation = ROIs.createRectangleROI(mapping["tiles"][i]["x"],mapping["tiles"][i]["y"],256*factor,256*factor,plane)
    // color for Tile
    r = (int)(mapping["tiles"][i]["R"+rel])
    g = (int)(mapping["tiles"][i]["G"+rel])
    b = (int)(mapping["tiles"][i]["B"+rel])
    
    v = mapping["tiles"][i]["class"]
    // Create class for each tile for continous mapping
    roiClass = getPathClass(v.toString())
    roiColor = getColorRGB(r,g,b)
    // Set color
    roiClass.setColor(roiColor)
    // Add Tile annotation
    annotations << PathObjects.createAnnotationObject(roi_annotation, roiClass)
}
// Add annotations
addObjects(annotations)