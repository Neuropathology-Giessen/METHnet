import qupath.lib.images.servers.LabeledImageServer

def imageData = getCurrentImageData()

def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
def pathOutput = buildFilePath(PROJECT_BASE_DIR,'export')
mkdirs(pathOutput)
def path = buildFilePath(pathOutput, name + "-labels.png")

double downsample = 256
resultingClass = getPathClass("Positive")
toChange = getAnnotationObjects().findAll{it.getPathClass() == null}
toChange.each{ it.setPathClass(resultingClass)}

def labelServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0, ColorTools.WHITE)
    .downsample(downsample)
    .addLabel('Positive',1)
    .multichannelOutput(false)
    .build()

writeImage(labelServer, path)