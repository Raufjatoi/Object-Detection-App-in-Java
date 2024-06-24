import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class YoloObjectDetection {
    static {
        // Load the OpenCV library
        System.load("D:\\Downloads\\opencv\\build\\java\\x64\\opencv_java490.dll");
    }

    public static void main(String[] args) {
        // Load OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Define the file paths
        String folderPath = "D:\\Documents\\GitHub\\java\\";
        String inputImagePath = folderPath + "group.jpg";
        String modelConfiguration = folderPath + "yolov4.cfg";
        String modelWeights = folderPath + "yolov4.weights";
        String classNamesFile = folderPath + "coco.names";

        // Load input image
        Mat image = Imgcodecs.imread(inputImagePath);
        if (image.empty()) {
            System.err.println("Cannot read image: " + inputImagePath);
            return;
        }

        // Load YOLO model
        Net net = Dnn.readNetFromDarknet(modelConfiguration, modelWeights);
        if (net.empty()) {
            System.err.println("Cannot load network using given configuration and weights files.");
            return;
        }

        // Load class names
        List<String> classNames = loadClassNames(classNamesFile);
        if (classNames.isEmpty()) {
            System.err.println("Cannot load class names.");
            return;
        }

        // Prepare the image for YOLO
        Mat blob = Dnn.blobFromImage(image, 1 / 255.0, new Size(416, 416), new Scalar(0, 0, 0), true, false);

        // Set input to the model
        net.setInput(blob);

        // Run forward pass to get output of the output layers
        List<Mat> result = new ArrayList<>();
        List<String> outBlobNames = getOutputNames(net);
        net.forward(result, outBlobNames);

        // Convert Mat to BufferedImage for drawing and annotation
        BufferedImage bufImage = matToBufferedImage(image);

        // Draw rectangles around detected objects and annotate
        Graphics2D g2d = bufImage.createGraphics();
        g2d.setStroke(new BasicStroke(2));
        Font font = new Font("Arial", Font.BOLD, 20); // Larger font size
        g2d.setFont(font);

        // Define colors for each class
        Map<String, Color> colorMap = createColorMap();

        float confThreshold = 0.5f;
        int maxObjects = 20; // Maximum number of objects to detect
        int objectsDetected = 0; // Counter for detected objects
        Map<String, Rect> objectMap = new HashMap<>(); // Map to store merged objects

        // Iterate over each level of the result
        for (Mat level : result) {
            for (int i = 0; i < level.rows(); i++) {
                Mat row = level.row(i);
                Mat scores = row.colRange(5, level.cols());
                Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                float confidence = (float) mm.maxVal;
                if (confidence > confThreshold) {
                    int classId = (int) mm.maxLoc.x;
                    int centerX = (int) (row.get(0, 0)[0] * image.cols());
                    int centerY = (int) (row.get(0, 1)[0] * image.rows());
                    int width = (int) (row.get(0, 2)[0] * image.cols());
                    int height = (int) (row.get(0, 3)[0] * image.rows());
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    // Check if a similar object already exists within a margin
                    boolean foundSimilar = false;
                    for (Map.Entry<String, Rect> entry : objectMap.entrySet()) {
                        Rect existingRect = entry.getValue();
                        if (areSimilar(existingRect, new Rect(left, top, width, height), 20)) {
                            foundSimilar = true;
                            break;
                        }
                    }

                    if (!foundSimilar) {
                        // Draw rectangle
                        Color color = colorMap.get(classNames.get(classId));
                        g2d.setColor(color);
                        g2d.drawRect(left, top, width, height);

                        // Annotate object
                        g2d.setColor(Color.BLACK); // Set label color to white
                        String label = classNames.get(classId) + ": " + new DecimalFormat("#.##").format(confidence);
                        int textX = left;
                        int textY = top - 10; // Adjust vertical position
                        g2d.drawString(label, textX, textY);

                        // Store the object in the map
                        objectMap.put(label, new Rect(left, top, width, height));

                        // Increment the object counter
                        objectsDetected++;

                        // Check if we have reached the maximum number of objects
                        if (objectsDetected >= maxObjects) {
                            break; // Stop processing further detections
                        }
                    }
                }
            }
            if (objectsDetected >= maxObjects) {
                break; // Stop processing further levels of result
            }
        }

        // Display the number of objects detected
        g2d.setColor(Color.WHITE);
        g2d.setFont(new Font("Arial", Font.BOLD, 24)); // Larger font size for count
        String countLabel = "Objects Detected: " + objectsDetected;
        g2d.drawString(countLabel, 20, 30);

        g2d.dispose();

        // Convert BufferedImage back to Mat
        image = bufferedImageToMat(bufImage);

        // Save annotated image
        String outputImagePath = folderPath + "detected.JPG";
        Imgcodecs.imwrite(outputImagePath, image);
    }

    // Define a method to check if two rectangles (objects) are similar enough to merge
    private static boolean areSimilar(Rect rect1, Rect rect2, int margin) {
        return Math.abs(rect1.x - rect2.x) < margin &&
                Math.abs(rect1.y - rect2.y) < margin &&
                Math.abs(rect1.width - rect2.width) < margin &&
                Math.abs(rect1.height - rect2.height) < margin;
    }

    private static List<String> loadClassNames(String filename) {
        List<String> classNames = new ArrayList<>();
        try {
            classNames = Files.readAllLines(Paths.get(filename));
        } catch (Exception e) {
            e.printStackTrace();
        }
        return classNames;
    }

    private static List<String> getOutputNames(Net net) {
        List<String> names = new ArrayList<>();
        List<Integer> outLayers = net.getUnconnectedOutLayers().toList();
        List<String> layersNames = net.getLayerNames();
        for (int i : outLayers) {
            names.add(layersNames.get(i - 1));
        }
        return names;
    }

    // Utility method to convert Mat to BufferedImage
    private static BufferedImage matToBufferedImage(Mat matrix) {
        int cols = matrix.cols();
        int rows = matrix.rows();
        int elemSize = (int) matrix.elemSize();
        byte[] data = new byte[cols * rows * elemSize];
        int type;

        matrix.get(0, 0, data);

        switch (matrix.channels()) {
            case 1:
                type = BufferedImage.TYPE_BYTE_GRAY;
                break;
            case 3:
                type = BufferedImage.TYPE_3BYTE_BGR;
                // bgr to rgb
                byte b;
                for (int i = 0; i < data.length; i = i + 3) {
                    b = data[i];
                    data[i] = data[i + 2];
                    data[i + 2] = b;
                }
                break;
            default:
                return null;
        }

        BufferedImage image = new BufferedImage(cols, rows, type);
        image.getRaster().setDataElements(0, 0, cols, rows, data);

        return image;
    }

    // Utility method to convert BufferedImage to Mat
    private static Mat bufferedImageToMat(BufferedImage image) {
        byte[] data = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        int width = image.getWidth();
        int height = image.getHeight();
        int channels = image.getSampleModel().getNumBands();
        Mat mat = new Mat(height, width, channels == 3 ? CvType.CV_8UC3 : CvType.CV_8UC1);
        mat.put(0, 0, data);
        return mat;
    }

    private static Map<String, Color> createColorMap() {
        Map<String, Color> colorMap = new HashMap<>();
        colorMap.put("person", Color.BLUE);
        colorMap.put("bicycle", Color.CYAN);
        colorMap.put("car", Color.RED);
        colorMap.put("motorbike", Color.MAGENTA);
        colorMap.put("aeroplane", Color.ORANGE);
        colorMap.put("bus", Color.YELLOW);
        colorMap.put("train", Color.PINK);
        colorMap.put("truck", Color.LIGHT_GRAY);
        colorMap.put("boat", Color.GREEN);
        colorMap.put("traffic light", Color.RED);
        colorMap.put("fire hydrant", Color.RED);
        colorMap.put("stop sign", Color.RED);
        colorMap.put("parking meter", Color.BLUE);
        colorMap.put("bench", Color.GRAY);
        colorMap.put("bird", Color.GREEN);
        colorMap.put("cat", Color.GREEN);
        colorMap.put("dog", Color.GREEN);
        colorMap.put("horse", Color.GREEN);
        colorMap.put("sheep", Color.GREEN);
        colorMap.put("cow", Color.GREEN);
        colorMap.put("elephant", Color.GREEN);
        colorMap.put("bear", Color.GREEN);
        colorMap.put("zebra", Color.GREEN);
        colorMap.put("giraffe", Color.GREEN);
        colorMap.put("backpack", Color.YELLOW);
        colorMap.put("umbrella", Color.YELLOW);
        colorMap.put("handbag", Color.YELLOW);
        colorMap.put("tie", Color.YELLOW);
        colorMap.put("suitcase", Color.YELLOW);
        colorMap.put("frisbee", Color.MAGENTA);
        colorMap.put("skis", Color.CYAN);
        colorMap.put("snowboard", Color.CYAN);
        colorMap.put("sports ball", Color.MAGENTA);
        colorMap.put("kite", Color.MAGENTA);
        colorMap.put("baseball bat", Color.MAGENTA);
        colorMap.put("baseball glove", Color.MAGENTA);
        colorMap.put("skateboard", Color.CYAN);
        colorMap.put("surfboard", Color.CYAN);
        colorMap.put("tennis racket", Color.MAGENTA);
        colorMap.put("bottle", Color.BLUE);
        colorMap.put("wine glass", Color.BLUE);
        colorMap.put("cup", Color.BLUE);
        colorMap.put("fork", Color.BLUE);
        colorMap.put("knife", Color.BLUE);
        colorMap.put("spoon", Color.BLUE);
        colorMap.put("bowl", Color.BLUE);
        colorMap.put("banana", Color.YELLOW);
        colorMap.put("apple", Color.YELLOW);
        colorMap.put("sandwich", Color.YELLOW);
        colorMap.put("orange", Color.YELLOW);
        colorMap.put("broccoli", Color.GREEN);
        colorMap.put("carrot", Color.GREEN);
        colorMap.put("hot dog", Color.YELLOW);
        colorMap.put("pizza", Color.YELLOW);
        colorMap.put("donut", Color.YELLOW);
        colorMap.put("cake", Color.YELLOW);
        colorMap.put("chair", Color.GRAY);
        colorMap.put("sofa", Color.GRAY);
        colorMap.put("pottedplant", Color.GREEN);
        colorMap.put("bed", Color.GRAY);
        colorMap.put("diningtable", Color.GRAY);
        colorMap.put("toilet", Color.GRAY);
        colorMap.put("tvmonitor", Color.GRAY);
        colorMap.put("laptop", Color.GRAY);
        colorMap.put("mouse", Color.GRAY);
        colorMap.put("remote", Color.GRAY);
        colorMap.put("keyboard", Color.GRAY);
        colorMap.put("cell phone", Color.GRAY);
        colorMap.put("microwave", Color.LIGHT_GRAY);
        colorMap.put("oven", Color.LIGHT_GRAY);
        colorMap.put("toaster", Color.LIGHT_GRAY);
        colorMap.put("sink", Color.LIGHT_GRAY);
        colorMap.put("refrigerator", Color.LIGHT_GRAY);
        colorMap.put("book", Color.WHITE);
        colorMap.put("clock", Color.WHITE);
        colorMap.put("vase", Color.WHITE);
        colorMap.put("scissors", Color.WHITE);
        colorMap.put("teddy bear", Color.PINK);
        colorMap.put("hair drier", Color.PINK);
        colorMap.put("toothbrush", Color.PINK);
        return colorMap;
    }
}

       
