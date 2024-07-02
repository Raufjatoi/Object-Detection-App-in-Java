package extra;
import java.util.List;
import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.utils.Converters;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.util.*;
import javax.imageio.ImageIO;

public class custom {

    static {
        // Load the OpenCV library
        System.load("D:\\Downloads\\opencv\\build\\java\\x64\\opencv_java490.dll");
    }

    public static void main(String[] args) {
        // Load OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Example usage
        String imagePath = "path_to_your_input_image.jpg";
        String detectedImagePath = performObjectDetection(imagePath);
        System.out.println("Detected image saved at: " + detectedImagePath);
    }

    public static String performObjectDetection(String inputImagePath) {
        // Load the input image
        Mat image = Imgcodecs.imread(inputImagePath);

        // Paths to YOLOv4 model files and class names
        String folderPath = "D:\\Documents\\GitHub\\java\\";
        String modelConfiguration = folderPath + "models/yolov4.cfg";
        String modelWeights = folderPath + "models/yolov4.weights";
        String classNamesFile = folderPath + "models/coco.names";

        // Load YOLOv4 model
        Net net = Dnn.readNetFromDarknet(modelConfiguration, modelWeights);
        if (net.empty()) {
            System.err.println("Cannot load network using given configuration and weights files.");
            return null;
        }

        // Load class names
        List<String> classNames = loadClassNames(classNamesFile);
        if (classNames.isEmpty()) {
            System.err.println("Cannot load class names.");
            return null;
        }

        // Perform object detection
        Mat blob = Dnn.blobFromImage(image, 1 / 255.0, new Size(416, 416), new Scalar(0, 0, 0), true, false);
        net.setInput(blob);

        List<Mat> result = new ArrayList<>();
        List<String> outBlobNames = getOutputNames(net);
        net.forward(result, outBlobNames);

        // Annotate detected objects
        annotateObjects(image, result, classNames);

        // Save the annotated image
        String outputImagePath = folderPath + "detected.png";
        Imgcodecs.imwrite(outputImagePath, image);

        return outputImagePath;
    }

    private static void annotateObjects(Mat image, List<Mat> result, List<String> classNames) {
        Graphics2D g2d = matToGraphics(image);
        g2d.setStroke(new BasicStroke(2));
        Font font = new Font("Arial", Font.BOLD, 20);
        g2d.setFont(font);

        Map<String, Color> colorMap = createColorMap();

        float confThreshold = 0.5f;
        int maxObjects = 20;
        int objectsDetected = 0;
        Map<String, Rect> objectMap = new HashMap<>();

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

                    boolean foundSimilar = false;
                    for (Map.Entry<String, Rect> entry : objectMap.entrySet()) {
                        Rect existingRect = entry.getValue();
                        if (areSimilar(existingRect, new Rect(left, top, width, height), 20)) {
                            foundSimilar = true;
                            break;
                        }
                    }

                    if (!foundSimilar) {
                        Color color = colorMap.getOrDefault(classNames.get(classId), Color.RED);
                        g2d.setColor(color);
                        g2d.drawRect(left, top, width, height);

                        g2d.setColor(Color.BLACK);
                        String label = classNames.get(classId) + ": " + new DecimalFormat("#.##").format(confidence);
                        int textX = left;
                        int textY = top - 10;
                        g2d.drawString(label, textX, textY);

                        objectMap.put(label, new Rect(left, top, width, height));
                        objectsDetected++;

                        if (objectsDetected >= maxObjects) {
                            break;
                        }
                    }
                }
            }
            if (objectsDetected >= maxObjects) {
                break;
            }
        }

        g2d.setColor(Color.WHITE);
        g2d.setFont(new Font("Arial", Font.BOLD, 24));
        String countLabel = "Objects Detected: " + objectsDetected;
        g2d.drawString(countLabel, 20, 30);

        g2d.dispose();
    }

    private static Graphics2D matToGraphics(Mat matrix) {
        BufferedImage bufImage = matToBufferedImage(matrix);
        return bufImage.createGraphics();
    }

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
        } catch (IOException e) {
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

    private static Map<String, Color> createColorMap() {
        Map<String, Color> colorMap = new HashMap<>();
        colorMap.put("person", Color.RED);
        colorMap.put("bicycle", Color.BLUE);
        colorMap.put("car", Color.GREEN);
        colorMap.put("motorbike", Color.MAGENTA);
        colorMap.put("aeroplane", Color.CYAN);
        colorMap.put("bus", Color.ORANGE);
        colorMap.put("train", Color.PINK);
        colorMap.put("truck", Color.YELLOW);
        colorMap.put("boat", Color.LIGHT_GRAY);
        colorMap.put("traffic light", Color.DARK_GRAY);
        return colorMap;
    }
}
