package extra;
import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import javax.swing.*;
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

public class liveobj {
    static {
        // Load the OpenCV library
        System.load("D:\\Downloads\\opencv\\build\\java\\x64\\opencv_java490.dll");
    }

    public static void main(String[] args) {
        // Load OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Define the file paths
        String folderPath = "D:\\Documents\\GitHub\\java\\";
        String modelConfiguration = folderPath + "yolov4.cfg";
        String modelWeights = folderPath + "yolov4.weights";
        String classNamesFile = folderPath + "coco.names";

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

        // Create a window to display the camera feed and detection results
        JFrame frame = new JFrame("Live Object Detection");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(800, 600);

        // Create a panel for drawing the camera feed
        JPanel panel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                if (bufImage != null) {
                    g.drawImage(bufImage, 0, 0, bufImage.getWidth(), bufImage.getHeight(), this);
                }
            }
        };
        frame.add(panel);
        frame.setVisible(true);

        // Open webcam
        VideoCapture camera = new VideoCapture(0); // 0 for the default camera
        if (!camera.isOpened()) {
            System.out.println("Error: Could not open camera.");
            return;
        }

        // Process frames from the webcam
        Mat frameMat = new Mat();
        BufferedImage bufImage = null; // Declare BufferedImage outside the loop

        while (true) {
            // Capture frame from the camera
            if (!camera.read(frameMat)) {
                System.out.println("Error: Cannot read frame.");
                break;
            }

            // Prepare the image for YOLO
            Mat blob = Dnn.blobFromImage(frameMat, 1 / 255.0, new Size(416, 416), new Scalar(0), true, false);

            // Set input to the model
            net.setInput(blob);

            // Run forward pass to get output of the output layers
            List<Mat> result = new ArrayList<>();
            List<String> outBlobNames = getOutputNames(net);
            net.forward(result, outBlobNames);

            // Convert Mat to BufferedImage for drawing and annotation
            bufImage = matToBufferedImage(frameMat);

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
                        int centerX = (int) (row.get(0, 0)[0] * frameMat.cols());
                        int centerY = (int) (row.get(0, 1)[0] * frameMat.rows());
                        int width = (int) (row.get(0, 2)[0] * frameMat.cols());
                        int height = (int) (row.get(0, 3)[0] * frameMat.rows());
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
                            Color color = (classNames.get(classId).equals("person")) ? Color.BLUE : Color.RED;
                            g2d.setColor(color);
                            g2d.drawRect(left, top, width, height);

                            // Annotate object
                            g2d.setColor(Color.BLACK); // Set label color to black
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

            // Update the panel with the new image
            panel.getGraphics().drawImage(bufImage, 0, 0, panel);
        }

        // Release resources
        camera.release();
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

    private static Map<String, Color> createColorMap() {
        Map<String, Color> colorMap = new HashMap<>();
        // Define colors for each class
        colorMap.put("person", Color.BLUE);
        colorMap.put("bicycle", Color.GREEN);
        colorMap.put("car", Color.RED);
        colorMap.put("motorbike", Color.CYAN);
        colorMap.put("aeroplane", Color.MAGENTA);
        colorMap.put("bus", Color.YELLOW);
        colorMap.put("train", Color.ORANGE);
        colorMap.put("truck", Color.PINK);
        colorMap.put("boat", Color.WHITE);
        colorMap.put("traffic light", Color.RED);
        colorMap.put("fire hydrant", Color.RED);
        colorMap.put("stop sign", Color.RED);
        colorMap.put("parking meter", Color.RED);
        colorMap.put("bench", Color.GRAY);
        colorMap.put("bird", Color.YELLOW);
        colorMap.put("cat", Color.YELLOW);
        colorMap.put("dog", Color.YELLOW);
        colorMap.put("horse", Color.YELLOW);
        colorMap.put("sheep", Color.YELLOW);
        colorMap.put("cow", Color.YELLOW);
        colorMap.put("elephant", Color.YELLOW);
        colorMap.put("bear", Color.YELLOW);
        colorMap.put("zebra", Color.YELLOW);
        colorMap.put("giraffe", Color.YELLOW);
        colorMap.put("backpack", Color.LIGHT_GRAY);
        colorMap.put("umbrella", Color.LIGHT_GRAY);
        colorMap.put("handbag", Color.LIGHT_GRAY);
        colorMap.put("tie", Color.CYAN);
        colorMap.put("suitcase", Color.CYAN);
        colorMap.put("frisbee", Color.GREEN);
        colorMap.put("skis", Color.GREEN);
        colorMap.put("snowboard", Color.GREEN);
        colorMap.put("sports ball", Color.GREEN);
        colorMap.put("kite", Color.GREEN);
        colorMap.put("baseball bat", Color.GREEN);
        colorMap.put("baseball glove", Color.GREEN);
        colorMap.put("skateboard", Color.GREEN);
        colorMap.put("surfboard", Color.GREEN);
        colorMap.put("tennis racket", Color.GREEN);
        colorMap.put("bottle", Color.CYAN);
        colorMap.put("wine glass", Color.CYAN);
        colorMap.put("cup", Color.CYAN);
        colorMap.put("fork", Color.CYAN);
        colorMap.put("knife", Color.CYAN);
        colorMap.put("spoon", Color.CYAN);
        colorMap.put("bowl", Color.CYAN);
        colorMap.put("banana", Color.YELLOW);
        colorMap.put("apple", Color.YELLOW);
        colorMap.put("sandwich", Color.YELLOW);
        colorMap.put("orange", Color.YELLOW);
        colorMap.put("broccoli", Color.YELLOW);
        colorMap.put("carrot", Color.YELLOW);
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
        colorMap.put("microwave", Color.GRAY);
        colorMap.put("oven", Color.GRAY);
        colorMap.put("toaster", Color.GRAY);
        colorMap.put("sink", Color.GRAY);
        colorMap.put("refrigerator", Color.GRAY);
        colorMap.put("book", Color.MAGENTA);
        colorMap.put("clock", Color.MAGENTA);
        colorMap.put("vase", Color.MAGENTA);
        colorMap.put("scissors", Color.MAGENTA);
        colorMap.put("teddy bear", Color.MAGENTA);
        colorMap.put("hair drier", Color.MAGENTA);
        colorMap.put("toothbrush", Color.MAGENTA);

        return colorMap;
    }
}
