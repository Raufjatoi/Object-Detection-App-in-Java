package extra;
import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class object_detection_app {
    static {
        // Load the OpenCV library
        System.load("D:\\Downloads\\opencv\\build\\java\\x64\\opencv_java490.dll");
    }

    public static void main(String[] args) {
        // Load OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Create main window
        JFrame frame = new JFrame("Object Detection");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(300, 200);
        frame.setLayout(new GridLayout(3, 1));

        // Create buttons for live and custom detection
        JButton liveDetectionButton = new JButton("Live Detection");
        JButton customDetectionButton = new JButton("Custom Detection");

        frame.add(new JLabel("Choose detection mode:", SwingConstants.CENTER));
        frame.add(liveDetectionButton);
        frame.add(customDetectionButton);

        liveDetectionButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                frame.dispose();
                startLiveDetection();
            }
        });

        customDetectionButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                frame.dispose();
                startCustomDetection();
            }
        });

        frame.setVisible(true);
    }
    private static void drawDetections(Mat frame, List<Mat> result, List<String> classNames) {
    for (Mat level : result) {
        for (int i = 0; i < level.rows(); ++i) {
            Mat row = level.row(i);
            Mat scores = row.colRange(5, level.cols());
            Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
            float confidence = (float) mm.maxVal;
            Point classIdPoint = mm.maxLoc;

            if (confidence > 0.5) {
                int centerX = (int) (row.get(0, 0)[0] * frame.cols());
                int centerY = (int) (row.get(0, 1)[0] * frame.rows());
                int width = (int) (row.get(0, 2)[0] * frame.cols());
                int height = (int) (row.get(0, 3)[0] * frame.rows());

                int left = centerX - width / 2;
                int top = centerY - height / 2;

                Imgproc.rectangle(frame, new Point(left, top), new Point(left + width, top + height), new Scalar(0, 255, 0));
                String label = classNames.get((int) classIdPoint.x);
                int[] baseLine = new int[1];
                Size labelSize = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, 1, baseLine);

                top = Math.max(top, (int) labelSize.height);
                Imgproc.putText(frame, label, new Point(left, top), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 255, 0));
            }
        }
    }
}


    private static void startLiveDetection() {
        // Load the YOLO model
        String folderPath = "D:\\Documents\\GitHub\\java\\";
        String modelConfiguration = folderPath + "yolov4.cfg";
        String modelWeights = folderPath + "yolov4.weights";
        String classNamesFile = folderPath + "coco.names";

        Net net = Dnn.readNetFromDarknet(modelConfiguration, modelWeights);
        List<String> classNames = loadClassNames(classNamesFile);

        if (net.empty() || classNames.isEmpty()) {
            System.err.println("Cannot load network or class names.");
            return;
        }

        // Open a video capture stream
        VideoCapture capture = new VideoCapture(0); // Use default camera

        if (!capture.isOpened()) {
            System.err.println("Cannot open camera.");
            return;
        }

        // Create a window for displaying the video
        JFrame liveFrame = new JFrame("Live Detection");
        liveFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        liveFrame.setSize(800, 600);
        liveFrame.setLayout(new BorderLayout());

        JLabel videoLabel = new JLabel();
        liveFrame.add(videoLabel, BorderLayout.CENTER);

        JButton stopButton = new JButton("Stop Detection");
        liveFrame.add(stopButton, BorderLayout.SOUTH);

        liveFrame.setVisible(true);

        stopButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                capture.release();
                liveFrame.dispose();
            }
        });

        // Start detection loop
        new Thread(() -> {
            Mat frame = new Mat();
            while (capture.read(frame)) {
                Mat blob = Dnn.blobFromImage(frame, 1 / 255.0, new Size(416, 416), new Scalar(0, 0, 0), true, false);
                net.setInput(blob);

                List<Mat> result = new ArrayList<>();
                List<String> outBlobNames = getOutputNames(net);
                net.forward(result, outBlobNames);

                drawDetections(frame, result, classNames);

                BufferedImage bufImage = matToBufferedImage(frame);
                ImageIcon icon = new ImageIcon(bufImage);
                videoLabel.setIcon(icon);

                try {
                    Thread.sleep(30); // Adjust the delay as needed
                } catch (InterruptedException interruptedException) {
                    interruptedException.printStackTrace();
                }
            }
        }).start();
    }

    private static void startCustomDetection() {
        // Ask for the image to detect
        JFileChooser fileChooser = new JFileChooser();
        int returnValue = fileChooser.showOpenDialog(null);
        if (returnValue != JFileChooser.APPROVE_OPTION) {
            System.out.println("No image selected.");
            return;
        }

        String inputImagePath = fileChooser.getSelectedFile().getPath();

        // Define the folder path and other file paths
        String folderPath = "D:\\Documents\\GitHub\\java\\";
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
                        Color color = colorMap.getOrDefault(classNames.get(classId), Color.RED);
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
                            break;
                        }
                    }
                }
            }
        }

        g2d.dispose();

        // Display the annotated image
        ImageIcon imageIcon = new ImageIcon(bufImage);
        JLabel jLabel = new JLabel();
        jLabel.setIcon(imageIcon);

        JFrame jFrame = new JFrame();
        jFrame.setLayout(new FlowLayout());
        jFrame.setSize(bufImage.getWidth(), bufImage.getHeight());
        jFrame.add(jLabel);
        jFrame.setVisible(true);
        jFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // Prompt to save the detected image
        int saveOption = JOptionPane.showConfirmDialog(null, "Do you want to save the detected image?", "Save Image", JOptionPane.YES_NO_OPTION);
        if (saveOption == JOptionPane.YES_OPTION) {
            JFileChooser saveFileChooser = new JFileChooser();
            saveFileChooser.setDialogTitle("Save Detected Image");
            saveFileChooser.setSelectedFile(new File("detected_image.png"));
            int userSelection = saveFileChooser.showSaveDialog(null);
            if (userSelection == JFileChooser.APPROVE_OPTION) {
                File fileToSave = saveFileChooser.getSelectedFile();
                try {
                    ImageIO.write(bufImage, "png", fileToSave);
                    JOptionPane.showMessageDialog(null, "Image saved successfully!");
                } catch (Exception ex) {
                    JOptionPane.showMessageDialog(null, "Failed to save the image.");
                }
            }
        }
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

    private static List<String> loadClassNames(String classNamesFile) {
        try {
            return Files.readAllLines(Paths.get(classNamesFile));
        } catch (Exception e) {
            e.printStackTrace();
            return new ArrayList<>();
        }
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
                // BGR to RGB conversion
                byte b;
                for (int i = 0; i < data.length; i += 3) {
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
        colorMap.put("person", Color.RED);
        colorMap.put("bicycle", Color.GREEN);
        colorMap.put("car", Color.BLUE);
        colorMap.put("motorcycle", Color.CYAN);
        colorMap.put("airplane", Color.MAGENTA);
        colorMap.put("bus", Color.ORANGE);
        colorMap.put("train", Color.PINK);
        colorMap.put("truck", Color.YELLOW);
        colorMap.put("boat", Color.GRAY);
        colorMap.put("traffic light", Color.BLACK);
        // Add more mappings as needed
        return colorMap;
    }

    private static boolean areSimilar(Rect r1, Rect r2, int margin) {
        return Math.abs(r1.x - r2.x) <= margin && Math.abs(r1.y - r2.y) <= margin &&
               Math.abs(r1.width - r2.width) <= margin && Math.abs(r1.height - r2.height) <= margin;
    }
}
