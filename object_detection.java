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

public class object_detection {
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
        frame.setSize(400, 300);
        frame.setLayout(new GridLayout(4, 1));

        // Create buttons for live and custom detection
        JButton liveDetectionButton = new JButton("Live Detection");
        JButton customDetectionButton = new JButton("Custom Detection");

        JLabel titleLabel = new JLabel("Choose Detection Mode", SwingConstants.CENTER);
        titleLabel.setFont(new Font("Arial", Font.BOLD, 24));
        frame.add(titleLabel);
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

    private static void drawDetections(Mat frame, List<Mat> result, List<String> classNames, Map<String, Color> colorMap, JLabel objectCountLabel) {
        int objectsDetected = 0;
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

                    String label = classNames.get((int) classIdPoint.x);
                    Color color = colorMap.getOrDefault(label, Color.RED);
                    Imgproc.rectangle(frame, new Point(left, top), new Point(left + width, top + height), new Scalar(color.getRed(), color.getGreen(), color.getBlue()), 2);
                    int[] baseLine = new int[1];
                    Size labelSize = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX, 0.75, 1, baseLine);

                    top = Math.max(top, (int) labelSize.height);
                    Imgproc.putText(frame, label, new Point(left, top), Imgproc.FONT_HERSHEY_SIMPLEX, 0.75, new Scalar(0, 0, 0), 2);
                    objectsDetected++;
                }
            }
        }
        objectCountLabel.setText("Objects Detected: " + objectsDetected);
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

        JLabel objectCountLabel = new JLabel("Objects Detected: 0", SwingConstants.CENTER);
        objectCountLabel.setFont(new Font("Arial", Font.BOLD, 20));
        liveFrame.add(objectCountLabel, BorderLayout.NORTH);

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

        // Create a color map for drawing bounding boxes
        Map<String, Color> colorMap = createColorMap();

        // Start detection loop
        new Thread(() -> {
            Mat frame = new Mat();
            while (capture.read(frame)) {
                Mat blob = Dnn.blobFromImage(frame, 1 / 255.0, new Size(416, 416), new Scalar(0, 0, 0), true, false);
                net.setInput(blob);

                List<Mat> result = new ArrayList<>();
                List<String> outBlobNames = getOutputNames(net);
                net.forward(result, outBlobNames);

                drawDetections(frame, result, classNames, colorMap, objectCountLabel);

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

        // Create a color map for drawing bounding boxes
        Map<String, Color> colorMap = createColorMap();

        int objectsDetected = 0;
        for (Mat level : result) {
            for (int i = 0; i < level.rows(); ++i) {
                Mat row = level.row(i);
                Mat scores = row.colRange(5, level.cols());
                Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                float confidence = (float) mm.maxVal;
                Point classIdPoint = mm.maxLoc;

                if (confidence > 0.5) {
                    int centerX = (int) (row.get(0, 0)[0] * image.cols());
                    int centerY = (int) (row.get(0, 1)[0] * image.rows());
                    int width = (int) (row.get(0, 2)[0] * image.cols());
                    int height = (int) (row.get(0, 3)[0] * image.rows());

                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    String label = classNames.get((int) classIdPoint.x);
                    Color color = colorMap.getOrDefault(label, Color.RED);

                    g2d.setColor(color);
                    g2d.drawRect(left, top, width, height);
                    g2d.setColor(Color.BLACK);
                    g2d.drawString(label, left, top - 10);
                    objectsDetected++;
                }
            }
        }
        g2d.dispose();

        // Display annotated image
        JFrame frame = new JFrame("Detected Objects - Total: " + objectsDetected);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(image.width(), image.height());

        JLabel imageLabel = new JLabel(new ImageIcon(bufImage));
        frame.add(imageLabel);
        frame.setVisible(true);
    }

    private static List<String> loadClassNames(String filePath) {
        List<String> classNames = new ArrayList<>();
        try {
            classNames = Files.readAllLines(Paths.get(filePath));
        } catch (Exception e) {
            e.printStackTrace();
        }
        return classNames;
    }

    private static BufferedImage matToBufferedImage(Mat matrix) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (matrix.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        int bufferSize = matrix.channels() * matrix.cols() * matrix.rows();
        byte[] buffer = new byte[bufferSize];
        matrix.get(0, 0, buffer);
        BufferedImage image = new BufferedImage(matrix.cols(), matrix.rows(), type);
        final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(buffer, 0, targetPixels, 0, buffer.length);
        return image;
    }

    private static List<String> getOutputNames(Net net) {
        List<String> names = new ArrayList<>();
        List<Integer> outLayers = net.getUnconnectedOutLayers().toList();
        List<String> layersNames = net.getLayerNames();
        outLayers.forEach((item) -> names.add(layersNames.get(item - 1)));
        return names;
    }

    private static Map<String, Color> createColorMap() {
        Map<String, Color> colorMap = new HashMap<>();
        colorMap.put("person", Color.RED);
        colorMap.put("bicycle", Color.BLUE);
        colorMap.put("car", Color.GREEN);
        colorMap.put("motorbike", Color.YELLOW);
        colorMap.put("aeroplane", Color.ORANGE);
        colorMap.put("bus", Color.CYAN);
        colorMap.put("train", Color.MAGENTA);
        colorMap.put("truck", Color.PINK);
        colorMap.put("boat", Color.LIGHT_GRAY);
        // Add more classes and colors as needed
        return colorMap;
    }
}
