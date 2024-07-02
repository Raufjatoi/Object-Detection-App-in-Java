package extra;
import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import javax.swing.*;
import java.awt.*;
import java.awt.Point;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class test2 {
    static {
        // Load the OpenCV library
        System.load("D:\\Downloads\\opencv\\build\\java\\x64\\opencv_java490.dll");
    }

    public static void main(String[] args) {
        // Load OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Load the YOLOv4 model
        String folderPath = "D:\\Documents\\GitHub\\java\\";
        String modelWeights = folderPath + "yolov4.weights";
        String modelConfiguration = folderPath + "yolov4.cfg";
        Net net = Dnn.readNetFromDarknet(modelConfiguration, modelWeights);

        if (net.empty()) {
            System.err.println("Cannot load network using given weights file: " + modelWeights +
                               " and configuration file: " + modelConfiguration);
            return;
        }

        // Load class names (optional, depending on your model setup)
        String classNamesFile = folderPath + "coco.names";
        List<String> classNames = Utils.loadClassNames(classNamesFile);

        // Open a video capture stream
        VideoCapture capture = new VideoCapture(0); // Use default camera

        if (!capture.isOpened()) {
            System.err.println("Cannot open camera.");
            return;
        }

        // Create a window for displaying the video
        JFrame frame = new JFrame("YOLOv4 Object Detection");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(800, 600);
        frame.setLayout(new BorderLayout());

        JLabel videoLabel = new JLabel();
        frame.add(videoLabel, BorderLayout.CENTER);

        JLabel objectCountLabel = new JLabel("Objects Detected: 0", SwingConstants.CENTER);
        objectCountLabel.setFont(new Font("Arial", Font.BOLD, 20));
        frame.add(objectCountLabel, BorderLayout.NORTH);

        frame.setVisible(true);

        // Start detection loop
        new Thread(() -> {
            Mat frameMat = new Mat();
            while (capture.read(frameMat)) {
                Mat blob = Dnn.blobFromImage(frameMat, 1.0 / 255.0, new Size(416, 416), new Scalar(0), true, false);
                net.setInput(blob);

                List<Mat> result = new ArrayList<>();
                List<String> outBlobNames = Utils.getOutputNames(net);
                net.forward(result, outBlobNames);

                // Process detections and draw bounding boxes
                int objectsDetected = Utils.drawDetections(frameMat, result, classNames, 0.65); // Adjusted confidence threshold

                // Update object count label
                objectCountLabel.setText("Objects Detected: " + objectsDetected);

                // Convert Mat to BufferedImage for display
                BufferedImage bufImage = Utils.matToBufferedImage(frameMat);
                ImageIcon icon = new ImageIcon(bufImage);
                videoLabel.setIcon(icon);

                try {
                    Thread.sleep(30); // Adjust the delay as needed
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }).start();
    }
}

class Utils {
    public static List<String> loadClassNames(String filePath) {
        List<String> classNames = new ArrayList<>();
        try {
            classNames = Files.readAllLines(Paths.get(filePath));
        } catch (IOException e) {
            e.printStackTrace();
        }
        return classNames;
    }

    public static List<String> getOutputNames(Net net) {
        List<String> names = new ArrayList<>();
        List<String> layersNames = net.getLayerNames();
        for (String layer : layersNames) {
            names.add(layer);
        }
        return names;
    }

    public static int drawDetections(Mat frame, List<Mat> result, List<String> classNames, double confidenceThreshold) {
        Map<String, Integer> objectCounts = new HashMap<>();
        int totalObjects = 0;

        for (Mat level : result) {
            for (int i = 0; i < level.rows(); ++i) {
                Mat row = level.row(i);
                Mat scores = row.colRange(5, level.cols());
                Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                float confidence = (float) mm.maxVal;
                org.opencv.core.Point classIdPoint = mm.maxLoc;

                if (confidence > confidenceThreshold) {
                    int centerX = (int) (row.get(0, 0)[0] * frame.cols());
                    int centerY = (int) (row.get(0, 1)[0] * frame.rows());
                    int width = (int) (row.get(0, 2)[0] * frame.cols());
                    int height = (int) (row.get(0, 3)[0] * frame.rows());

                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    String label = classNames.get((int) classIdPoint.x);
                    Imgproc.rectangle(frame, new Point(left, top), new Point(left + width, top + height), new Scalar(0, 255, 0), 2);

                    // Count each detected object
                    objectCounts.put(label, objectCounts.getOrDefault(label, 0) + 1);
                    totalObjects++;
                }
            }
        }

        // Display object counts in console (optional)
        objectCounts.forEach((label, count) -> System.out.println(label + ": " + count));

        return totalObjects;
    }

    public static BufferedImage matToBufferedImage(Mat matrix) {
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
}
