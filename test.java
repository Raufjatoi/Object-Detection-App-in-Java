import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.Point;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ObjectDetectionApp {
    static {
        // Load the OpenCV library
        System.load("D:\\Downloads\\opencv\\build\\java\\x64\\opencv_java490.dll");
    }

    public static void main(String[] args) {
        // Load OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Create main window
        JFrame frame = new JFrame("Object Detection App");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setExtendedState(JFrame.MAXIMIZED_BOTH); // Fullscreen window
        frame.setLayout(new BorderLayout());

        // Header
        JPanel headerPanel = new JPanel();
        headerPanel.setLayout(new BorderLayout());
        headerPanel.setBackground(Color.DARK_GRAY);
        JLabel titleLabel = new JLabel("Object Detection App", SwingConstants.CENTER);
        titleLabel.setFont(new Font("Arial", Font.BOLD, 32));
        titleLabel.setForeground(Color.WHITE);

        // Profile pictures
        JPanel profilePanel = new JPanel();
        profilePanel.setBackground(Color.DARK_GRAY);
        profilePanel.setLayout(new FlowLayout(FlowLayout.RIGHT));
        profilePanel.add(createProfilePic("rp.PNG"));
        profilePanel.add(createProfilePic("umar.jpeg"));
        profilePanel.add(createProfilePic("ahsan.jpeg"));

        headerPanel.add(titleLabel, BorderLayout.CENTER);
        headerPanel.add(profilePanel, BorderLayout.EAST);

        // Center panel for buttons
        JPanel centerPanel = new JPanel();
        centerPanel.setBackground(Color.DARK_GRAY);
        centerPanel.setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(10, 10, 10, 10);

        JButton liveDetectionButton = new JButton("Live Detection");
        liveDetectionButton.setFont(new Font("Arial", Font.BOLD, 24));
        JButton customDetectionButton = new JButton("Custom Detection");
        customDetectionButton.setFont(new Font("Arial", Font.BOLD, 24));

        gbc.gridx = 0;
        gbc.gridy = 0;
        centerPanel.add(liveDetectionButton, gbc);
        gbc.gridy = 1;
        centerPanel.add(customDetectionButton, gbc);

        // Footer
        JPanel footerPanel = new JPanel();
        footerPanel.setBackground(Color.DARK_GRAY);
        JLabel footerLabel = new JLabel("By Rauf, Ahsan, and Umar", SwingConstants.CENTER);
        footerLabel.setFont(new Font("Arial", Font.PLAIN, 16));
        footerLabel.setForeground(Color.WHITE);

        JButton aboutButton = new JButton("About");
        aboutButton.setFont(new Font("Arial", Font.PLAIN, 16));
        aboutButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                try {
                    Desktop.getDesktop().browse(new File("about.html").toURI());
                } catch (Exception ex) {
                    ex.printStackTrace();
                }
            }
        });

        footerPanel.setLayout(new BorderLayout());
        footerPanel.add(footerLabel, BorderLayout.CENTER);
        footerPanel.add(aboutButton, BorderLayout.EAST);

        // Add panels to frame
        frame.add(headerPanel, BorderLayout.NORTH);
        frame.add(centerPanel, BorderLayout.CENTER);
        frame.add(footerPanel, BorderLayout.SOUTH);

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

    private static JLabel createProfilePic(String filePath) {
        try {
            BufferedImage profilePic = ImageIO.read(new File(filePath));
            Image scaledImage = profilePic.getScaledInstance(70, 70, Image.SCALE_SMOOTH);
            return new JLabel(new ImageIcon(scaledImage));
        } catch (IOException e) {
            e.printStackTrace();
            return new JLabel("Profile");
        }
    }

    private static void drawDetections(Mat frame, List<Mat> result, List<String> classNames, Map<String, Color> colorMap, JLabel objectCountLabel) {
        int objectsDetected = 0;
        for (Mat level : result) {
            for (int i = 0; i < level.rows(); ++i) {
                Mat row = level.row(i);
                Mat scores = row.colRange(5, level.cols());
                Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                float confidence = (float) mm.maxVal;
                org.opencv.core.Point classIdPoint = mm.maxLoc;

                if (confidence > 0.65) {  // Confidence threshold 0.65
                    int centerX = (int) (row.get(0, 0)[0] * frame.cols());
                    int centerY = (int) (row.get(0, 1)[0] * frame.rows());
                    int width = (int) (row.get(0, 2)[0] * frame.cols());
                    int height = (int) (row.get(0, 3)[0] * frame.rows());

                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    String label = classNames.get((int) classIdPoint.x);
                    Color color = colorMap.getOrDefault(label, Color.RED);
                    Imgproc.rectangle(frame, new Point(left, top), new Point(left + width, top + height), new Scalar(color.getBlue(), color.getGreen(), color.getRed()), 2);
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

        JPanel buttonPanel = new JPanel();
        buttonPanel.setLayout(new FlowLayout());
        buttonPanel.setBackground(Color.DARK_GRAY);

        JButton stopButton = new JButton("Stop Detection");
        JButton backButton = new JButton("Back");
        buttonPanel.add(backButton);
        buttonPanel.add(stopButton);

        liveFrame.add(buttonPanel, BorderLayout.SOUTH);

        liveFrame.setVisible(true);

        backButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                capture.release();
                liveFrame.dispose();
                main(null);
            }
        });

        stopButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                capture.release();
                liveFrame.dispose();
                main(null);
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

                ImageIcon icon = new ImageIcon(matToBufferedImage(frame));
                videoLabel.setIcon(icon);
                liveFrame.repaint();
            }
            capture.release();
        }).start();
    }

    private static void startCustomDetection() {
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

        // Select an image file for custom detection
        JFileChooser fileChooser = new JFileChooser();
        int result = fileChooser.showOpenDialog(null);

        if (result == JFileChooser.APPROVE_OPTION) {
            File selectedFile = fileChooser.getSelectedFile();
            String filePath = selectedFile.getAbsolutePath();
            Mat frame = Imgcodecs.imread(filePath);

            if (frame.empty()) {
                System.err.println("Cannot read image file.");
                return;
            }

            // Detect objects in the image
            Mat blob = Dnn.blobFromImage(frame, 1 / 255.0, new Size(416, 416), new Scalar(0, 0, 0), true, false);
            net.setInput(blob);
            List<Mat> resultMat = new ArrayList<>();
            List<String> outBlobNames = getOutputNames(net);
            net.forward(resultMat, outBlobNames);

            // Create a window to display the image with detections
            JFrame customFrame = new JFrame("Custom Detection");
            customFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            customFrame.setSize(800, 600);
            customFrame.setLayout(new BorderLayout());

            JLabel imageLabel = new JLabel();
            ImageIcon icon = new ImageIcon(matToBufferedImage(frame));
            imageLabel.setIcon(icon);
            customFrame.add(imageLabel, BorderLayout.CENTER);

            JLabel objectCountLabel = new JLabel("Objects Detected: 0", SwingConstants.CENTER);
            objectCountLabel.setFont(new Font("Arial", Font.BOLD, 20));
            customFrame.add(objectCountLabel, BorderLayout.NORTH);

            JPanel buttonPanel = new JPanel();
            buttonPanel.setLayout(new FlowLayout());
            buttonPanel.setBackground(Color.DARK_GRAY);

            JButton backButton = new JButton("Back");
            buttonPanel.add(backButton);

            customFrame.add(buttonPanel, BorderLayout.SOUTH);
            customFrame.setVisible(true);

            // Create a color map for drawing bounding boxes
            Map<String, Color> colorMap = createColorMap();

            // Draw detections on the image
            drawDetections(frame, resultMat, classNames, colorMap, objectCountLabel);

            backButton.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    customFrame.dispose();
                    main(null);
                }
            });
        }
    }

    private static List<String> loadClassNames(String fileName) {
        List<String> classNames = new ArrayList<>();
        try {
            classNames = Files.readAllLines(Paths.get(fileName));
        } catch (IOException e) {
            e.printStackTrace();
        }
        return classNames;
    }

    private static List<String> getOutputNames(Net net) {
        List<String> names = new ArrayList<>();
        List<Integer> outLayers = net.getUnconnectedOutLayers().toList();
        List<String> layersNames = net.getLayerNames();

        outLayers.forEach((item) -> names.add(layersNames.get(item - 1)));
        return names;
    }

    private static BufferedImage matToBufferedImage(Mat frame) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (frame.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        BufferedImage image = new BufferedImage(frame.width(), frame.height(), type);
        frame.get(0, 0, ((DataBufferByte) image.getRaster().getDataBuffer()).getData());
        return image;
    }

    private static Map<String, Color> createColorMap() {
        Map<String, Color> colorMap = new HashMap<>();
        colorMap.put("person", Color.RED);
        colorMap.put("car", Color.GREEN);
        colorMap.put("bicycle", Color.BLUE);
        colorMap.put("motorbike", Color.CYAN);
        colorMap.put("bus", Color.MAGENTA);
        colorMap.put("truck", Color.YELLOW);
        return colorMap;
    }
}
