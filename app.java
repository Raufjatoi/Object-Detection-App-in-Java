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
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class app {
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
        headerPanel.setBackground(new Color(0, 102, 204));
        JLabel titleLabel = new JLabel("Object Detection App", SwingConstants.CENTER);
        titleLabel.setFont(new Font("Arial", Font.BOLD, 32));
        titleLabel.setForeground(Color.WHITE);
        titleLabel.setBorder(BorderFactory.createEmptyBorder(20, 0, 20, 0));

        // Profile pictures with hover effect
        JPanel profilePanel = new JPanel();
        profilePanel.setBackground(new Color(0, 102, 204));
        profilePanel.setLayout(new FlowLayout(FlowLayout.RIGHT));
        profilePanel.add(createProfilePic("rp.PNG", "Live Detection", e -> startLiveDetection(frame)));
        profilePanel.add(createProfilePic("umar.jpeg", "Custom Detection", e -> startCustomDetection(frame)));
        profilePanel.add(createProfilePic("ahsan.jpeg", "About", e -> showAbout()));

        headerPanel.add(titleLabel, BorderLayout.CENTER);
        headerPanel.add(profilePanel, BorderLayout.EAST);

        // Center panel for buttons
        JPanel centerPanel = new JPanel();
        centerPanel.setBackground(Color.DARK_GRAY);
        centerPanel.setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(10, 10, 10, 10);

        JButton liveDetectionButton = createStyledButton("Live Detection", e -> startLiveDetection(frame));
        JButton customDetectionButton = createStyledButton("Custom Detection", e -> startCustomDetection(frame));

        gbc.gridx = 0;
        gbc.gridy = 0;
        centerPanel.add(liveDetectionButton, gbc);
        gbc.gridy = 1;
        centerPanel.add(customDetectionButton, gbc);

        // Footer
        JPanel footerPanel = new JPanel();
        footerPanel.setBackground(new Color(0, 102, 204));
        footerPanel.setLayout(new BorderLayout());
        footerPanel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        JLabel footerLabel = new JLabel("By Rauf, Ahsan, and Umar", SwingConstants.CENTER);
        footerLabel.setFont(new Font("Arial", Font.PLAIN, 16));
        footerLabel.setForeground(Color.WHITE);

        JButton aboutButton = new JButton("About");
        aboutButton.setFont(new Font("Arial", Font.PLAIN, 16));
        aboutButton.addActionListener(e -> showAbout());

        footerPanel.add(footerLabel, BorderLayout.CENTER);
        footerPanel.add(aboutButton, BorderLayout.EAST);

        // Add panels to frame
        frame.add(headerPanel, BorderLayout.NORTH);
        frame.add(centerPanel, BorderLayout.CENTER);
        frame.add(footerPanel, BorderLayout.SOUTH);

        frame.setVisible(true);
    }

    private static JButton createStyledButton(String text, ActionListener action) {
        JButton button = new JButton(text);
        button.setFont(new Font("Arial", Font.BOLD, 24));
        button.setForeground(Color.WHITE);
        button.setBackground(new Color(51, 153, 255));
        button.setFocusPainted(false);
        button.setBorderPainted(false);
        button.setOpaque(true);
        button.addActionListener(action);
        button.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseEntered(MouseEvent e) {
                button.setBackground(new Color(0, 102, 204));
            }

            @Override
            public void mouseExited(MouseEvent e) {
                button.setBackground(new Color(51, 153, 255));
            }
        });
        return button;
    }

    private static JLabel createProfilePic(String filePath, String toolTipText, ActionListener action) {
        try {
            BufferedImage profilePic = ImageIO.read(new File(filePath));
            Image scaledImage = profilePic.getScaledInstance(70, 70, Image.SCALE_SMOOTH);
            JLabel label = new JLabel(new ImageIcon(scaledImage));
            label.setToolTipText(toolTipText);
            label.addMouseListener(new MouseAdapter() {
                @Override
                public void mouseClicked(MouseEvent e) {
                    action.actionPerformed(null);
                }

                @Override
                public void mouseEntered(MouseEvent e) {
                    label.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
                }

                @Override
                public void mouseExited(MouseEvent e) {
                    label.setCursor(Cursor.getDefaultCursor());
                }
            });
            return label;
        } catch (Exception e) {
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
                Point classIdPoint = mm.maxLoc;

                if (confidence > 0.65) {  // Confidence threshold 0.65
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

    private static void startLiveDetection(JFrame frame) {
        frame.dispose();

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
        objectCountLabel.setFont(new Font("Arial", Font.BOLD, 24));
        objectCountLabel.setForeground(Color.WHITE);
        liveFrame.add(objectCountLabel, BorderLayout.SOUTH);

        liveFrame.setVisible(true);

        Mat frameMatrix = new Mat();

        // Object detection loop
        while (liveFrame.isVisible() && capture.read(frameMatrix)) {
            // Preprocess the frame
            Mat blob = Dnn.blobFromImage(frameMatrix, 1 / 255.0, new Size(416, 416), new Scalar(0), true, false);
            net.setInput(blob);

            // Perform forward pass
            List<Mat> result = new ArrayList<>();
            List<String> outBlobNames = getOutputNames(net);
            net.forward(result, outBlobNames);

            // Draw detections on the frame
            drawDetections(frameMatrix, result, classNames, createColorMap(classNames), objectCountLabel);

            // Display the frame
            ImageIcon imageIcon = new ImageIcon(convertMatToImage(frameMatrix));
            videoLabel.setIcon(imageIcon);

            try {
                Thread.sleep(33); // Delay for ~30 FPS
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        // Release resources
        capture.release();
    }

    private static void startCustomDetection(JFrame frame) {
        frame.dispose();

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

        // Open a file chooser to select an image
        JFileChooser fileChooser = new JFileChooser();
        int result = fileChooser.showOpenDialog(null);

        if (result != JFileChooser.APPROVE_OPTION) {
            return;
        }

        File selectedFile = fileChooser.getSelectedFile();

        // Load the selected image
        Mat frameMatrix = Imgcodecs.imread(selectedFile.getAbsolutePath());

        if (frameMatrix.empty()) {
            System.err.println("Cannot read the image file.");
            return;
        }

        // Preprocess the frame
        Mat blob = Dnn.blobFromImage(frameMatrix, 1 / 255.0, new Size(416, 416), new Scalar(0), true, false);
        net.setInput(blob);

        // Perform forward pass
        List<Mat> output = new ArrayList<>();
        List<String> outBlobNames = getOutputNames(net);
        net.forward(output, outBlobNames);

        // Draw detections on the frame
        drawDetections(frameMatrix, output, classNames, createColorMap(classNames), new JLabel());

        // Display the result
        JFrame resultFrame = new JFrame("Custom Detection");
        resultFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        resultFrame.setSize(800, 600);
        resultFrame.setLayout(new BorderLayout());

        JLabel imageLabel = new JLabel();
        imageLabel.setIcon(new ImageIcon(convertMatToImage(frameMatrix)));
        resultFrame.add(new JScrollPane(imageLabel), BorderLayout.CENTER);

        JLabel objectCountLabel = new JLabel("Objects Detected: 0", SwingConstants.CENTER);
        objectCountLabel.setFont(new Font("Arial", Font.BOLD, 24));
        objectCountLabel.setForeground(Color.WHITE);
        resultFrame.add(objectCountLabel, BorderLayout.SOUTH);

        resultFrame.setVisible(true);
    }

    private static List<String> getOutputNames(Net net) {
        List<String> names = new ArrayList<>();
        List<Integer> outLayers = net.getUnconnectedOutLayers().toList();
        List<String> layersNames = net.getLayerNames();

        for (Integer item : outLayers) {
            names.add(layersNames.get(item - 1));
        }
        return names;
    }

    private static List<String> loadClassNames(String classNamesFile) {
        List<String> classNames = new ArrayList<>();
        try {
            classNames = Files.readAllLines(Paths.get(classNamesFile));
        } catch (Exception e) {
            e.printStackTrace();
        }
        return classNames;
    }

    private static BufferedImage convertMatToImage(Mat mat) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (mat.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        mat.get(0, 0, ((DataBufferByte) image.getRaster().getDataBuffer()).getData());
        return image;
    }

    private static Map<String, Color> createColorMap(List<String> classNames) {
        Map<String, Color> colorMap = new HashMap<>();
        for (String className : classNames) {
            colorMap.put(className, Color.getHSBColor((float) Math.random(), 0.9f, 0.9f));
        }
        return colorMap;
    }

    private static void showAbout() {
        JOptionPane.showMessageDialog(null, "Object Detection App\nBy Rauf, Ahsan, and Umar", "About", JOptionPane.INFORMATION_MESSAGE);
    }
}
