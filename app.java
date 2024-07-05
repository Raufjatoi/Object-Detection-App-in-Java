import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class app {
    static {
        // Load the OpenCV library
        System.load("D:\\Downloads\\opencv\\build\\java\\x64\\opencv_java490.dll");
    }

    private static JFrame mainFrame;

    public static void main(String[] args) {
        // Load OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Create main window
        mainFrame = new JFrame("Object Detection App");
        mainFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        mainFrame.setExtendedState(JFrame.MAXIMIZED_BOTH); // Fullscreen window
        mainFrame.setLayout(new BorderLayout());
        mainFrame.getRootPane().setBorder(BorderFactory.createMatteBorder(5, 5, 5, 5, Color.BLACK)); // Border for the window

        // Header
        JPanel headerPanel = new JPanel();
        headerPanel.setLayout(new BorderLayout());
        headerPanel.setBackground(new Color(0, 102, 204));
        headerPanel.setBorder(BorderFactory.createMatteBorder(0, 0, 5, 0, Color.BLACK)); // Border for the header

        // Cool emoji or icon (replace with your preferred image)
        ImageIcon coolIcon = new ImageIcon("cool_detection_icon.png");
        JLabel coolLabel = new JLabel(coolIcon);
        coolLabel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        headerPanel.add(coolLabel, BorderLayout.WEST);

        JLabel titleLabel = new JLabel("Object Detection App", SwingConstants.CENTER);
        titleLabel.setFont(new Font("Arial", Font.BOLD, 32));
        titleLabel.setForeground(Color.WHITE);
        titleLabel.setBorder(BorderFactory.createEmptyBorder(20, 0, 20, 0));
        headerPanel.add(titleLabel, BorderLayout.CENTER);

        // Profile pictures with hover effect
        JPanel profilePanel = new JPanel();
        profilePanel.setBackground(new Color(0, 102, 204));
        profilePanel.setLayout(new FlowLayout(FlowLayout.RIGHT));
        profilePanel.add(createProfilePic("rp.PNG", "Live Detection", e -> startLiveDetection(mainFrame)));
        profilePanel.add(createProfilePic("umar.jpeg", "Custom Detection", e -> startCustomDetection(mainFrame)));
        profilePanel.add(createProfilePic("ahsan.jpeg", "About", e -> showAbout()));

        headerPanel.add(profilePanel, BorderLayout.EAST);

        // Center panel for buttons
        JPanel centerPanel = new JPanel();
        centerPanel.setBackground(Color.DARK_GRAY);
        centerPanel.setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(10, 10, 10, 10);

        JButton liveDetectionButton = createStyledButton("Live Detection", e -> startLiveDetection(mainFrame));
        JButton customDetectionButton = createStyledButton("Custom Detection", e -> startCustomDetection(mainFrame));

        gbc.gridx = 0;
        gbc.gridy = 0;
        centerPanel.add(liveDetectionButton, gbc);
        gbc.gridy = 1;
        centerPanel.add(customDetectionButton, gbc);

        // Footer
        JPanel footerPanel = new JPanel();
        footerPanel.setBackground(new Color(0, 102, 204));
        footerPanel.setLayout(new BorderLayout());
        footerPanel.setBorder(BorderFactory.createMatteBorder(5, 0, 0, 0, Color.BLACK)); // Border for the footer
        JLabel footerLabel = new JLabel("By Rauf, Ahsan, and Umar", SwingConstants.CENTER);
        footerLabel.setFont(new Font("Arial", Font.PLAIN, 16));
        footerLabel.setForeground(Color.WHITE);

        JButton aboutButton = new JButton("About");
        aboutButton.setFont(new Font("Arial", Font.PLAIN, 16));
        aboutButton.addActionListener(e -> showAbout());

        footerPanel.add(footerLabel, BorderLayout.CENTER);
        footerPanel.add(aboutButton, BorderLayout.EAST);

        // Add panels to main frame
        mainFrame.add(headerPanel, BorderLayout.NORTH);
        mainFrame.add(centerPanel, BorderLayout.CENTER);
        mainFrame.add(footerPanel, BorderLayout.SOUTH);

        mainFrame.setVisible(true);
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
            ImageIcon icon = new ImageIcon(scaledImage);

            JLabel label = new JLabel(icon);
            label.setToolTipText(toolTipText);
            label.setBorder(BorderFactory.createLineBorder(Color.BLACK, 2));
            label.addMouseListener(new MouseAdapter() {
                @Override
                public void mouseClicked(MouseEvent e) {
                    action.actionPerformed(null);
                }

                @Override
                public void mouseEntered(MouseEvent e) {
                    label.setBorder(BorderFactory.createLineBorder(Color.WHITE, 2));
                    label.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
                }

                @Override
                public void mouseExited(MouseEvent e) {
                    label.setBorder(BorderFactory.createLineBorder(Color.BLACK, 2));
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
        Map<String, Rectangle> drawnObjects = new HashMap<>(); // Track drawn objects

        for (Mat level : result) {
            for (int i = 0; i < level.rows(); ++i) {
                Mat row = level.row(i);
                Mat scores = row.colRange(5, level.cols());
                Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                float confidence = (float) mm.maxVal;
                Point classIdPoint = mm.maxLoc;

                if (confidence > 0.65) {
                    int centerX = (int) (row.get(0, 0)[0] * frame.cols());
                    int centerY = (int) (row.get(0, 1)[0] * frame.rows());
                    int width = (int) (row.get(0, 2)[0] * frame.cols());
                    int height = (int) (row.get(0, 3)[0] * frame.rows());

                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    int classId = (int) classIdPoint.x;
                    String label = classNames.get(classId);
                    Color color = colorMap.get(label);

                    Rectangle objectRect = new Rectangle(left, top, width, height);
                    if (!isOverlapping(objectRect, drawnObjects)) {
                        drawnObjects.put(label, objectRect);

                        Imgproc.rectangle(frame, new Point(left, top), new Point(left + width, top + height), new Scalar(color.getRed(), color.getGreen(), color.getBlue()), 3);
                        Imgproc.putText(frame, label, new Point(left, top - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 1, new Scalar(color.getRed(), color.getGreen(), color.getBlue()), 3);
                        objectsDetected++;
                    }
                }
            }
        }
        objectCountLabel.setText("Objects Detected: " + objectsDetected);
    }

    private static boolean isOverlapping(Rectangle rect, Map<String, Rectangle> drawnObjects) {
        for (Rectangle existingRect : drawnObjects.values()) {
            if (rect.intersects(existingRect)) {
                return true;
            }
        }
        return false;
    }

    private static void startLiveDetection(JFrame mainFrame) {
        JFrame liveFrame = new JFrame("Live Detection");
        liveFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        liveFrame.setSize(800, 600);
        liveFrame.setLayout(new BorderLayout());

        JLabel imageLabel = new JLabel();
        liveFrame.add(imageLabel, BorderLayout.CENTER);

        JButton backButton = new JButton("Back");
        backButton.addActionListener(e -> liveFrame.dispose());
        liveFrame.add(backButton, BorderLayout.SOUTH);

        VideoCapture capture = new VideoCapture(0);
        if (!capture.isOpened()) {
            JOptionPane.showMessageDialog(mainFrame, "Error: Unable to open camera.");
            return;
        }

        JLabel objectCountLabel = new JLabel("Objects Detected: 0", SwingConstants.CENTER);
        objectCountLabel.setFont(new Font("Arial", Font.BOLD, 16));
        objectCountLabel.setForeground(Color.RED);
        liveFrame.add(objectCountLabel, BorderLayout.NORTH);

        Thread liveThread = new Thread(() -> {
            Mat frame = new Mat();
            Net net = Dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights");
            net.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV);
            net.setPreferableTarget(Dnn.DNN_TARGET_CPU);

            List<String> classNames = getClassNames("coco.names");
            Map<String, Color> colorMap = getColorMap(classNames);

            while (capture.isOpened()) {
                capture.read(frame);
                if (!frame.empty()) {
                    List<Mat> result = new ArrayList<>();
                    Mat blob = Dnn.blobFromImage(frame, 1 / 255.0, new Size(416, 416), new Scalar(0), true, false);
                    net.setInput(blob);
                    List<String> outBlobNames = net.getUnconnectedOutLayersNames();
                    net.forward(result, outBlobNames);

                    drawDetections(frame, result, classNames, colorMap, objectCountLabel);

                    BufferedImage image = matToBufferedImage(frame);
                    if (image != null) {
                        SwingUtilities.invokeLater(() -> imageLabel.setIcon(new ImageIcon(image)));
                    }
                }
            }
        });

        liveThread.start();
        liveFrame.setVisible(true);
    }

    private static void startCustomDetection(JFrame mainFrame) {
        JFrame customFrame = new JFrame("Custom Detection");
        customFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        customFrame.setSize(800, 600);
        customFrame.setLayout(new BorderLayout());

        JLabel imageLabel = new JLabel();
        customFrame.add(imageLabel, BorderLayout.CENTER);

        JButton backButton = new JButton("Back");
        backButton.addActionListener(e -> customFrame.dispose());
        customFrame.add(backButton, BorderLayout.SOUTH);

        JButton saveButton = new JButton("Save Image");
        customFrame.add(saveButton, BorderLayout.NORTH);

        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);

        int returnValue = fileChooser.showOpenDialog(mainFrame);
        if (returnValue == JFileChooser.APPROVE_OPTION) {
            File selectedFile = fileChooser.getSelectedFile();
            Mat image = Imgcodecs.imread(selectedFile.getAbsolutePath());

            if (image.empty()) {
                JOptionPane.showMessageDialog(mainFrame, "Error: Unable to read image.");
                return;
            }

            Net net = Dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights");
            net.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV);
            net.setPreferableTarget(Dnn.DNN_TARGET_CPU);

            List<String> classNames = getClassNames("coco.names");
            Map<String, Color> colorMap = getColorMap(classNames);

            Mat blob = Dnn.blobFromImage(image, 1 / 255.0, new Size(416, 416), new Scalar(0), true, false);
            net.setInput(blob);
            List<String> outBlobNames = net.getUnconnectedOutLayersNames();
            List<Mat> result = new ArrayList<>();
            net.forward(result, outBlobNames);

            JLabel objectCountLabel = new JLabel("Objects Detected: 0", SwingConstants.CENTER);
            objectCountLabel.setFont(new Font("Arial", Font.BOLD, 16));
            objectCountLabel.setForeground(Color.RED);
            customFrame.add(objectCountLabel, BorderLayout.NORTH);

            drawDetections(image, result, classNames, colorMap, objectCountLabel);

            BufferedImage outputImage = matToBufferedImage(image);
            if (outputImage != null) {
                imageLabel.setIcon(new ImageIcon(outputImage));

                saveButton.addActionListener(e -> {
                    JFileChooser saveChooser = new JFileChooser();
                    saveChooser.setDialogTitle("Save Image");
                    int userSelection = saveChooser.showSaveDialog(customFrame);

                    if (userSelection == JFileChooser.APPROVE_OPTION) {
                        File fileToSave = saveChooser.getSelectedFile();
                        try {
                            ImageIO.write(outputImage, "png", fileToSave);
                            JOptionPane.showMessageDialog(customFrame, "Image saved successfully!");
                        } catch (Exception ex) {
                            ex.printStackTrace();
                            JOptionPane.showMessageDialog(customFrame, "Error: Unable to save image.");
                        }
                    }
                });
            }
        }
        customFrame.setVisible(true);
    }

    private static List<String> getClassNames(String filePath) {
        List<String> classNames = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                classNames.add(line.trim());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return classNames;
    }

    private static Map<String, Color> getColorMap(List<String> classNames) {
        Map<String, Color> colorMap = new HashMap<>();
        Random random = new Random();
        for (String className : classNames) {
            colorMap.put(className, new Color(random.nextInt(256), random.nextInt(256), random.nextInt(256)));
        }
        return colorMap;
    }

    private static BufferedImage matToBufferedImage(Mat mat) {
        if (mat == null || mat.empty()) {
            return null;
        }
        int type = mat.channels() > 1 ? BufferedImage.TYPE_3BYTE_BGR : BufferedImage.TYPE_BYTE_GRAY;
        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        mat.get(0, 0, ((DataBufferByte) image.getRaster().getDataBuffer()).getData());
        return image;
    }

    private static void showAbout() {
        String message = "Object Detection App\nVersion 1.0\n\n"
            + "Developed by:\n"
            + "Rauf: Model development\n"
            + "Ahsan: GUI development\n"
            + "Umar: UI design\n\n"
            + "Usage Instructions:\n"
            + " - Click 'Live Detection' for real-time object detection.\n"
            + " - Click 'Custom Detection' to detect objects in a selected image.";
            
        JOptionPane.showMessageDialog(mainFrame, message, "Object Detection App", JOptionPane.INFORMATION_MESSAGE);
    }
    
}
