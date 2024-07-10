// importing libs 

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
import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

// main class 
public class test {
    static {
        // loading the OpenCV library
        System.load("D:\\Downloads\\opencv\\build\\java\\x64\\opencv_java490.dll");
    }

    public static JFrame mainFrame;

    public static void main(String[] args) {
        // loading OpenCV library again
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // creating main window
        mainFrame = new JFrame("Object Detection App in Java");
        mainFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        mainFrame.setExtendedState(JFrame.MAXIMIZED_BOTH); // fullscreen window
        mainFrame.setLayout(new BorderLayout());
        mainFrame.getRootPane().setBorder(BorderFactory.createMatteBorder(5, 5, 5, 5, Color.BLACK)); // border for the window

        // header
        JPanel headerPanel = new JPanel();
        headerPanel.setLayout(new BorderLayout());
        headerPanel.setBackground(new Color(0, 102, 204));
        headerPanel.setBorder(BorderFactory.createMatteBorder(0, 0, 5, 0, Color.BLACK)); // Border for the header

        ImageIcon coolIcon = new ImageIcon("col.jpg");
        JLabel coolLabel = new JLabel(coolIcon);
        coolLabel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        headerPanel.add(coolLabel, BorderLayout.WEST);

        JLabel titleLabel = new JLabel("Object Detection App in Java", SwingConstants.CENTER);
        titleLabel.setFont(new Font("Arial", Font.BOLD, 32));
        titleLabel.setForeground(Color.WHITE);
        titleLabel.setBorder(BorderFactory.createEmptyBorder(20, 0, 20, 0));
        headerPanel.add(titleLabel, BorderLayout.CENTER);

        // profile pictures with hover effects
        JPanel profilePanel = new JPanel();
        profilePanel.setBackground(new Color(0, 102, 204));
        profilePanel.setLayout(new FlowLayout(FlowLayout.RIGHT));
        profilePanel.add(createProfilePic("rp.PNG", "Rauf", e -> openWebPage("about.html")));
        profilePanel.add(createProfilePic("umar.jpeg", "Custom Detection", e -> openWebPage("about.html")));
        profilePanel.add(createProfilePic("ahsan.jpeg", "About", e -> openWebPage("about.html")));
        
        headerPanel.add(profilePanel, BorderLayout.EAST);

        // center panel for buttons
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

        // footer
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
        
        // adding panels to main frame
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
        Map<String, Rectangle> drawnObjects = new HashMap<>(); // track drawn objects
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
        JFrame detectionFrame = new JFrame("Live Detection");
        detectionFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        detectionFrame.setSize(800, 600);
        detectionFrame.setLayout(new BorderLayout());

        JLabel imageLabel = new JLabel();
        JLabel objectCountLabel = new JLabel("Objects Detected: 0", SwingConstants.CENTER);
        objectCountLabel.setFont(new Font("Arial", Font.BOLD, 18));
        objectCountLabel.setForeground(Color.WHITE);

        JPanel imagePanel = new JPanel(new BorderLayout());
        imagePanel.setBackground(Color.DARK_GRAY);
        imagePanel.add(imageLabel, BorderLayout.CENTER);
        imagePanel.add(objectCountLabel, BorderLayout.SOUTH);

        JButton backButton = new JButton("Back");
        backButton.addActionListener(e -> {
            detectionFrame.dispose();
            mainFrame.setVisible(true);
        });

        detectionFrame.add(imagePanel, BorderLayout.CENTER);
        detectionFrame.add(backButton, BorderLayout.SOUTH);
        detectionFrame.setVisible(true);

        new Thread(() -> {
            String cfgFile = "yolov3.cfg";
            String weightsFile = "yolov3.weights";
            String namesFile = "coco.names";

            Net net = Dnn.readNetFromDarknet(cfgFile, weightsFile);
            List<String> classNames = new ArrayList<>();
            try (BufferedReader br = new BufferedReader(new FileReader(namesFile))) {
                String line;
                while ((line = br.readLine()) != null) {
                    classNames.add(line);
                }
            } catch (IOException e) {
                e.printStackTrace();
                return;
            }

            Map<String, Color> colorMap = new HashMap<>();
            Random rng = new Random();
            for (String className : classNames) {
                colorMap.put(className, new Color(rng.nextInt(256), rng.nextInt(256), rng.nextInt(256)));
            }

            VideoCapture capture = new VideoCapture(0);
            Mat frame = new Mat();

            while (capture.read(frame)) {
                if (!detectionFrame.isVisible()) {
                    break;
                }

                List<Mat> result = new ArrayList<>(2);
                Mat blob = Dnn.blobFromImage(frame, 0.00392, new Size(416, 416), new Scalar(0), true, false);
                net.setInput(blob);
                List<Mat> out = new ArrayList<>();
                List<String> outNames = net.getUnconnectedOutLayersNames();
                net.forward(out, outNames);

                drawDetections(frame, out, classNames, colorMap, objectCountLabel);

                ImageIcon imageIcon = new ImageIcon(Mat2BufferedImage(frame));
                imageLabel.setIcon(imageIcon);
                imageLabel.repaint();

                try {
                    Thread.sleep(30);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
            capture.release();
        }).start();
    }

    private static void startCustomDetection(JFrame mainFrame) {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
        int returnValue = fileChooser.showOpenDialog(mainFrame);
        if (returnValue == JFileChooser.APPROVE_OPTION) {
            File selectedFile = fileChooser.getSelectedFile();
            String filePath = selectedFile.getAbsolutePath();

            String cfgFile = "yolov3.cfg";
            String weightsFile = "yolov3.weights";
            String namesFile = "coco.names";

            Net net = Dnn.readNetFromDarknet(cfgFile, weightsFile);
            List<String> classNames = new ArrayList<>();
            try (BufferedReader br = new BufferedReader(new FileReader(namesFile))) {
                String line;
                while ((line = br.readLine()) != null) {
                    classNames.add(line);
                }
            } catch (IOException e) {
                e.printStackTrace();
                return;
            }

            Map<String, Color> colorMap = new HashMap<>();
            Random rng = new Random();
            for (String className : classNames) {
                colorMap.put(className, new Color(rng.nextInt(256), rng.nextInt(256), rng.nextInt(256)));
            }

            Mat image = Imgcodecs.imread(filePath);
            Mat blob = Dnn.blobFromImage(image, 0.00392, new Size(416, 416), new Scalar(0), true, false);
            net.setInput(blob);
            List<Mat> out = new ArrayList<>();
            List<String> outNames = net.getUnconnectedOutLayersNames();
            net.forward(out, outNames);

            drawDetections(image, out, classNames, colorMap, new JLabel());

            ImageIcon imageIcon = new ImageIcon(Mat2BufferedImage(image));
            JLabel imageLabel = new JLabel(imageIcon);
            JScrollPane scrollPane = new JScrollPane(imageLabel);

            JFrame detectionFrame = new JFrame("Custom Detection");
            detectionFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            detectionFrame.setSize(800, 600);
            detectionFrame.setLayout(new BorderLayout());
            detectionFrame.add(scrollPane, BorderLayout.CENTER);

            JButton backButton = new JButton("Back");
            backButton.addActionListener(e -> {
                detectionFrame.dispose();
                mainFrame.setVisible(true);
            });

            detectionFrame.add(backButton, BorderLayout.SOUTH);
            detectionFrame.setVisible(true);
        }
    }

    private static BufferedImage Mat2BufferedImage(Mat matrix) {
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

    private static void showAbout() {
        JFrame aboutFrame = new JFrame("About");
        aboutFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        aboutFrame.setSize(400, 300);
        aboutFrame.setLayout(new BorderLayout());

        JLabel aboutLabel = new JLabel("<html><div style='text-align: center;'>"
                + "<h1>About</h1>"
                + "<p>This Object Detection App is developed by Rauf, Ahsan, and Umar.</p>"
                + "<p>It uses OpenCV and YOLO for object detection.</p>"
                + "<p>Version 1.0</p>"
                + "</div></html>", SwingConstants.CENTER);
        aboutLabel.setFont(new Font("Arial", Font.PLAIN, 16));
        aboutFrame.add(aboutLabel, BorderLayout.CENTER);

        JButton closeButton = new JButton("Close");
        closeButton.addActionListener(e -> aboutFrame.dispose());
        aboutFrame.add(closeButton, BorderLayout.SOUTH);

        aboutFrame.setVisible(true);
    }

    private static void openWebPage(String url) {
        try {
            Desktop.getDesktop().browse(new URI(url));
        } catch (IOException | URISyntaxException e) {
            e.printStackTrace();
        }
    }
}
