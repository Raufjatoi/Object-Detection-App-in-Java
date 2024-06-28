import java.util.List; // Import java.util.List explicitly

import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.util.*;
import javax.imageio.ImageIO;

public class test {

    static {
        // Load OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        try {
            // Create a server socket on port 8000
            ServerSocket serverSocket = new ServerSocket(8000);
            System.out.println("Server started. Listening on port 8000...");

            while (true) {
                // Wait for client connection
                Socket clientSocket = serverSocket.accept();
                System.out.println("Client connected: " + clientSocket.getInetAddress());

                // Handle client request in a separate thread
                new Thread(() -> handleClientRequest(clientSocket)).start();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void handleClientRequest(Socket clientSocket) {
        try {
            // Read HTTP request from client
            BufferedReader in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
            String line;
            StringBuilder request = new StringBuilder();
            while ((line = in.readLine()) != null && !line.isEmpty()) {
                request.append(line).append("\r\n");
            }
            System.out.println("Received request:\n" + request);

            // Extract image data from POST request
            String contentType = "";
            int contentLength = 0;
            for (String header : request.toString().split("\r\n")) {
                if (header.startsWith("Content-Type:")) {
                    contentType = header.split(": ")[1];
                } else if (header.startsWith("Content-Length:")) {
                    contentLength = Integer.parseInt(header.split(": ")[1]);
                }
            }

            if (contentType.contains("multipart/form-data")) {
                byte[] imageData = new byte[contentLength];
                in.read(imageData);

                // Perform object detection
                Mat image = Imgcodecs.imdecode(new MatOfByte(imageData), Imgcodecs.IMREAD_COLOR);

                String folderPath = "D:\\Documents\\GitHub\\java\\";
                String modelConfiguration = folderPath + "models/yolov4.cfg";
                String modelWeights = folderPath + "models/yolov4.weights";
                String classNamesFile = folderPath + "models/coco.names";

                Net net = Dnn.readNetFromDarknet(modelConfiguration, modelWeights);
                if (net.empty()) {
                    System.err.println("Cannot load network using given configuration and weights files.");
                    return;
                }

                List<String> classNames = loadClassNames(classNamesFile);
                if (classNames.isEmpty()) {
                    System.err.println("Cannot load class names.");
                    return;
                }

                Mat blob = Dnn.blobFromImage(image, 1 / 255.0, new Size(416, 416), new Scalar(0, 0, 0), true, false);
                net.setInput(blob);

                List<Mat> result = new ArrayList<>();
                List<String> outBlobNames = getOutputNames(net);
                net.forward(result, outBlobNames);

                BufferedImage bufImage = matToBufferedImage(image);
                Graphics2D g2d = bufImage.createGraphics();
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

                String outputImagePath = folderPath + "detected.png";
                ImageIO.write(bufImage, "png", new File(outputImagePath));

                // Send JSON response with path to annotated image
                String jsonResponse = "{\"message\": \"Object detection completed.\", \"imagePath\": \"" + outputImagePath + "\"}";
                PrintWriter out = new PrintWriter(clientSocket.getOutputStream(), true);
                out.println("HTTP/1.1 200 OK");
                out.println("Content-Type: application/json");
                out.println("Content-Length: " + jsonResponse.length());
                out.println();
                out.println(jsonResponse);
                out.flush();

                clientSocket.close();
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
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
        // Define colors for each class (use unique colors for better visibility)
        colorMap.put("person", new Color(255, 0, 0)); // Red
        colorMap.put("bicycle", new Color(0, 255, 0)); // Green
        colorMap.put("car", new Color(0, 0, 255)); // Blue
        // Add more classes and colors as needed
        return colorMap;
    }
}
