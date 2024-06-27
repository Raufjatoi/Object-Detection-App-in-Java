import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.*;
import java.net.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class custom {

    static {
        // Load the OpenCV library
        System.load("D:\\Downloads\\opencv\\build\\java\\x64\\opencv_java490.dll");
    }

    public static void main(String[] args) {
        // Load OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

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
            byte[] imageData = null;
            if (request.toString().contains("Content-Type: image")) {
                ByteArrayOutputStream imageBuffer = new ByteArrayOutputStream();
                while (in.ready()) {
                    imageBuffer.write(in.read());
                }
                imageData = imageBuffer.toByteArray();
            }

            // Process image if data is received
            if (imageData != null) {
                // Perform object detection
                Mat image = Imgcodecs.imdecode(new MatOfByte(imageData), Imgcodecs.IMREAD_COLOR);

                String folderPath = "D:\\Documents\\GitHub\\java\\";
                String modelConfiguration = folderPath + "yolov4.cfg";
                String modelWeights = folderPath + "yolov4.weights";
                String classNamesFile = folderPath + "coco.names";

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
                Imgcodecs.imwrite(outputImagePath, image);

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
        // Define colors for each class
        colorMap.put("person", Color.BLUE);
        colorMap.put("bicycle", Color.RED);
        colorMap.put("car", Color.RED);
        colorMap.put("motorbike", Color.RED);
        colorMap.put("aeroplane", Color.RED);
        colorMap.put("bus", Color.RED);
        colorMap.put("train", Color.RED);
        colorMap.put("truck", Color.RED);
        colorMap.put("boat", Color.RED);
        colorMap.put("traffic light", Color.RED);
        colorMap.put("fire hydrant", Color.RED);
        colorMap.put("stop sign", Color.RED);
        colorMap.put("parking meter", Color.RED);
        colorMap.put("bench", Color.RED);
        colorMap.put("bird", Color.RED);
        colorMap.put("cat", Color.RED);
        colorMap.put("dog", Color.RED);
        colorMap.put("horse", Color.RED);
        colorMap.put("sheep", Color.RED);
        colorMap.put("cow", Color.RED);
        colorMap.put("elephant", Color.RED);
        colorMap.put("bear", Color.RED);
        colorMap.put("zebra", Color.RED);
        colorMap.put("giraffe", Color.RED);
        colorMap.put("backpack", Color.RED);
        colorMap.put("umbrella", Color.RED);
        colorMap.put("handbag", Color.RED);
        colorMap.put("tie", Color.RED);
        colorMap.put("suitcase", Color.RED);
        colorMap.put("frisbee", Color.RED);
        colorMap.put("skis", Color.RED);
        colorMap.put("snowboard", Color.RED);
        colorMap.put("sports ball", Color.RED);
        colorMap.put("kite", Color.RED);
        colorMap.put("baseball bat", Color.RED);
        colorMap.put("baseball glove", Color.RED);
        colorMap.put("skateboard", Color.RED);
        colorMap.put("surfboard", Color.RED);
        colorMap.put("tennis racket", Color.RED);
        colorMap.put("bottle", Color.RED);
        colorMap.put("wine glass", Color.RED);
        colorMap.put("cup", Color.RED);
        colorMap.put("fork", Color.RED);
        colorMap.put("knife", Color.RED);
        colorMap.put("spoon", Color.RED);
        colorMap.put("bowl", Color.RED);
        colorMap.put("banana", Color.RED);
        colorMap.put("apple", Color.RED);
        colorMap.put("sandwich", Color.RED);
        colorMap.put("orange", Color.RED);
        colorMap.put("broccoli", Color.RED);
        colorMap.put("carrot", Color.RED);
        colorMap.put("hot dog", Color.RED);
        colorMap.put("pizza", Color.RED);
        colorMap.put("donut", Color.RED);
        colorMap.put("cake", Color.RED);
        colorMap.put("chair", Color.RED);
        colorMap.put("sofa", Color.RED);
        colorMap.put("pottedplant", Color.RED);
        colorMap.put("bed", Color.RED);
        colorMap.put("diningtable", Color.RED);
        colorMap.put("toilet", Color.RED);
        colorMap.put("tvmonitor", Color.RED);
        colorMap.put("laptop", Color.RED);
        colorMap.put("mouse", Color.RED);
        colorMap.put("remote", Color.RED);
        colorMap.put("keyboard", Color.RED);
        colorMap.put("cell phone", Color.RED);
        colorMap.put("microwave", Color.RED);
        colorMap.put("oven", Color.RED);
        colorMap.put("toaster", Color.RED);
        colorMap.put("sink", Color.RED);
        colorMap.put("refrigerator", Color.RED);
        colorMap.put("book", Color.RED);
        colorMap.put("clock", Color.RED);
        colorMap.put("vase", Color.RED);
        colorMap.put("scissors", Color.RED);
        colorMap.put("teddy bear", Color.RED);
        colorMap.put("hair drier", Color.RED);
        colorMap.put("toothbrush", Color.RED);
        return colorMap;
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
}

       
