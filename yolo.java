import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class yolo {
    static {
        // Load the OpenCV library
        System.load("D:\\Downloads\\opencv\\build\\java\\x64\\opencv_java490.dll");
    }

    public static void main(String[] args) {
        // Load OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Define the file paths
        String folderPath = "D:\\Documents\\GitHub\\java\\";
        String inputImagePath = folderPath + "food.JPG";
        String modelConfiguration = folderPath + "yolov3.cfg";
        String modelWeights = folderPath + "yolov3.weights";
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
        g2d.setColor(Color.GREEN);
        g2d.setStroke(new BasicStroke(2));
        Font font = new Font("Arial", Font.BOLD, 16);
        g2d.setFont(font);

        float confThreshold = 0.5f;
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

                    // Draw rectangle
                    g2d.drawRect(left, top, width, height);

                    // Annotate object
                    String label = classNames.get(classId) + ": " + confidence;
                    int textX = left;
                    int textY = top - 5;
                    g2d.drawString(label, textX, textY);
                }
            }
        }

        g2d.dispose();

        // Convert BufferedImage back to Mat
        image = bufferedImageToMat(bufImage);

        // Save annotated image
        String outputImagePath = folderPath + "detected.JPG";
        Imgcodecs.imwrite(outputImagePath, image);
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

    // Utility method to convert BufferedImage to Mat
    private static Mat bufferedImageToMat(BufferedImage image) {
        byte[] data = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        int width = image.getWidth();
        int height = image.getHeight();
        int channels = image.getSampleModel().getNumBands();
        Mat mat = new Mat(height, width, channels == 3 ? CvType.CV_8UC3 : CvType.CV_8UC1);
        mat.put(0, 0, data);
        return mat;
    }
}
