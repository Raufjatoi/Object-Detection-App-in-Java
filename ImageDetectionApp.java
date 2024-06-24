import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.imgproc.Imgproc;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

public class ImageDetectionApp {
    static {
        // Load opencv_java490.dll from the specified path
        System.load("D:\\Downloads\\opencv\\build\\java\\x64\\opencv_java490.dll");
    }


    public static void main(String[] args) {
        // Load OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Load input image
        String inputImagePath = "food.JPG";
        Mat image = Imgcodecs.imread(inputImagePath);

        // Path to your cascade file (e.g., haarcascade_frontalface_alt.xml)
        // Replace with an actual cascade file for object detection if available
        String cascadePath = "haarcascade_frontalface_alt.xml";
        CascadeClassifier detector = new CascadeClassifier();
        detector.load(cascadePath);

        // Detect objects in the image
        MatOfRect detectedObjects = new MatOfRect();
        detector.detectMultiScale(image, detectedObjects);

        // Convert Mat to BufferedImage for drawing and annotation
        BufferedImage bufImage = matToBufferedImage(image);

        // Draw rectangles around detected objects and annotate
        Graphics2D g2d = bufImage.createGraphics();
        g2d.setColor(Color.GREEN);
        g2d.setStroke(new BasicStroke(2));

        Font font = new Font("Arial", Font.BOLD, 16);
        g2d.setFont(font);

        for (Rect rect : detectedObjects.toArray()) {
            // Draw rectangle
            g2d.drawRect(rect.x, rect.y, rect.width, rect.height);

            // Annotate object
            String label = "Object";
            int textX = rect.x;
            int textY = rect.y - 5;
            g2d.drawString(label, textX, textY);
        }

        g2d.dispose();

        // Convert BufferedImage back to Mat
        image = bufferedImageToMat(bufImage);

        // Save annotated image
        String outputImagePath = "detected.JPG";
        Imgcodecs.imwrite(outputImagePath, image);
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
        Mat mat = new Mat(height, width, channels == 3 ? org.opencv.core.CvType.CV_8UC3 : org.opencv.core.CvType.CV_8UC1);
        mat.put(0, 0, data);
        return mat;
    }
}
