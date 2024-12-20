#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// Variables for color selection
bool selectColor = false;
Scalar lower_color, upper_color;
Point clickedPoint(-1, -1);

// Variables for undo functionality
vector<Mat> canvasHistory;
Mat canvas;

// Mouse callback function to capture the clicked point
void mouseCallback(int event, int x, int y, int, void*) {
    if (event == EVENT_LBUTTONDOWN) {
        clickedPoint = Point(x, y);
        selectColor = true;
    }
}

// Function to display a color palette
void showColorPalette(Mat& frame) {
    int paletteWidth = 50;
    int colors[6][3] = {{0, 255, 0}, {255, 0, 0}, {0, 0, 255}, {0, 255, 255}, {255, 0, 255}, {255, 255, 0}};
    for (int i = 0; i < 6; ++i) {
        rectangle(frame, Point(i * paletteWidth, 0), Point((i + 1) * paletteWidth, 50), Scalar(colors[i][0], colors[i][1], colors[i][2]), -1);
    }
}

// Function to handle palette clicks
void handlePaletteClick(int x, int y) {
    int paletteWidth = 50;
    if (y <= 50) {
        int index = x / paletteWidth;
        if (index >= 0 && index < 6) {
            switch (index) {
                case 0: lower_color = Scalar(35, 140, 60); upper_color = Scalar(85, 255, 255); break; // Green
                case 1: lower_color = Scalar(100, 150, 60); upper_color = Scalar(140, 255, 255); break; // Blue
                case 2: lower_color = Scalar(0, 150, 60); upper_color = Scalar(10, 255, 255); break;  // Red
                case 3: lower_color = Scalar(25, 150, 60); upper_color = Scalar(35, 255, 255); break; // Yellow
                case 4: lower_color = Scalar(140, 150, 60); upper_color = Scalar(170, 255, 255); break; // Magenta
                case 5: lower_color = Scalar(20, 100, 100); upper_color = Scalar(30, 255, 255); break; // Cyan
            }
            cout << "Selected color from palette: " << index << endl;
        }
    }
}

int main() {
    VideoCapture cap(0); // Open the default camera
    if (!cap.isOpened()) {
        cout << "Error opening camera!" << endl;
        return -1;
    }

    Mat frame, hsv, mask;
    canvas = Mat::zeros(480, 640, CV_8UC3); // Blank canvas
    canvasHistory.push_back(canvas.clone()); // Save initial state 
    int prev_x = -1, prev_y = -1;

    // Set the mouse callback
    namedWindow("Air Canvas");
    setMouseCallback("Air Canvas", mouseCallback);

    while (true) {
        cap >> frame; // Capture frame
        if (frame.empty()) {
            cout << "Error: Empty frame captured!" << endl;
            break; // Exit the loop if no frame is captured
        }

        flip(frame, frame, 1); // Flip for mirror effect

        // Draw the color palette
        showColorPalette(frame);

        // Convert to HSV color space
        cvtColor(frame, hsv, COLOR_BGR2HSV);

        // Handle mouse clicks
        if (selectColor && clickedPoint.x != -1 && clickedPoint.y != -1) {
            if (clickedPoint.y <= 50) {
                // If clicked on palette
                handlePaletteClick(clickedPoint.x, clickedPoint.y);
            } else {
                // If clicked elsewhere in the frame, get HSV value
                Vec3b pixel = hsv.at<Vec3b>(clickedPoint.y, clickedPoint.x);
                int h = pixel[0], s = pixel[1], v = pixel[2];
                lower_color = Scalar(h - 10, max(s - 50, 0), max(v - 50, 0));
                upper_color = Scalar(h + 10, min(s + 50, 255), min(v + 50, 255));
                cout << "Selected color from click: H=" << h << " S=" << s << " V=" << v << endl;
            }
            selectColor = false;
        }

        // Threshold the image using selected color range
        if (!lower_color.empty() && !upper_color.empty()) {
            inRange(hsv, lower_color, upper_color, mask);
            erode(mask, mask, Mat(), Point(-1, -1), 2);
            dilate(mask, mask, Mat(), Point(-1, -1), 2);

            // Find contours
            vector<vector<Point>> contours;
            findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            if (!contours.empty()) {
                // Find largest contour
                double maxArea = 0;
                vector<Point> largestContour;
                for (const auto& contour : contours) {
                    double area = contourArea(contour);
                    if (area > maxArea) {
                        maxArea = area;
                        largestContour = contour;
                    }
                }

                // Get the center of the largest contour
                Moments m = moments(largestContour);
                if (m.m00 != 0) {
                    int x = int(m.m10 / m.m00);
                    int y = int(m.m01 / m.m00);

                    // Draw line on canvas
                    if (prev_x != -1 && prev_y != -1) {
                        canvasHistory.push_back(canvas.clone()); // Save canvas state
                        line(canvas, Point(prev_x, prev_y), Point(x, y), Scalar(0, 255, 0), 4);
                    }

                    prev_x = x;
                    prev_y = y;
                } else {
                    prev_x = -1;
                    prev_y = -1;
                }
            }
        }

        // Merge canvas and frame
        Mat output;
        if (!frame.empty() && !canvas.empty()) {
            add(frame, canvas, output);
        } else {
            cout << "Error: Invalid frame or canvas!" << endl;
            break;
        }

        // Display instructions
        putText(output, "Press 's' to save, 'c' to clear, 'u' to undo", Point(10, 460), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

        // Display output
        if (!output.empty()) {
            imshow("Air Canvas", output);
        } else {
            cout << "Error: Output is empty!" << endl;
            break;
        }

        imshow("Mask", mask);

        // Handle key events
        char key = waitKey(1);
        if (key == 'q') {
            break; // Quit
        } else if (key == 's') {
            imwrite("drawing.jpg", canvas);
            cout << "Drawing saved as drawing.jpg" << endl;
        } else if (key == 'c') {
            canvas = Mat::zeros(480, 640, CV_8UC3); // Clear canvas
            canvasHistory.clear();
            canvasHistory.push_back(canvas.clone());
            cout << "Canvas cleared!" << endl;
        } else if (key == 'u') {
            if (canvasHistory.size() > 1) {
                canvasHistory.pop_back(); // Remove last state
                canvas = canvasHistory.back().clone(); // Restore previous state
                cout << "Undo last stroke!" << endl;
            }
        }
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
