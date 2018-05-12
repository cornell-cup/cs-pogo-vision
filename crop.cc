#include <apriltag/apriltag.h>
#include <apriltag/tag16h5.h>
#include <apriltag/tag36h11.h>
#include <apriltag/tag36artoolkit.h>
#include <ctime>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

using namespace cv;
using std::vector;

#define TAG_SIZE 6.5f

int main(int argc, char** argv) {
    // Display usage
    if (argc < 2) {
        printf("Usage: %s [cameras...]\n", argv[0]);
        return -1;
    }

    // Parse arguments
    vector<VideoCapture> devices;
    vector<int> device_ids;
    vector<Mat> device_camera_matrix;
    vector<Mat> device_dist_coeffs;

    for (int i = 1; i < argc; i++) {
        int id = atoi(argv[i]);
        VideoCapture device(id);
        if (!device.isOpened()) {
            std::cerr << "Failed to open video capture device " << id << std::endl;
            continue;
        }

        std::ifstream fin;
        fin.open(argv[i]);
        if (fin.fail()) {
            std::cerr << "Failed to open file " << argv[i] << std::endl;
            continue;
        }

        Mat camera_matrix, dist_coeffs;
        std::string line;
        // TODO Error checking
        while (std::getline(fin, line)) {
            std::stringstream line_stream(line);
            std::string key, equals;
            line_stream >> key >> equals;
            if (key == "camera_matrix") {
                vector<double> data;
                for (int i = 0; i < 9; i++) {
                    double v;
                    line_stream >> v;
                    data.push_back(v);
                }
                camera_matrix = Mat(data, true).reshape(1, 3);
            }
            else if (key == "dist_coeffs") {
                vector<double> data;
                for (int i = 0; i < 5; i++) {
                    double v;
                    line_stream >> v;
                    data.push_back(v);
                }
                dist_coeffs = Mat(data, true).reshape(1, 1);
            }
            else {
                std::cerr << "Unrecognized key '" << key << "' in file " << argv[i] << std::endl;
            }
        }

        if (camera_matrix.rows != 3 || camera_matrix.cols != 3) {
            std::cerr << "Error reading camera_matrix in file " << argv[i] << std::endl;
            continue;
        }

        if (dist_coeffs.rows != 1 || dist_coeffs.cols != 5) {
            std::cerr << "Error reading dist_coeffs in file " << argv[i] << std::endl;
            continue;
        }
	device.set(CV_CAP_PROP_FRAME_WIDTH, 720);
	device.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
        device.set(CV_CAP_PROP_FPS, 30);

        devices.push_back(device);
        device_ids.push_back(id);
        device_camera_matrix.push_back(camera_matrix);
        device_dist_coeffs.push_back(dist_coeffs);
    }

    // Initialize detector
    apriltag_family_t* tf = tag36h11_create();
    tf->black_border = 1;

    apriltag_detector_t* td = apriltag_detector_create();
    apriltag_detector_add_family(td, tf);
    td->quad_decimate = 1.0;
    td->quad_sigma = 0.0;
    td->nthreads = 2;
    td->debug = 0;
    td->refine_edges = 0;
    td->refine_decode = 0;
    td->refine_pose = 0;

    int key = 0;
    Mat frame, gray;
    Mat roiImg, img;
    float max_X = 0;
    float max_Y = 0;
    float min_X = 999999;
    float min_Y = 999999;
    int saved = 0;
    while (key != 27) { // Quit on escape keypress
        for (size_t i = 0; i < devices.size(); i++) {
            if (!devices[i].isOpened()) {
                continue;
            }

            devices[i] >> frame;
            cvtColor(frame, gray, COLOR_BGR2GRAY);
            image_u8_t im = {
                .width = gray.cols,
                .height = gray.rows,
                .stride = gray.cols,
                .buf = gray.data
            };
	    zarray_t* init_detections = apriltag_detector_detect(td, &im);
	    if (saved == 0){
            for (int j = 0; j < zarray_size(init_detections); j++) {
                // Get the ith detection
                apriltag_detection_t *det;
                zarray_get(init_detections, j, &det);
                if ((det->id) == 1 || (det->id) == 5 || (det->id) == 3 || (det->id) == 2) {
		    line(frame, Point(det->p[0][0], det->p[0][1]),
                            Point(det->p[1][0], det->p[1][1]),
                            Scalar(0, 0xff, 0), 2);
                    line(frame, Point(det->p[0][0], det->p[0][1]),
                            Point(det->p[3][0], det->p[3][1]),
                            Scalar(0, 0, 0xff), 2);
                    line(frame, Point(det->p[1][0], det->p[1][1]),
                            Point(det->p[2][0], det->p[2][1]),
                            Scalar(0xff, 0, 0), 2);
                    line(frame, Point(det->p[2][0], det->p[2][1]),
                            Point(det->p[3][0], det->p[3][1]),
                            Scalar(0xff, 0, 0), 2);
                    if((det->p[0][0]) < min_X){
			min_X = det->p[0][0];
		    }
		    if((det->p[3][0]) < min_X){
			min_X = det->p[3][0];
		    }
		    if((det->p[2][1]) < min_Y){
			min_Y = det->p[2][1];
		    }
		    if((det->p[3][1]) < min_Y){
			min_Y = det->p[3][1];
		    }
                    if((det->p[2][0]) > max_X){
			max_X = det->p[2][0];
		    }
                    if((det->p[1][0]) > max_X){
			max_X = det->p[1][0];
		    }
		    if((det->p[0][1]) > max_Y){
			max_Y = det->p[0][1];
		    }
		    if((det->p[1][1]) > max_Y){
			max_Y = det->p[1][1];
		    }
		}
            }
            }


            if (key == 'x'){
	    min_X -= 7;
	    max_X += 7;
	    min_Y -= 7;
	    max_Y += 7;
            std::ofstream fout;
            fout.open("crop.calib", std::ofstream::out);
            fout << min_X << " " << min_Y << " " << max_X << " " << max_Y;
	    fout << std::endl;
            fout.close();
            
            img = frame.clone();
            Rect roi = Rect(floor(min_X), floor(min_Y), floor(max_X - min_X), floor(max_Y - min_Y));
	    roiImg = img(roi);
            cvtColor(roiImg, gray, COLOR_BGR2GRAY);
            printf("crop.calib saved\n");
            saved = 1;}

	    zarray_destroy(init_detections);

            if (saved == 0){
                imshow(std::to_string(i), frame);
	    }
            else{
     	        img = frame.clone();
            	Rect roi = Rect(floor(min_X), floor(min_Y), floor(max_X - min_X), floor(max_Y - min_Y));
	    	roiImg = img(roi);
                imshow(std::to_string(i), roiImg);
            }
        }

        key = waitKey(16);
    }
}
