#include <apriltag/apriltag.h>
#include <apriltag/tag16h5.h>
#include <apriltag/tag36h11.h>
#include <apriltag/tag36artoolkit.h>
#include <curl/curl.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <wiringSerial.h>

using namespace cv;
using std::vector;

#define TAG_SIZE 6.5f

int main(int argc, char** argv) {
    // Display usage
    if (argc < 3) {
        printf("Usage: %s <basestation url> [cameras...]\n", argv[0]);
        return -1;
    }
    // Parse arguments
    CURL *curl;
    curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Failed to initialize curl" << std::endl;
        return -1;
    }
    curl_easy_setopt(curl, CURLOPT_URL, argv[1]);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 200L);

    vector<VideoCapture> devices;
    vector<int> device_ids;
    vector<Mat> device_camera_matrix;
    vector<Mat> device_dist_coeffs;
    vector<Mat> device_transform_matrix;
    for (int i = 2; i < argc; i++) {
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
        Mat camera_matrix, dist_coeffs, transform_matrix;
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
                camera_matrix = Mat(data, true).reshape(1,3);
            }
            else if (key == "dist_coeffs") {
                vector<double> data;
                for (int i = 0; i < 5; i++) {
                    double v;
                    line_stream >> v;
                    data.push_back(v);
                }
                dist_coeffs = Mat(data, true).reshape(1,1);
            }
            else if (key == "transform_matrix"){
                vector<double> data;
                for (int i = 0; i < 16; i++){
                  double v;
                  line_stream >> v;
                  data.push_back(v);
                }
                transform_matrix = Mat(data, true).reshape(1,4);
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

        if (transform_matrix.rows != 4 || transform_matrix.cols != 4){
            std::cerr << "Error reading transform_matrix in file " << argv[i] << std::endl;
            continue;
        }

        //device.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
        //device.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
        device.set(CV_CAP_PROP_FRAME_WIDTH, 720);
        device.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
        device.set(CV_CAP_PROP_FPS, 30);
        devices.push_back(device);
        device_ids.push_back(id);
        device_camera_matrix.push_back(camera_matrix);
        device_dist_coeffs.push_back(dist_coeffs);
        device_transform_matrix.push_back(transform_matrix);
    }
    // Initialize detector
    apriltag_family_t* tf = tag16h5_create();
    tf->black_border = 1;
    apriltag_detector_t* td = apriltag_detector_create();
    apriltag_detector_add_family(td, tf);
    td->quad_decimate = 1.0;
    td->quad_sigma = 0.0;
    td->nthreads = 4;
    td->debug = 0;
    td->refine_edges = 1;
    td->refine_decode = 0;
    td->refine_pose = 0;
    int key = 0;
    float max_X;
    float max_Y;
    float min_X;
    float min_Y;
    std::ifstream infile("crop.calib");
    infile >> min_X;
    infile >> min_Y;
    infile >> max_X;
    infile >> max_Y;
    infile.close();
    float center_X = (max_X + min_X)/2;
    float center_Y = (max_Y + min_Y)/2;
    int fd;
    Mat frame, gray;
    Mat cframe, img;
    char postDataBuffer[100];
    int sOpen = 0;

    while (key != 27) { // Quit on escape keypress

    	// map to store weighted coordinates [x, y, z, roll, pitch, yaw, total weight]
    	// weightings are based on distance from camera to tag
        std::map <int, float[7]> weightedCoords;

        for (size_t i = 0; i < devices.size(); i++) {
            if (!devices[i].isOpened()) {
                continue;
            }

            devices[i] >> cframe;
            img = cframe.clone();
            Rect roi = Rect(floor(min_X), floor(min_Y), floor(max_X - min_X), floor(max_Y - min_Y));
	    frame = img(roi);
            cvtColor(frame, gray, COLOR_BGR2GRAY);
            image_u8_t im = {
                .width = gray.cols,
                .height = gray.rows,
                .stride = gray.cols,
                .buf = gray.data
            };

	    zarray_t* detections = apriltag_detector_detect(td, &im);            
	    vector<Point2f> img_points(4);
            vector<Point3f> obj_points(4);
            Mat rvec(3, 1, CV_64FC1);
            Mat tvec(3, 1, CV_64FC1);

            printf("~~~~~~~~~~~~~ Camera %d ~~~~~~~~~~~~\n", (int)i);

            // get the coordinates of the camera
            vector<double> data2;
            data2.push_back(0);
            data2.push_back(0);
            data2.push_back(0);
            data2.push_back(1);
            Mat genout = Mat(data2,true).reshape(1, 4);
            Mat cameraXYZS = device_transform_matrix[i] * genout;
            printf("coordinates: (%f, %f, %f)\n", cameraXYZS.at<double>(0), cameraXYZS.at<double>(1), cameraXYZS.at<double>(2));
	    char charX[10];
	    char charY[10];
	    char data[128];
            for (int j = 0; j < zarray_size(detections); j++) {
	
                // Get the ith detection
                apriltag_detection_t *det;
                zarray_get(detections, j, &det);
		if ((det->id) == 6){
		  if ((fd = serialOpen("/dev/ttyACM0", 115200)) < 0){
    		    fprintf(stderr, "Unable to open serial device: %s\n", strerror (errno));
		    }
		  else{
		    if (!sOpen){
		      fd = serialOpen("/dev/ttyACM0", 115200);
		      sOpen = 1;
		    }
		  }
		    float bot_X = ((det->p[0][0] + det->p[1][0])/2);
		    float bot_Y = ((det->p[0][1] + det->p[3][1])/2);
		    float prop_x = bot_X < center_X ? -min_X/bot_X : bot_X/max_X;
		    float prop_y = bot_Y > center_Y ? min_Y/bot_Y : -bot_Y/max_Y;
		    prop_x += 1; // range from 0-2
		    prop_y += 1; // range from 0-2
		    prop_x *= 100; // convert from decimal to int
		    prop_y *= 100; // convert from decimal to int
		    printf("%d %d\n", (int)floor(prop_x), (int)floor(prop_y));
		    
		    sprintf(charX, "%d", (int)floor(prop_x));
		    sprintf(charY, "%d", (int)floor(prop_y));
		    //sprintf(charX, "%d", (int)20);
		    //sprintf(charY, "%d", (int)30);
		    sprintf(data, "%s%s", charX, charY);
		    printf("%s", data);
                    serialPuts(fd, data);
		  
		}
                // Draw onto the frame
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

                // Compute transformation using PnP
                img_points[0] = Point2f(det->p[0][0], det->p[0][1]);
                img_points[1] = Point2f(det->p[1][0], det->p[1][1]);
                img_points[2] = Point2f(det->p[2][0], det->p[2][1]);
                img_points[3] = Point2f(det->p[3][0], det->p[3][1]);

                obj_points[0] = Point3f(-TAG_SIZE * 0.5f, -TAG_SIZE * 0.5f, 0.f);
                obj_points[1] = Point3f( TAG_SIZE * 0.5f, -TAG_SIZE * 0.5f, 0.f);
                obj_points[2] = Point3f( TAG_SIZE * 0.5f,  TAG_SIZE * 0.5f, 0.f);
                obj_points[3] = Point3f(-TAG_SIZE * 0.5f,  TAG_SIZE * 0.5f, 0.f);

                solvePnP(obj_points, img_points, device_camera_matrix[i],
                        device_dist_coeffs[i], rvec, tvec);
                Matx33d r;
                Rodrigues(rvec,r);

                vector<double> data;
                data.push_back(r(0,0));
                data.push_back(r(0,1));
                data.push_back(r(0,2));
                data.push_back(tvec.at<double>(0));
                data.push_back(r(1,0));
                data.push_back(r(1,1));
                data.push_back(r(1,2));
                data.push_back(tvec.at<double>(1));
                data.push_back(r(2,0));
                data.push_back(r(2,1));
                data.push_back(r(2,2));
                data.push_back(tvec.at<double>(2));
                data.push_back(0);
                data.push_back(0);
                data.push_back(0);
                data.push_back(1);
                Mat tag2cam = Mat(data,true).reshape(1, 4);

                Mat tag2orig = device_transform_matrix[i] * tag2cam;
                Mat tagXYZS = tag2orig * genout;

                // compute euler angles
             
                float eulerAngles[3];
                eulerAngles[0] = asin(tag2orig.at<double>(2,0));
                eulerAngles[1] = atan2(tag2orig.at<double>(2,1), tag2orig.at<double>(2,2));
                eulerAngles[2] = atan2(tag2orig.at<double>(1,0), tag2orig.at<double>(0,0));         
                float rd = (180.0/3.14159); // constant to convert radians to degrees

                // compute distance from camera to tag
                
                float xDistance = cameraXYZS.at<double>(0)-tagXYZS.at<double>(0);
                float yDistance = cameraXYZS.at<double>(1)-tagXYZS.at<double>(1);
                float zDistance = cameraXYZS.at<double>(2)-tagXYZS.at<double>(2);
                float distance  = sqrt(pow(xDistance,2) + pow(yDistance,2) + pow(zDistance,2));

                printf("----Tag %d\n", det->id);
                printf("xyz: (%3.3f,%3.3f,%3.3f)\n", tagXYZS.at<double>(0), tagXYZS.at<double>(1), tagXYZS.at<double>(2));
                printf("rpy: (%3.3f,%3.3f,%3.3f)\n", eulerAngles[0]*rd, eulerAngles[1]*rd, eulerAngles[2]*rd);
                printf("Camera to Tag: %3.3f\n", distance);

                // sava distances in weighted coords map

                weightedCoords[det->id][0] += tagXYZS.at<double>(0)/distance;
                weightedCoords[det->id][1] += tagXYZS.at<double>(1)/distance;
                weightedCoords[det->id][2] += tagXYZS.at<double>(2)/distance;
                weightedCoords[det->id][3] += eulerAngles[0]*rd/distance;
                weightedCoords[det->id][4] += eulerAngles[1]*rd/distance;
                weightedCoords[det->id][5] += eulerAngles[2]*rd/distance;
                weightedCoords[det->id][6] += 1/distance;

                // Send data to basestation - incomplete
                sprintf(postDataBuffer, "{\"id\":%d,\"x\":%f,\"y\":%f,\"z\":%f}",
                                det->id, tagXYZS.at<double>(0), tagXYZS.at<double>(1), tagXYZS.at<double>(2));
                curl_easy_setopt(curl, CURLOPT_POSTFIELDS, postDataBuffer);
                // TODO Check for error response
                curl_easy_perform(curl);
            }

            zarray_destroy(detections);

            imshow(std::to_string(i), frame);
        }

        printf("~~~~~~~~~~~~~ Overall Weightings ~~~~~~~~~~~~\n");

        std::map<int, float[7]>::iterator it = weightedCoords.begin();

        while (it != weightedCoords.end()) {
            printf("----Tag %d\n", it->first);
            printf("xyz: (%3.3f,%3.3f,%3.3f)\n", it->second[0]/it->second[6], it->second[1]/it->second[6], it->second[2]/it->second[6]);
            printf("rpy: (%3.3f,%3.3f,%3.3f)\n", it->second[3]/it->second[6], it->second[4]/it->second[6], it->second[5]/it->second[6]); 
            it++;
        }
        printf("==============================================\n");
	    

        key = waitKey(16);
    }

    curl_easy_cleanup(curl);
}