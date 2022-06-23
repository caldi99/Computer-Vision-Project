#include"../include/Trainer.h"

Trainer::Trainer() {

}


//training
bool Trainer::trainDictionary(std::string output_path) {
    try {
        //training set
        if (dataset_paths.size() < 1) {
            std::cout << "Dataset not loaded." << std::endl;
            return false;
        }


        std::cout << "=== BUILDING DICTIONARY ===" << std::endl;

        cv::Mat features;
        //cv::Ptr<cv::Feature2D> sift = cv::SIFT::create();
        cv::Ptr<cv::Feature2D> sift = cv::SIFT::create(50, 3, 0.12); //max 50 points, 0.12 / 3 contrast for filtering out (a little stronger than default)
        std::cout << "Extracting features..." << std::endl;

        //extract features from dataset
#pragma omp parallel for schedule(dynamic,3)
        for (int i = 0; i < dataset_paths.size(); i += TRAIN_STRIDE) {

            //get current image
            cv::Mat current = cv::imread(dataset_paths[i], cv::IMREAD_GRAYSCALE);

            //dictionary for whole image: do not split into positive and negative
            cv::GaussianBlur(current, current, cv::Size(3, 3), 5);//process boat
            std::vector<cv::KeyPoint> keypoint;
            cv::Mat descriptor;
            sift->detectAndCompute(current, cv::Mat(), keypoint, descriptor);

#pragma omp critical
            {
                features.push_back(descriptor);
            }

            current.deallocate();

            std::cout << std::fixed << std::setprecision(2) << (float)i / dataset_paths.size() * 100 << "% completed" << '\r';
        }

        std::cout << "Done." << std::endl << "Training with KMeans..." << std::endl;

        //use bowkmeanstrainer to cluster features in CLASS_COUNT classes (rule of thumb: 10 * feature class, but this is single class...)
        cv::TermCriteria tc(3, 100, 0.2);
        int attempts = 3;
        cv::BOWKMeansTrainer trainer(CLASS_COUNT, tc, attempts, cv::KMEANS_PP_CENTERS);
        dictionary = trainer.cluster(features);

        std::cout << "Done." << std::endl << "Saving to disk..." << std::endl;

        //save dictionary
        cv::FileStorage fs(output_path, cv::FileStorage::WRITE);
        fs << "dictionary" << dictionary;
        fs.release();

        std::cout << "Done." << std::endl;

        return true;
    }
    catch (cv::Exception e) {
        std::cout << "Error training dictionary: " << e.what() << std::endl;
        return false;
    }
}
bool Trainer::trainSVM(std::string output_path) {
    try {
        std::cout << "=== TRAINING SVM ===" << std::endl;

        //SVM: build training data
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED); //bf should be fine? otherwise flann
        extractor = cv::SIFT::create();
        bow_extractor = new cv::BOWImgDescriptorExtractor(extractor, matcher);
        bow_extractor->setVocabulary(dictionary);

        cv::Mat training_data(0, CLASS_COUNT, CV_32FC1);
        std::vector<int> labels;
        cv::Ptr<cv::Feature2D> sift = cv::SIFT::create(50, 3, 0.12); //max 50 points, 0.12 / 3 contrast for filtering out (a little stronger than default)

        std::cout << "Building SVM training data..." << std::endl;

        int pathlength = dataset_root_path.length();//+ IMG_FOLDER.length() + 1;

#pragma omp parallel for schedule(dynamic,3)
        for (int i = SVM_OFFSET; i < dataset_paths.size(); i += SVM_STRIDE) {

            //get current image
            std::string current_name = dataset_paths.at(i).substr(pathlength, dataset_paths.at(i).length() - pathlength - IMG_EXTENSION.length());
            cv::Mat current = cv::imread(dataset_paths[i], cv::IMREAD_GRAYSCALE);

            std::string path = "../Dataset/Txts_4800";
            //get ground truth file
            std::string gt = path  + "\\" + current_name + ".txt";
            std::ifstream groundtruth(gt);
            if (!groundtruth.is_open()) { //no ground truth for this image. does not necessarily mean no ships, so skipping to be safe
                std::cout << "Failed to open ground truth for " + current_name + ", skipping" << std::endl;
                continue;
            }

            //get coordinates of boats from ground truth file and build positive samples
            //bit of mismatch to keep in mind here. GT is x,y but opencv is r,c
            std::vector<cv::Range> rangex, rangey;
            for (std::string line; std::getline(groundtruth, line); ) {
                //line has format class:X;Y;W;Z;
                std::string clas,x1, y1, w, h;
                try {
                    line = line.substr(2,line.length());
                    x1 = line.substr(0, line.find(" "));
                    line = line.substr(x1.length() + 1);
                    y1 = line.substr(0, line.find(" "));
                    line = line.substr(y1.length() + 1);
                    w = line.substr(0, line.find(" "));
                    line = line.substr(w.length() + 1);
                    h = line.substr(0, line.find(" "));
                }
                catch (std::exception e) {
                    std::cout << "Error parsing GT file for " << current_name << " (maybe wrong format?)" << std::endl;
                    std::cout << "Exception details:" << e.what() << std::endl;
                    return false;
                }

                try {


                    
                    //float x_2 = x_1 + (std::stof(w) * 1280);
                    //float y_2 = y_1 + (std::stof(h) * 720);
                    float x_2 = (std::stof(w) * 1280);
                    float y_2 = (std::stof(h) * 720);
                    float x_1 = (std::stof(x1) * 1280) - (x_2/2);
                    if (x_1 <= 0)
                        x_1 = 0;
                    
                    float y_1 = (std::stof(y1) * 720) - (y_2/2) ;
                    if (y_1 <= 0)
                        y_1 = 0;

                    x_2 += x_1;
                    if (x_2 > 1280)
                        x_2 = 1280;
                    
                    y_2 += y_1;
                    if (x_2 > 720)
                        x_2 = 720;
                    
                    //std::cout << "X1 : " << x_1 << " X2: " << x_2 << " Y1: " << y_1 << " Y2: " << y_2 << std::endl;
                    
                    
                    rangex.push_back(cv::Range(static_cast<int>(x_1), static_cast<int>(x_2)));
                    rangey.push_back(cv::Range(static_cast<int>(y_1), static_cast<int>(y_2)));
                }
                catch (std::exception e) {
                    std::cout << "Error accepting coordinates from GT file for " << current_name << std::endl;
                    //std::cout << "I parsed: " << x1 << " " << x2 << " " << y1 << " " << y2 << std::endl;
                    std::cout << "Exception details:" << e.what() << std::endl;
                    return false;
                }
            }

            std::cout << std::endl<< current_name << std::endl;

            //crop out and process positive samples
            for (int j = 0; j < rangex.size(); j++) {
                //crop out boat

               
                cv::Mat boat = current(rangey.at(j), rangex.at(j));


                cv::GaussianBlur(boat, boat, cv::Size(3, 3), 5);
                //cv::imshow("boat",boat);
                //cv::waitKey();

                //process boat
                std::vector<cv::KeyPoint> keypoints;
                cv::Mat descriptors;
                sift->detectAndCompute(boat, cv::Mat(), keypoints, descriptors);

                //show keypoints?
                //cv::Mat annotated;
                //cv::drawKeypoints(boat, keypoints, annotated);
                //cv::imshow("Keypoints for boat " + std::to_string(j) + " in " + current_name, annotated);
                //cv::waitKey();

                //make into dictionary words
                bow_extractor->compute(boat, keypoints, descriptors);

#pragma omp critical
                {
                    training_data.push_back(descriptors);
                    std::vector<int> newlabels(descriptors.rows, 1);
                    labels.insert(labels.end(), newlabels.begin(), newlabels.end());
                }
            }

            //NEGATIVE SAMPLES
            //build mask
            cv::Mat mask = cv::Mat::ones(current.rows, current.cols, CV_8U) * 255;
            for (int j = 0; j < rangex.size(); j++)
                mask(cv::Range(rangey.at(j)), cv::Range(rangex.at(j))) = 0;
            //run detection
            std::vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            cv::Mat negative;
            cv::GaussianBlur(current, negative, cv::Size(3, 3), 5);
            sift->detectAndCompute(negative, mask, keypoints, descriptors);

            //show keypoints?
            //cv::Mat annotated;
            //cv::drawKeypoints(current, keypoints, annotated);
            //cv::imshow("Keypoints for negative in " + current_name, annotated);
            //cv::waitKey();

            //make into dictionary words
            bow_extractor->compute(negative, keypoints, descriptors);

#pragma omp critical
            {
                training_data.push_back(descriptors);
                std::vector<int> newlabels(descriptors.rows, 0);
                labels.insert(labels.end(), newlabels.begin(), newlabels.end());
            }

            //std::cout << current_name << " completed" << std::endl;

            std::cout << std::fixed << std::setprecision(2) << (float)i / dataset_paths.size() * 100 << "% completed" << '\r';

            //cv::destroyAllWindows();
        }

        std::cout << "Done." << std::endl << "Training SVM..." << std::endl;

        //cv::imwrite("svm_training_data", training_data);

        //SVM: train
        cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
        svm->setType(cv::ml::SVM::Types::NU_SVC);
        //svm->setType(cv::ml::SVM::ONE_CLASS);
        svm->setKernel(cv::ml::SVM::RBF);
        svm->setNu(0.0001); //needs to be cross validated, nu = [0,1]

        // default might be fine
        //svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 500, FLT_EPSILON)); //FLT_EPS vs 1e-6

        svm->train(training_data, cv::ml::ROW_SAMPLE, labels); //std::vector<int>(training_data.rows, 1));//std::vector<int>(CLASS_COUNT, 1));
        svm->save(output_path);

        //done?
        std::cout << "Done." << std::endl;
    }
    catch (cv::Exception e) {
        std::cout << "Error training SVM: " << e.what() << std::endl;
        return 1;
    }
}

bool Trainer::loadDatasetPaths(std::string dataset_folder) {
    try {
        dataset_root_path = dataset_folder;
        cv::String path(dataset_root_path + /*"\\" + IMG_FOLDER + "\\" /* + FILE_PREFIX + */ "*" + IMG_EXTENSION);
        cv::glob(path, dataset_paths, false);
        std::cout << dataset_paths.size() << " files found with pattern " << path << std::endl;
        std::cout << "starts with " << dataset_paths.at(0) << std::endl;
        std::cout << "ends with " << dataset_paths.at(dataset_paths.size() - 1) << std::endl;
    }
    catch (cv::Exception e) {
        std::cout << "Error loading dataset: " << e.what() << std::endl;
        return false;
    }
}

bool Trainer::loadDictionary(std::string dictionary_path) {
    try {
        cv::FileStorage fs(dictionary_path, cv::FileStorage::READ);
        fs["dictionary"] >> dictionary;
        fs.release();
        return true;
    }
    catch (cv::Exception e) {
        std::cout << "Error loading dictionary: " << e.what() << std::endl;
        return false;
    }
}