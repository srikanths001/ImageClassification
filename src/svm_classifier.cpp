//Train a Bag of Words based SVM classifier to classify images.
//Needs OpenCV 3+

#include <stdio.h>
#include <iostream>
#include <vector>

//#include <opencv2/ml/ml.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/xfeatures2d.hpp>
#include "opencv2/highgui.hpp"
#include <opencv2/ml.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv; 
using namespace std;
using namespace cv::xfeatures2d;
using namespace cv::ml;

char ch[30];

//--------Using SURF as feature extractor and Flann based matching for assigning a new point to the nearest one in the dictionary
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
Ptr<SURF> detector = SURF::create(500);
//---dictionary size=number of cluster's centroids
int dictionarySize = 1500;
TermCriteria tc(CV_TERMCRIT_ITER, 10, 0.001);
int retries = 1;
int flags = KMEANS_PP_CENTERS;
BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
BOWImgDescriptorExtractor bowDE(detector, matcher);

void collectclasscentroids() 
{
	int i,j;
	//The example dataset contains ordered images of 4 classes, with each class containing 60 images.
	for(j=1;j<=4;j++)
	{
		for(i=1;i<=60;i++)
		{
			sprintf( ch,"%s%d%s%d%s","train/",j," (",i,").jpg");
			const char* imageName = ch;
			std::string imgName1 = imageName;
			cout<<"name of img: "<<imgName1<<endl;
			Mat img;
			//img = cvLoadImage(imageName,0);
			img = imread(imageName,IMREAD_GRAYSCALE);
			if(!img.data){
			cout<<"no data\n";return;}
			vector<KeyPoint> keypoint;
			detector->detect(img, keypoint);
			Mat features;
			detector->compute(img, keypoint, features);
			cout<<"Features size: "<<features.size()<<endl;
			bowTrainer.add(features);
		}
	}
	return;
}

int main( int argc, char** argv )
{
	int i,j;
	Mat img2;
	cout<<"Vector quantization..."<<endl;
	collectclasscentroids();
	cout<<"Done..."<<endl;
	std::vector<Mat> descriptors = bowTrainer.getDescriptors();
	int count=0;
	for(vector<Mat>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)
	{
		count+=iter->rows;
	}
	int descr_count = bowTrainer.descriptorsCount();
	cout<<"Clustering "<<count<<" features and vec Descriptor size: "<<descriptors.size()<<" Descr_count: "<<descr_count<<endl;
	//choosing cluster's centroids as dictionary's words
	Mat dictionary = bowTrainer.cluster();
	bowDE.setVocabulary(dictionary);
	cout<<"extracting histograms in the form of BOW for each image "<<endl;
	Mat labels(0, 1, CV_32FC1);
	Mat trainingData(0, dictionarySize, CV_32FC1);
	int k=0;
	vector<KeyPoint> keypoint1;
	Mat bowDescriptor1;
	//extracting histogram in the form of bow for each image
	//The example dataset contains ordered images of 4 classes, with each class containing 60 images.
	for(j=1;j<=4;j++)
	{
		for(i=1;i<=60;i++){
			sprintf( ch,"%s%d%s%d%s","train/",j," (",i,").jpg");
			const char* imageName = ch;
			//std::string imgName2 = imageName;
			//img2 = cvLoadImage(imageName,0);
			img2 = imread(imageName, IMREAD_GRAYSCALE);
			detector->detect(img2, keypoint1);

			bowDE.compute(img2, keypoint1, bowDescriptor1);
			trainingData.push_back(bowDescriptor1);
			labels.push_back((int) j);
		}
	}
		
	cout<<"Size of training data: "<<trainingData.size()<<endl;

	//Setting up SVM parameters
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	//Optimized params C=312.5 for RBF kernel
	svm->setC(312.50000000000000);//0.1
	svm->setKernel(SVM::RBF);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, (int)1e7, 1e-6));

	printf("%s\n","Training SVM classifier");

	//bool res=svm.train(trainingData,labels,cv::Mat(),cv::Mat(),params);
	cout<<"Training data size: "<<trainingData.size()<<" . Labels size: "<<labels.size()<<endl;
	//cout<<labels<<endl;
	svm->train(trainingData, ROW_SAMPLE, labels);

	cout<<"Processing evaluation data..."<<endl;

	Mat groundTruth(0, 1, CV_32FC1);
	Mat evalData(0, dictionarySize, CV_32FC1);
	k=0;
	vector<KeyPoint> keypoint2;
	Mat bowDescriptor2;

	Mat results(0, 1, CV_32FC1);;
	for(j=1;j<=4;j++)
	{
		for(i=1;i<=60;i++)
		{			
			sprintf( ch,"%s%d%s%d%s","eval/",j," (",i,").jpg");
			const char* imageName = ch;
			img2 = imread(imageName,IMREAD_GRAYSCALE);
		
			detector->detect(img2, keypoint2);
			bowDE.compute(img2, keypoint2, bowDescriptor2);
			
			evalData.push_back(bowDescriptor2);
			groundTruth.push_back((float) j);
			float response = svm->predict(bowDescriptor2);
			results.push_back(response);
		}
	}

	//calculate the number of unmatched classes 
	double errorRate = (double) countNonZero(groundTruth- results) / evalData.rows;
	printf("%s%f","Error rate is \n",errorRate);
	return 0;
}


