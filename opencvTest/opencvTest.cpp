
#include<opencv2/opencv.hpp>
#include<iostream>
#include <vector>

using namespace std;
using namespace cv;


void smoothing(const Mat & img)
{
	namedWindow("image_smoothed", WINDOW_NORMAL);
	Mat smoothed_img;
	GaussianBlur(img, smoothed_img, Size(5, 5), 3, 3);
	
	imshow("image_smoothed", smoothed_img);
	waitKey(0);
	
}

void show_info(const Mat &img)
{
	cout << "depth : " << img.depth() << endl;
	cout << "columns : " << img.cols << endl;
	cout << "rows : " << img.rows << endl;
	cout << "dim : " << img.dims << endl;
	cout << "size : " << img.size << endl;
}

void show_pixels_intensity_bgr(const Mat &img)
{
	for (int i = 0; i < img.rows; ++i)
	{
		for (int j = 0; j < img.cols; ++j)
		{
			Vec3b pixel = img.at<Vec3b>(i, j);
			cout << (pixel.val[0] + pixel.val[1] + pixel.val[2]) / 3 << " ";
		}
		cout << endl;
	}
}

void show_pixels_intensity_one_channel(const Mat& img)
{
	for (int i = 0; i < img.rows; ++i)
	{
		for (int j = 0; j < img.cols; ++j)
		{
			int pixel = img.at<uchar>(i, j);
			cout << pixel<< " ";
		}
		cout << endl;
	}
}

void tp_ex1(const Mat &img)
{
	namedWindow("calcHist Demo", WINDOW_NORMAL);
	
	Mat resized_img;

	resize(img, resized_img, Size(5, 5));

	//histogram with calcHist

	vector<Mat> bgr_planes;
	split(resized_img, bgr_planes);

	auto hist_size = 256;
	float range[] = { 0, 256 };
	const float* hist_range = { range };
	auto uniform = true, accumulate = false;

	Mat b_hist, g_hist, r_hist;
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &hist_size, &hist_range, uniform, accumulate);
	
	//show_info(b_hist);
	show_pixels_intensity_one_channel(bgr_planes[0]);
	
	for (int i = 0; i < 256; ++i)
	{
		if (b_hist.at<float>(i))
			cout << i << " : " << b_hist.at<float>(i) << endl;
	}

	//histogram manual calc
	vector<int> my_b_hist(256, 0);
	
	for (int i = 0; i < bgr_planes[0].rows; ++i)
	{
		for (int j = 0; j < bgr_planes[0].cols; ++j)
		{
			int pixel = bgr_planes[0].at<uchar>(i, j);
			my_b_hist[pixel]++;
		}
	}

	for (int i = 0; i < 256; i++)
		if (my_b_hist[i])
			cout << i << " : " << my_b_hist[i] << endl;


	//plot hist

	int hist_w = 512, hist_h = 400;
	int bin_w = cvRound((double)hist_w / hist_size);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < hist_size; i++)
	{
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}
	
	imshow("calcHist Demo", histImage);
	waitKey();
	
}

//create image from matrix


Mat img_create(const vector<vector<int>> &matrix, bool kernel)
{
	const int rows = matrix.size();
	const int cols = matrix[0].size();

	int type = CV_32S;
	
	Mat img(rows, cols, type, Scalar(0));

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; ++j)
		{
				img.at<int>(i, j) = matrix[i][j];
		}
	}


	return img;
}

Mat img_create(const vector<vector<float>>& matrix)
{
	const int rows = matrix.size();
	const int cols = matrix[0].size();

	int type = CV_32F;

	Mat img(rows, cols, type, Scalar(0));

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; ++j)
		{
				img.at<float>(i, j) = matrix[i][j];
		}
	}


	return img;
}

float luminance_one_channel(const Mat &img)
{
	float luminance = 0;
	
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; ++j)
			luminance += img.at<uchar>(i, j);

	return luminance / (img.rows * img.cols * 1.0);
}

float contrast_one_channel(const Mat& img)
{
	const auto luminance = luminance_one_channel(img);
	float variance = 0;
	
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; ++j)
			variance += (img.at<uchar>(i, j) - luminance) * (img.at<uchar>(i, j) - luminance);

	//variance 
	variance /= (img.rows * img.cols);
	
	return sqrt(variance);
}

//BORDER_REPLICATE pour refaire, BORDER_CONSTAT ajoute des 0
Mat linear_filter(const Mat& img, const Mat& kernel, BorderTypes border)
{
	Mat res;
	img.copyTo(res);

	rotate(kernel, kernel, ROTATE_90_COUNTERCLOCKWISE);
	rotate(kernel, kernel, ROTATE_90_COUNTERCLOCKWISE);
	filter2D(img, res, -1, kernel, Point(-1, -1), 0, border);

	//show_pixels_intensity_one_channel(res);

	return res;
}

Mat seuil_mat(Mat img, Mat origin, int seuil)
{
	Mat res;
	img.copyTo(res);
	
	for (int i = 0; i < res.rows; i++)
	{
		for (int j = 0; j < res.cols; ++j)
		{
			switch (res.depth())
			{
			case CV_8U:
			{
				if (res.at<uchar>(i, j) <= seuil)
					res.at<uchar>(i, j) = origin.at<uchar>(i, j);
				break;
			}
			case CV_8S:
			{
				if (res.at<schar>(i, j) <= seuil)
					res.at<schar>(i, j) = origin.at<schar>(i, j);
				break;
			}
			case CV_16U:
			{
				if (res.at<ushort>(i, j) <= seuil)
					res.at<ushort>(i, j) = origin.at<ushort>(i, j);
				break;
			}
			case CV_16S:
			{
				if (res.at<short>(i, j) <= seuil)
					res.at<short>(i, j) = origin.at<short>(i, j);
				break;
			}
			case CV_32S:
			{
				if (res.at<int>(i, j) <= seuil)
					res.at<int>(i, j) = origin.at<int>(i, j);
				break;
			}
			case CV_32F:
			{
				if (res.at<float>(i, j) <= seuil)
					res.at<float>(i, j) = origin.at<float>(i, j);
				break;
			}
			case CV_64F:
			{
				if (res.at<double>(i, j) <= seuil)
					res.at<double>(i, j) = origin.at<double>(i, j);
				break;
			}

			}
		}
	}

	return res;
}


void print_matrix(const Mat &mat)
{
	for (int r = 0; r < mat.rows; r++) {
		for (int c = 0; c < mat.cols; c++) {

			switch (mat.depth())
			{
			case CV_8U:
			{
				printf("%*u ", 3, mat.at<uchar>(r, c));
				break;
			}
			case CV_8S:
			{
				printf("%*hhd ", 4, mat.at<schar>(r, c));
				break;
			}
			case CV_16U:
			{
				printf("%*hu ", 5, mat.at<ushort>(r, c));
				break;
			}
			case CV_16S:
			{
				printf("%*hd ", 6, mat.at<short>(r, c));
				break;
			}
			case CV_32S:
			{
				printf("%*d ", 6, mat.at<int>(r, c));
				break;
			}
			case CV_32F:
			{
				printf("%*.4f ", 10, mat.at<float>(r, c));
				break;
			}
			case CV_64F:
			{
				printf("%*.4f ", 10, mat.at<double>(r, c));
				break;
			}
			}
		} printf("\n");
	} printf("\n");
}



int main()
{
	//Mat const img = imread("aces.jpg", IMREAD_COLOR);
	//if (img.empty())
	//	return -1;
	//
	//namedWindow("image", WINDOW_NORMAL);
	//imshow("image", img);

	//smoothing(img);
	//waitKey(0);
	//
	//destroyAllWindows();

	//tp_ex1(img);

	vector<vector<int>> matrix = {
		{100, 100, 50, 50, 200},
		{100, 50, 50, 50, 200},
		{100, 200, 200, 200, 200},
		{100, 200, 200, 200, 200}
	};

	vector<vector<float>> matrix5 = {
		{100, 100, 100, 100},
		{100, 100, 150, 100},
		{200, 100, 100, 100},
		{200, 100, 100, 200},
	};

	vector<vector<float>> matrix2 = {
		{1,2, 3},
		{4, 5, 6},
		{7, 8, 9}
	};

	vector<vector<int>> matrix4 = {
		{50, 100, 200},
		{50, 100, 200},
		{50, 100, 200}
	};
	
	vector<vector<int>> matrix3= {
		{1, 1, 1, 5},
		{1, 1, 5, 5},
		{1, 5, 5, 8},
		{5, 5, 8, 10}
	};

	vector<vector<float>> kernel = {
		{-1,-2, -1},
		{0, 0, 0},
		{1, 2, 1}
	};

	vector<vector<float>> median8 = {
		{1/9, 1/9, 1/9},
		{1 / 9, 1 / 9, 1 / 9},
		{1 / 9, 1 / 9, 1 / 9}
	};
	vector<vector<float>> median4 = {
		{0, 1.0/5, 0},
		{1.0/5, 1.0/5, 1.0/5},
		{0, 1.0/5, 0}
	};

	vector<vector<float>> laplace4 = {
		{0, 1, 0},
		{1, -4, 1},
		{0, 1, 0}
	};


	vector<vector<float>> laplace8 = {
		{1, 1, 1},
		{1, -8, 1},
		{1, 1, 1}
	};

	const auto img = img_create(matrix5);
	const auto ker = img_create(laplace8);
	
	/*cout << luminance_one_channel(img) << endl;
	cout << contrast_one_channel(img) << endl;*/

	
	const auto filtered_img = linear_filter(img, ker,	BORDER_CONSTANT);

	std::cout << "\nimg ============= " << endl;
	print_matrix(img);
	std::cout << "\nkernel ============= " << endl;
	print_matrix(ker);
	
	std::cout << "\nfilter result ============= " << endl;
	print_matrix(filtered_img);

	std::cout << "\nthresholded result  ============= " << endl;
	Mat seuiled_mat = seuil_mat(filtered_img, img, 120);
	print_matrix(seuiled_mat);
	//Mat median_blured_img;
	//medianBlur(img, median_blured_img, 3);

	//cout << "\nmedian ============= " << endl;
	//print_matrix(median_blured_img);

	

	
	return 0;
}


/* erosion : we take a structuring element(can be a 3x3 matrix for example), then we put it on each pixel of the picture, we keep only the elements
p such that p && b is true, for every b in the structuring element */
