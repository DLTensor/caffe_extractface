#include <stdio.h>  // for snprintf
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"
#include "leveldb/db.h"
#include "leveldb/write_batch.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

// liu 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>  // for mkdir

using namespace caffe;  // NOLINT(build/namespaces)

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);

int main(int argc, char** argv) {
  return feature_extraction_pipeline<float>(argc, argv);
//  return feature_extraction_pipeline<double>(argc, argv);
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  const int num_required_args = 5;
  if (argc < num_required_args) {
    LOG(ERROR)<<
    "This program takes in a trained network and an input data layer, and then"
    " extract features of the input data produced by the net.\n"
    "Usage: test_det_net  pretrained_net_param"
    "  feature_extraction_proto_file num_mini_batches"
		"  output_dir"
    "  [CPU/GPU] [DEVICE_ID=0]\n"
    "Note: the feature blob names is fixed as 'fc_8_det' in code\n";
    return 1;
  }
  int arg_pos = num_required_args;

  arg_pos = num_required_args;
  if (argc > arg_pos && strcmp(argv[arg_pos], "GPU") == 0) {
    LOG(ERROR)<< "Using GPU";
    uint device_id = 0;
    if (argc > arg_pos + 1) {
      device_id = atoi(argv[arg_pos + 1]);
      CHECK_GE(device_id, 0);
    }
    LOG(ERROR) << "Using Device_id=" << device_id;
    Caffe::SetDevice(device_id);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }
  Caffe::set_phase(Caffe::TEST);

  arg_pos = 0;  // the name of the executable
  string pretrained_binary_proto(argv[++arg_pos]);

  string feature_extraction_proto(argv[++arg_pos]);
  shared_ptr<Net<Dtype> > feature_extraction_net(
      new Net<Dtype>(feature_extraction_proto));
  feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);
	// to get image_paths
	const vector<shared_ptr<Layer<float> > > layers = feature_extraction_net->layers();
	const caffe::ImageDataLayer<float> *image_layer = dynamic_cast<caffe::ImageDataLayer<float>* >(layers[0].get());
	CHECK(image_layer);
			
  const string blob_name = "fc_8_det";
  
  CHECK(feature_extraction_net->has_blob(blob_name))   \
		<< "Unknown feature blob name " << blob_name      \
		<< " in the network " << feature_extraction_proto;


  int num_mini_batches = atoi(argv[++arg_pos]);
	string output_dir = argv[++arg_pos];
	CHECK_EQ(mkdir(output_dir.c_str(),0744), 0) << "mkdir " << output_dir << " failed";

  LOG(ERROR)<< "Extracting Features";

  vector<Blob<float>*> input_vec;
  int image_index=0;
  for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
    feature_extraction_net->Forward(input_vec);
		
		const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net->blob_by_name(blob_name);
	
		int batch_size = feature_blob->num();
		
		int dim_features = feature_blob->count() / batch_size;
		CHECK_EQ(dim_features, 4) << "the dim of feature is not equal to 4";
		
		Dtype* feature_blob_data;
		int x1, y1, x2, y2;
		for (int n = 0; n < batch_size; ++n) {
			feature_blob_data = feature_blob->mutable_cpu_data() + feature_blob->offset(n);

			x1 = feature_blob_data[0];
			y1 = feature_blob_data[1];
			x2 = feature_blob_data[2];
			y2 = feature_blob_data[3];
			
			string image_path	= image_layer->lines_[image_index].first;
			//LOG(ERROR) << "image_index " << image_index << " " <<  image_path   \
								 << " x1 " << feature_blob_data[0] << " y1 " << feature_blob_data[1] \
								 << " x2 " << feature_blob_data[2] << " y2 " << feature_blob_data[3];
			
			cv::Mat img_origin = cv::imread(image_path);
			
			std::vector<string> part_names;
			boost::split(part_names, image_path, boost::is_any_of("/"));
			string subname = part_names[part_names.size()-1];             // the last element is the image name.
			string out_path(output_dir + "/" + subname);
			
			//LOG(ERROR) << subname;
			line(img_origin, cv::Point(x1, y1), cv::Point(x2, y1), cv::Scalar(0, 0, 255), 3);
			line(img_origin, cv::Point(x2, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 3);
			line(img_origin, cv::Point(x2, y2), cv::Point(x1, y2), cv::Scalar(0, 0, 255), 3);
			line(img_origin, cv::Point(x1, y2), cv::Point(x1, y1), cv::Scalar(0, 0, 255), 3);
			CHECK_EQ(imwrite(output_dir + "/" + subname, img_origin), true) << "write image " + out_path + " failed";
			
			image_index ++ ;
			if (image_index>=image_layer->lines_.size()){
				LOG(ERROR) << "Restarting data prefetching from start.";
				image_index = 0;
			}
			// (image_index>image_layer->lines_.size()-1)?(image_index=0):(image_index++);
		}
		
  }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)
  // write the last batch
  
  LOG(ERROR)<< "Successfully extracted the features!";
  return 0;
}

