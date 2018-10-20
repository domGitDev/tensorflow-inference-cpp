#pragma once
#include <vector>
#include <fstream>
#include <utility>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <tensorflow/cc/ops/const_op.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/ops/image_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/platform/logging.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/platform/types.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/util/command_line_flags.h>

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;
using namespace ::tensorflow::ops;


class OCRInference
{
public:
	OCRInference();

	Status initGraph();
	Status loadGraph(string filename);
	Status initSession();
	Status closeSession();
	Status restoreCheckpoint(string checkpoint_file, string meta_file);
	static Status ReadEntireFile(
		tensorflow::Env* env, const string& filename, Tensor* output);
	Status ImageTensorFromFile(
		const int input_height, const int input_width);

	static Tensor FloatTensorFromMat(const cv::Mat& img);
	Status runInference(const cv::Mat& img);	

	static int WIDTH;
	static int HEIGHT;
	static string CHARS;
	static tensorflow::Scope root;
private:
	std::unique_ptr<tensorflow::Session> session;
	//tensorflow::GraphDef graph;
};