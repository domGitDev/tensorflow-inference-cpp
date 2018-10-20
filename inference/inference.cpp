/*
	command args: --graph_file=../logs/exported/graph.pb 
				  --checkpoint_file=../logs/exported/my_model
				  --meta_file=../logs/exported/my_model.meta
*/
#include "inference.h"
#include <experimental/filesystem>

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;
using tensorflow::int64;

using namespace tensorflow;
using namespace ::tensorflow::ops;


// command line args
string graph_file = "../logs/exported/graph.pb";
string checkpoint_file = "../logs/exported/my_model";
string meta_file = "../logs/exported/my_model.meta";


int OCRInference::WIDTH = 128;
int OCRInference::HEIGHT = 64;
string OCRInference::CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-";
tensorflow::Scope OCRInference::root = tensorflow::Scope::NewRootScope();

OCRInference::OCRInference()
{
	
}


Status OCRInference::initGraph()
{
	tensorflow::GraphDef graph_model;
	Status status = root.ToGraphDef(&graph_model);
	if (!status.ok()) {
		LOG(ERROR) << status.ToString() << std::endl;;
		return status;
	}

	status = session->Create(graph_model);
	if (!status.ok()) {
		LOG(ERROR) << status.ToString() << std::endl;;
		return status;
	}
	return Status::OK();
}

Status OCRInference::loadGraph(string filename)
{
	tensorflow::GraphDef graph_def;
	auto status = ReadBinaryProto(Env::Default(), filename, &graph_def);
	if (!status.ok()) {
		LOG(ERROR) << status.ToString() << std::endl;;
		return status;
	}

	status = session->Create(graph_def);
	if (!status.ok()) {
		LOG(ERROR) << status.ToString() << std::endl;
		return status;
	}
	TF_CHECK_OK(session->Run({}, {}, { "init" }, nullptr));
	return Status::OK();
}

Status OCRInference::restoreCheckpoint(string checkpoit_dir, string meta_file)
{
	Status status;
	
	tensorflow::MetaGraphDef graph_def;
	status = ReadBinaryProto(Env::Default(), meta_file, &graph_def);
	if (!status.ok()) {
		LOG(ERROR) << status.ToString() << std::endl;
		return status;
	}
	
	root.ToGraphDef(graph_def.mutable_graph_def());
	status = session->Extend(*graph_def.mutable_graph_def());
	if (!status.ok()) {
		LOG(ERROR) << status.ToString() << std::endl;
		return status;
	}

	// Add the graph to the session
	// Read weights from the saved checkpoint
	Tensor checkpointPathTensor(DT_STRING, TensorShape());
	checkpointPathTensor.scalar<std::string>()() = checkpoit_dir;
	status = session->Run(
		{ { graph_def.mutable_saver_def()->filename_tensor_name(), checkpointPathTensor }, },
		{},
		{ graph_def.mutable_saver_def()->restore_op_name() },
		nullptr);
	if (!status.ok()) {
		LOG(ERROR) << status.ToString() << std::endl;
		return status;
	}
	
	return Status::OK();
}

Status OCRInference::initSession()
{
	Status status;

	tensorflow::Session* session_pointer = nullptr;
	tensorflow::SessionOptions options;
	status = tensorflow::NewSession(options, &session_pointer);
	if (!status.ok()) {
		LOG(ERROR) << status.ToString() << std::endl;;
		return status;
	}
	
	session.reset(session_pointer);
	root = tensorflow::Scope::NewRootScope();
	return Status::OK();
}

Status OCRInference::closeSession()
{
	TF_RETURN_IF_ERROR(session->Close());
	return Status::OK();
}


Status OCRInference::ReadEntireFile(tensorflow::Env* env, const string& filename, Tensor* output) {
	tensorflow::uint64 file_size = 0;
	TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

	string contents;
	contents.resize(file_size);

	std::unique_ptr<tensorflow::RandomAccessFile> file;
	TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

	tensorflow::StringPiece data;
	TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
	if (data.size() != file_size) {
		return tensorflow::errors::DataLoss("Truncated read of '", filename,
			"' expected ", file_size, " got ",
			data.size());
	}
	output->scalar<string>()() = data.ToString();
	return Status::OK();
}


Status OCRInference::ImageTensorFromFile(
	const int input_height=HEIGHT, const int input_width=WIDTH)
{
	const int wanted_channels = 3;
	tensorflow::Output image_reader;
	root = tensorflow::Scope::NewRootScope();

	auto file_reader = Placeholder(root.WithOpName("input_image"), tensorflow::DataType::DT_FLOAT);
	auto img_reshape = Reshape(root.WithOpName("img_reshape"), file_reader, { input_height, input_width, 3 });
	auto dims_expander = ExpandDims(root.WithOpName("image_batch"), img_reshape, 0);

	return Status::OK();
}


Tensor OCRInference::FloatTensorFromMat(const cv::Mat& img)
{
	cv::Mat temp;
	//cv::cvtColor(img, temp, CV_BGR2RGB);
	img.convertTo(temp, CV_32FC3);
	Tensor input_img(tensorflow::DT_FLOAT, tensorflow::TensorShape({ temp.cols, temp.rows, temp.channels() }));

	auto *p = input_img.flat<float>().data();
	memcpy(p, temp.data, sizeof(CV_32FC3) * temp.cols * temp.rows * temp.channels());

	return input_img;
}

Status  OCRInference::runInference(const cv::Mat& img)
{
	auto input = FloatTensorFromMat(img);

	std::vector<Tensor> img_outputs(1);
	std::vector<std::pair<string, Tensor>> img_inputs = {
		{ "input_image", input }
	};

	auto status = session->Run({ img_inputs }, { "image_batch" }, {}, {&img_outputs });
	if (!status.ok()) {
		LOG(ERROR) << status.ToString() << std::endl;
		return status;
	}

	std::vector<Tensor> outputs(2);
	std::vector<std::pair<string, Tensor>> inputs = {
		{ "input_x", img_outputs[0]}
	};

	status = session->Run({ inputs }, { "length_predictions", "digits_predictions" }, {}, &outputs);
	if (!status.ok()) {
		LOG(ERROR) << status.ToString() << std::endl;
		return status;
	}

	auto length = outputs[0].scalar<tensorflow::int64>();
	auto digits = outputs[1].flat<tensorflow::int64>().data();

	std::stringstream ss;	
	for (int i = 0; i < length(); i++)
	{
		ss << CHARS.at(digits[i]);
	}
	std::cout << "Length: " << length << std::endl; 
	std::cout << "Number: " << ss.str() << std::endl;
	
	cv::Mat mat(
		img_outputs[0].dim_size(1), img_outputs[0].dim_size(2),
		CV_32FC3, img_outputs[0].flat<float>().data());
	mat.convertTo(mat, CV_8UC3);

	cv::namedWindow("inference", cv::WINDOW_AUTOSIZE);
	cv::imshow("inference", mat);
	cv::waitKey();
	cv::destroyAllWindows();

	return Status::OK();
}


int main(int argc, char** argv)
{
	// Define argument list
	std::vector<Flag> flag_list = {
		Flag("graph", &graph_file, "Path to load graph data."),
		Flag("checkpoint", &checkpoint_file, "Path to restore checkpoint."),
		Flag("meta", &meta_file, "Path to load graph meta data."),
	};

	if (argc > 1)
	{
		string usage = tensorflow::Flags::Usage(argv[0], flag_list);
		const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);

		if (!parse_result) {
			LOG(ERROR) << usage << std::endl;;
			system("pause");
			return -1;
		}
	}

	// We need to call this to set up global state for TensorFlow.
	tensorflow::port::InitMain(argv[0], &argc, &argv);

	OCRInference* instance = new OCRInference();
	auto status = instance->initSession();
	{
		if (!status.ok()) {
			LOG(ERROR) << status.ToString() << std::endl;;
			return -1;
		}
	}
	
	instance->loadGraph(graph_file);
	instance->ImageTensorFromFile(OCRInference::HEIGHT, OCRInference::WIDTH);
	instance->restoreCheckpoint(checkpoint_file, meta_file);

	std::string image_path;
	/*namespace stdfs = std::experimental::filesystem;
	const stdfs::directory_iterator end{};
	
	for (stdfs::directory_iterator iter{ "../testImages" }; iter != end; ++iter)
	{
		if (stdfs::is_regular_file(*iter))
		{
			cv::Mat img = cv::imread(iter->path().string().c_str());
			instance->runInference(img);
			std::cout << std::endl;
		}
	}*/
	do
	{
		std::cout << "Image Path: ";
		std::cin >> image_path;
		if (image_path.length() > 0)
		{
			cv::Mat img = cv::imread(image_path);
			instance->runInference(img);
		}
		std::cout << std::endl;

	} while (image_path != "-1");
	
	instance->closeSession();

	system("pause");
	return 0;
}