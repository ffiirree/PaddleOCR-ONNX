#include <onnxruntime_cxx_api.h>
#include <assert.h>
#include <opencv2/opencv.hpp>

int main(int argc, char * argv[])
{
    cv::Mat image;
    cv::imread("../water_ouzel.jpeg").convertTo(image, CV_32F, 1 / 255.0f);

    image = (image - cv::Scalar{ 0.406, 0.456, 0.485 }) / cv::Scalar{ 0.225, 0.224, 0.229 };

    cv::Mat img_channels[3];
	cv::split(image, img_channels);
    
    std::memcpy(image.data, img_channels[2].data, 224 * 224 * 4);
    std::memcpy(image.data + 224 * 224 * 4, img_channels[1].data, 224 * 224 * 4);
    std::memcpy(image.data + 224 * 224 * 8, img_channels[0].data, 224 * 224 * 4);

    Ort::Env env;
    Ort::Session session(env, "../mobilenetv3s.onnx", Ort::SessionOptions{nullptr});


    size_t input_tensor_size = 224 * 224 * 3;
    std::vector<int64_t> input_shape{ 1, 3, 224, 224 };
    std::vector<const char*> input_node_names = {"input"};
    std::vector<const char*> output_node_names = {"output"};

    // create input tensor object from data values
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)image.data, input_tensor_size, input_shape.data(), 4);
    assert(input_tensor.IsTensor());

    // score model & input tensor, get back output tensor
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
    assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

    // Get pointer to output tensor float values
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    // assert(abs(floatarr[0] - 0.000045) < 1e-6);

    auto result_ = std::distance(floatarr, std::max_element(floatarr, floatarr + 999));
    printf("classification target(water_ouzel-20): %ld\n", result_);

    return 0;
}