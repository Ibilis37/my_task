#include <torch/torch.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <vector>

using namespace std;
class Box {
public:
    int x1, y1, x2, y2;
    float confid;
    int name_index;
    Box(int x1, int y1, int x2, int y2, float confid, int name_index) {
        this->x1 = x1;
        this->y1 = y1;
        this->x2 = x2;
        this->y2 = y2;
        this->confid = confid;
        this->name_index = name_index;
    }
};

torch::jit::script::Module LoadModel(const std::string& model_path) {
    torch::jit::script::Module model = torch::jit::load(model_path);
    return model;
}

torch::Tensor TransformImage(const cv::Mat& image) {
    cv::Mat image_float;
    image.convertTo(image_float, CV_32F, 1.0 / 255); // normalization
    cvtColor(image_float, image_float, cv::COLOR_BGR2RGB);

    auto tensor_image = torch::from_blob(image_float.data, { 1, 640, 640, 3 }, torch::kF32);
    tensor_image = tensor_image.permute({ 0, 3, 1, 2 }); //  HxWxC to CxHxW.

    return tensor_image.clone();
}

float IntersectionOverUnion(Box& fb, Box& sb) {
    float bx_intersect = max(min(fb.x2, sb.x2) - min(fb.x1, sb.x1), 0) * max(min(fb.y2, sb.y2) - min(fb.y1, sb.y1), 0);
    float bx_union = (fb.x2 - fb.x1) * (fb.y2 - fb.y1) + (sb.x2 - sb.x1) * (sb.y2 - sb.y1) - bx_intersect;
    return bx_intersect / bx_union;
}

vector<Box> NonMaxSuppression(vector<Box>& boxes, float iou_threshold) {
    vector<Box> supBoxes;
    for (Box box : boxes) {
        bool valid = true;
        for (Box supBox : supBoxes) {
            if (IntersectionOverUnion(box, supBox) > iou_threshold) {
                valid = false;
                break;
            }
        }
        if (valid == true) {
            supBoxes.push_back(box);
        }
    }
    return supBoxes;
}


vector<Box> GetBoxes(at::Tensor& outputs, float confid_threshold = 0.35, float iou_threshold = 0.25)
{
    vector<Box> candidates;
    for (unsigned short ibatch = 0; ibatch < outputs.sizes()[0]; ibatch++) {
        for (unsigned short ibox = 0; ibox < outputs.sizes()[2]; ibox++) {
            // defining the class of the detected object
            float confid, p_max = 0;
            int iclass = 0;
            size_t sz = outputs[ibatch].size(0);
            for(int i = 0; i < sz-4; i++)
            {
                confid = outputs[ibatch][i+4][ibox].item<float>();
                if (confid > p_max)
                {
                    p_max = confid;
                    iclass = i;
                }
            }
            confid = p_max;
            // defining the Box-elements fields and creating a vector of them
            if (confid >= confid_threshold) {
                unsigned short center_x, center_y, width, height, x1, x2, y1, y2;
                //
                center_x = outputs[ibatch][0][ibox].item<int>();
                center_y = outputs[ibatch][1][ibox].item<int>();
                width = outputs[ibatch][2][ibox].item<int>();
                height = outputs[ibatch][3][ibox].item<int>();
                //
                x1 = center_x - width / 2;
                y1 = center_y - height / 2;
                x2 = center_x + width / 2;
                y2 = center_y + height / 2;
                candidates.push_back(Box(x1, y1, x2, y2, confid, iclass));
            }
        }
    }
    sort(candidates.begin(), candidates.end(), [](Box b1, Box b2) {return b1.confid > b2.confid; });
    vector<Box> boxes = NonMaxSuppression(candidates, iou_threshold);
    return boxes;
}

void DrawBoxes(cv::Mat& img, vector<Box>& boxes, vector<string> names) {
    cv::Scalar rect_color(200, 0, 200);
    unsigned short font_scale = 2;

    for (Box box : boxes) {
        string text = names[box.name_index] + ". With p = " + to_string(box.confid);
        cv::rectangle(img, { box.x1,box.y1 }, { box.x2,box.y2 }, rect_color, 2);
        cv::rectangle(img, { box.x1, box.y1 - font_scale * 12 }, { box.x1 + (unsigned short)text.length() * font_scale * 9, box.y1 }, rect_color, -1); // прямоугольная иконка для текста
        cv::putText(img, text, { box.x1,box.y1 }, cv::FONT_HERSHEY_PLAIN, font_scale, { 255,255,255 }, 2);
    }
}

int main() {
    std::string video_path = "C:\\Users\\my_test_video_cup.MOV";
    std::string video_output_path = "C:\\Users\\Диана\\Desktop\\my_result_video.MOV";
    std::string model_path = "C:\\Users\\bbest.torchscript";

    torch::jit::script::Module model = LoadModel(model_path);

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open video file." << std::endl;
        return -1;
    }

    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    cv::VideoWriter video_writer(video_output_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(frame_width, frame_height));
 
    cv::Mat frame;
    while (cap.read(frame)) {
        vector<string> classes_names = { "Cup" };
        cv::Size original_size = frame.size();
        cv::resize(frame, frame, cv::Size(640, 640));
        torch::Tensor tensor_frame = TransformImage(frame);

        model.eval();
        torch::NoGradGuard no_grad;
        torch::Tensor predictions = model.forward({ tensor_frame }).toTensor();

        vector<Box> boxes = GetBoxes(predictions);
        DrawBoxes(frame, boxes, classes_names);
        cv::resize(frame, frame, original_size);

        video_writer.write(frame);
    }

    cap.release();
    video_writer.release();

    return 0;
}