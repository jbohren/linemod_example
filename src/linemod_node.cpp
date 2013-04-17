
#include <iostream>
#include <cassert>

#include <ros/ros.h>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <sensor_msgs/Image.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

//! Line-2D with iterative re-wieghted least squares post-alignment
class Line2DI {
public:
  Line2DI();
  ~Line2DI();

  void add_template(std::string object_name, cv::Mat image, cv::Mat mask, cv::Matx33d R, cv::Vec3d T);
  void image_cb(const sensor_msgs::ImageConstPtr& msg);

  void draw_response(
      const std::vector<cv::linemod::Template>& templates,
      int num_modalities, 
      cv::Mat& dst,
      cv::Point offset, 
      int T);
private:

  double matching_threshold_;

  std::vector<cv::Mat> Ts_, Rs_;
  cv::Ptr<cv::linemod::Detector> linemod_detector_;
};

Line2DI::Line2DI() :
  matching_threshold_(80.0),
  linemod_detector_(cv::linemod::getDefaultLINE())
{

}

Line2DI::~Line2DI() { }

void Line2DI::add_template(std::string object_name, cv::Mat image, cv::Mat mask, cv::Matx33d R, cv::Vec3d T)
{
    // Construct the multi-modal sources vector
    std::vector<cv::Mat> sources(1);
    sources[0] = image;
    //sources[1] = depth;

    // Store template sources and generate templates
    linemod_detector_->addTemplate(sources, object_name, mask);

    Rs_.push_back(cv::Mat(R));
    Ts_.push_back(cv::Mat(T));
}


void Line2DI::draw_response(
    const std::vector<cv::linemod::Template>& templates,
    int num_modalities, 
    cv::Mat& dst,
    cv::Point offset, 
    int T)
{

  static const cv::Scalar COLORS[5] = { 
    CV_RGB(0, 0, 255),
    CV_RGB(0, 255, 0),
    CV_RGB(255, 255, 0),
    CV_RGB(255, 140, 0),
    CV_RGB(255, 0, 0)
  };

  for (int m = 0; m < num_modalities; ++m)
  {
    // NOTE: Original demo recalculated max response for each feature in the TxT
    // box around it and chose the display color based on that response. Here
    // the display color just depends on the modality.
    cv::Scalar color = COLORS[m];

    for (int i = 0; i < (int)templates[m].features.size(); ++i)
    {
      cv::linemod::Feature f = templates[m].features[i];
      cv::Point pt(f.x + offset.x, f.y + offset.y);
      cv::circle(dst, pt, T / 2, color);
    }
  }
}

void Line2DI::image_cb(const sensor_msgs::ImageConstPtr& msg) 
{
  // Get the opencv image from the ros image message
  cv_bridge::CvImageConstPtr msg_image = cv_bridge::toCvShare(msg, "bgr8");
  const cv::Mat &color = msg_image->image;

  // Initialize display image
  cv::Mat display = color;

  // TODO: Resize the image if necessary?
  std::vector<cv::Mat> sources;
  sources.push_back(color);

  std::vector<cv::linemod::Match> matches;
  std::vector<std::string> class_ids;
  std::vector<cv::Mat> quantized_images;

  // Perform matching
  linemod_detector_->match(
      sources, 
      matching_threshold_, 
      matches, 
      class_ids, 
      quantized_images);


  int num_modalities = (int)linemod_detector_->getModalities().size();
  std::set<std::string> visited;

  for (int i = 0; 
       (i < (int)matches.size()) && (visited.size() < (size_t)linemod_detector_->numClasses()); 
       ++i)
  {
    cv::linemod::Match m = matches[i];

    if (visited.insert(m.class_id).second) {
      ROS_DEBUG("Similarity: %5.1f%%; x: %3d; y: %3d; class: %s; template: %3d\n",
                m.similarity, m.x, m.y, m.class_id.c_str(), m.template_id);

      // Draw matching template
      const std::vector<cv::linemod::Template>& templates = 
        linemod_detector_->getTemplates(m.class_id, m.template_id);

      draw_response(templates,
                    num_modalities,
                    display, 
                    cv::Point(m.x, m.y),
                    linemod_detector_->getT(0));
      
      cv::namedWindow("LINEMOD");
      cv::imshow("LINEMOD", display);
      cv::waitKey(1);
    }
  }
}

int main(int argc, char** argv) 
{

  // ROS initialization and argument sanitization
  ros::init(argc, argv, "linemod");
  ros::NodeHandle nh;

  // Parse arguments
  namespace po = boost::program_options;
  namespace fs = boost::filesystem;

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("data-path", po::value<std::string>()->required(), "path to training data")
    ;

  po::positional_options_description pdesc;
  pdesc.add("data-path", 1);

  po::variables_map vm;
  //po::store(po::parse_command_line(argc, argv, desc), vm);
  po::store(po::command_line_parser(argc, argv).options(desc).positional(pdesc).run(), vm);
  try {

    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 1;
    }

    po::notify(vm);    
  } catch(boost::program_options::error &ex) {
    std::cerr << "Invalid arguments: " << ex.what() << std::endl;
    return -1;
  }

  // Set up the linemod detector
  Line2DI linei;
  
  // Load yaml file with metadata and poses
  fs::path data_path = fs::path(vm["data-path"].as<std::string>().c_str());
  fs::path render_info_yaml_path = data_path / fs::path("render_info.yml");
  cv::FileStorage render_info(render_info_yaml_path.string(), cv::FileStorage::READ);

  int width = render_info["width"],
      height = render_info["height"];
  double near = render_info["near"],
         far = render_info["far"],
         fx = render_info["fx"],
         fy = render_info["fy"];

  int n_poses = 0;
  render_info["n_poses"] >> n_poses;

  if(unsigned(n_poses) != render_info["poses"].size()) {
    std::cerr<<"ERROR: Metadata file is missing pose entries. The file only contains "<<render_info["poses"].size()<<" poses, but it should contain "<<n_poses<<std::endl;
    return -1;
  }

  // Load the poses from file
  std::cerr<<"Loading "<<n_poses<<" poses..."<<std::endl;

  std::string file_index_format = render_info["filename_format"];

  cv::FileNode poses = render_info["poses"];
  int pose_index = 0;


  // Extract the poses and load the images for each pose
  for(cv::FileNodeIterator pose_it = poses.begin();
      pose_it != poses.end();
      ++pose_it, pose_index++ )
  {
    std::cout<<"\x1B[2K"<<"\x1B[0E";
    std::cout<<(pose_index+1)<<" / "<<n_poses;
    std::flush(std::cout);
    
    cv::Matx33d R;
    for(int i=0; i<9; i++) {
      R << (double)(*pose_it)["R"][i];
    }

    cv::Vec3d T;
    for(int i=0; i<3; i++) {
      T << (double)(*pose_it)["T"][i];
    }

    // Load template images from disk
    cv::Mat depth, image, mask;
    {
      using namespace boost;

      std::string file_index_str = str(format(file_index_format) % pose_index);

      std::string image_filename = str(format("image_%s.png") % file_index_str);
      std::string depth_filename = str(format("depth_%s.png") % file_index_format);
      std::string mask_filename = str(format("mask_%s.png") % file_index_format);

      image = cv::imread((data_path / fs::path(image_filename)).string());
      depth = cv::imread((data_path / fs::path(depth_filename)).string());
      mask = cv::imread((data_path / fs::path(mask_filename)).string());
    }

    // Store the template
    linei.add_template("object", image, mask, R, T);
  
  }
  std::cout<<std::endl;
  
  image_transport::ImageTransport im_transport(nh);
  image_transport::Subscriber sub = im_transport.subscribe("image", 1, boost::bind(&Line2DI::image_cb, &linei, _1));

  ros::spin();
  
  return 0;
}
