#include "gaussian.h"
#include "gaussian_kernel.h"
#include <math.h>

#include <iostream>
#include <fstream>

using namespace cv;

using std::cout;
using std::cin;
using std::endl;

ImageFilter::ImageFilter(const string &_filename)
{

    cout <<   "\n/*********************************************/ " << "\n";
    cout <<	    "     Image Filter Base class                  "   << "\n";
    cout <<	    "/*********************************************/ " << "\n\n";
    filename = _filename;
    load_bmp_image();
    setup_filter();
}

void ImageFilter::setup_filter( )
{
    float lFilter[WINDOW_SIZE*WINDOW_SIZE] = {  1.f/16,  2.f/16,  1.f/16,
                                                2.f/16,  4.f/16,  2.f/16,
                                                1.f/16,  2.f/16,  1.f/16  };
    memcpy(filter, lFilter, WINDOW_SIZE*WINDOW_SIZE*sizeof(float));
}

void ImageFilter::load_bmp_image()
{
	img = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	if (!img.data)                              // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
	}
	pixelColor = (float*)malloc(img.cols * img.rows * sizeof(float));
	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			pixelColor[y * img.cols + x] = (float)img.at<uchar>(y, x);
		}
	}
}

void ImageFilter::write_bmp_image( )
{
	unsigned char *pixelColor;
	pixelColor = (unsigned char*)malloc(img.cols* img.rows* sizeof(unsigned char));

	for (int i = 0; i < (img.rows * img.cols); i++)
	{
		pixelColor[i] = (unsigned char)GPU_output[i];
	}
	out = Mat(img.rows, img.cols, CV_8U, pixelColor);
	imwrite("out.bmp", out);
	fprintf(stdout, "Saved BMP file.\n");
}

void ImageFilter::read_GPU_filtered_image()
{
    size_t origin[3];
    size_t region[3];
    cl_int status = 0;
    origin[0] = origin[1] = origin[2] = 0;
    region[0] = img.cols; region[1] = img.rows; region[2] = 1;
    status = clEnqueueReadImage(commandQueue, ocl_filtered_image, CL_TRUE, origin, region, 0, 0, GPU_output, 0, NULL, NULL);
    LOG_OCL_ERROR(status, "clEnqueueReadImage failed" );
}

void ImageFilter::load_GPU_raw_image()
{
    size_t origin[3];
    size_t region[3];
    cl_int status = 0;
    origin[0] = origin[1] = origin[2] = 0;
    region[0] = img.cols; region[1] = img.rows; region[2] = 1;
	pixels = (void *)pixelColor;
    status = clEnqueueWriteImage(commandQueue, ocl_raw, CL_TRUE, origin, region, 0, 0, pixels, 0, NULL, NULL);
    LOG_OCL_ERROR(status, "clEnqueueWriteImage failed" );
}

void ImageFilter::print_GPU_Timer()
{
    printf("GPU execution time is.......... %lf (ms)\n", 1000*timer_GPU.GetElapsedTime());
}

void ImageFilter::init_GPU_OpenCL( )
{
    //Allocate GPU output image memory
    GPU_output = NULL;
    GPU_output = (float*) calloc(1, img.rows*img.cols*sizeof(float) );
    deviceType = CL_DEVICE_TYPE_GPU;
    setupOCLPlatform();
    setupOCLProgram();
    setupOCLkernels();
    setupOCLbuffers();

    gwsize[0] = img.cols;
    gwsize[1] = img.rows;
    lwsize[0] = lwsize[1] = 16;
}

void ImageFilter::setupOCLPlatform()
{
    cl_int status;
    //Setup the OpenCL Platform,
    //Get the first available platform. Use it as the default platform
    status = clGetPlatformIDs(1, &platform, NULL);
    LOG_OCL_ERROR(status, "Error # clGetPlatformIDs" );

    //Get the first available device
    status = clGetDeviceIDs (platform, deviceType, 1, &device, NULL);
    LOG_OCL_ERROR(status, "Error # clGetDeviceIDs" );

    //Create an execution context for the selected platform and device.
    cl_context_properties cps[3] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };
    context = clCreateContextFromType(
        cps,
        deviceType,
        NULL,
        NULL,
        &status);
    LOG_OCL_ERROR(status, "Error # clCreateContextFromType" );

    // Create command queue
    commandQueue = clCreateCommandQueue(context,
                                        device,
                                        0,
                                        &status);
    LOG_OCL_ERROR(status, "Error # clCreateCommandQueue" );
}

cl_int ImageFilter::setupOCLProgram()
{
    cl_int status;
    program = clCreateProgramWithSource(context, 1,
                (const char **)&gaussian_kernel, NULL, &status);
    LOG_OCL_ERROR(status, "clCreateProgramWithSource Failed" );

    // Build the program
    status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if(status != CL_SUCCESS)
    {
        if(status == CL_BUILD_PROGRAM_FAILURE)
            LOG_OCL_COMPILER_ERROR(program, device);
        LOG_OCL_ERROR(status, "clBuildProgram Failed" );
    }
    return status;

}

cl_int ImageFilter::setupOCLkernels()
{
    cl_int status;
    // Create the OpenCL kernel
    gd_kernel = clCreateKernel(program, "gaussian_filter_kernel", &status);
    LOG_OCL_ERROR(status, "clCreateKernel Failed" );

    return status;
}

cl_int ImageFilter::setupOCLbuffers()
{
    cl_int status;
    //Intermediate reusable cl buffers
    cl_image_format image_format;
    image_format.image_channel_data_type = CL_FLOAT;
    image_format.image_channel_order = CL_R;
    ocl_raw = clCreateImage2D(
        context,
        CL_MEM_READ_ONLY,
        &image_format,
        img.rows, img.cols, 0, 	
        NULL,
        &status);
    LOG_OCL_ERROR(status, "clCreateImage Failed" );

    ocl_filtered_image = clCreateImage2D(
        context,
        CL_MEM_WRITE_ONLY,
        &image_format,
        img.rows, img.cols, 0, 	
        NULL,
        &status);
    LOG_OCL_ERROR(status, "clCreateImage Failed" );

    ocl_filter = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,
        WINDOW_SIZE*WINDOW_SIZE*sizeof(float),
        filter,
        &status);
    LOG_OCL_ERROR(status, "clCreateBuffer Failed" );

    //Create OpenCL device output buffer
    return status;
}

void ImageFilter::run_GPU()
{
    load_GPU_raw_image();
    run_gaussian_filter_kernel();
    read_GPU_filtered_image();
}

void ImageFilter::run_gaussian_filter_kernel()
{
    cl_event	wlist[2];
    cl_int status;

    int windowSize = WINDOW_SIZE;
    status = clSetKernelArg(gd_kernel, 0, sizeof(cl_mem), (void*)&ocl_raw);
    status = clSetKernelArg(gd_kernel, 1, sizeof(cl_mem), (void*)&ocl_filtered_image);
    status = clSetKernelArg(gd_kernel, 2, sizeof(cl_mem), (void*)&ocl_filter);
    status = clSetKernelArg(gd_kernel, 3, sizeof(int), (void*)&windowSize);
    status = clEnqueueNDRangeKernel(
                        commandQueue,
                        gd_kernel,
                        2,
                        NULL,
                        gwsize,
                        lwsize,
                        0,
                        NULL,
                        &wlist[0]);
    LOG_OCL_ERROR(status, "clEnqueueNDRangeKernel Failed" );
    clWaitForEvents(1, &wlist[0]);
}
int main(int argc, char* argv[])
{
    //if(argc < 2)
    //{
    //    std::cout << "Usage: chapter9.median.exe sample.bmp\n";
    //    std::cout << "The file sample.bmp is available in the input_images directory. This \n";
    //    std::cout << "This should be a grayscale image. and the height and width should be amultiple of 16 pixels\n";
    //    return 0;
    //}
    //ImageFilter*	img_filter = new ImageFilter(std::string(argv[1]));
	ImageFilter*	img_filter = new ImageFilter("sample.bmp");
    unsigned int num_of_frames = 0;
    try
    {
        img_filter->init_GPU_OpenCL();
        img_filter->start_GPU_Timer();
        img_filter->run_GPU();
        img_filter->stop_GPU_Timer();
        img_filter->print_GPU_Timer();
        img_filter-> write_bmp_image( );
        delete(img_filter);
    }
#ifdef __CL_ENABLE_EXCEPTIONS
    catch(cl::Error err)
    {
        std::cout << "Error: " << err.what() << "(" << err.err() << ")" << std::endl;
        cout << "Please check CL/cl.h for error code" << std::endl;
        delete(img_filter);
    }
#endif
    catch(string msg)
    {
        std::cout << "Exception caught: " << msg << std::endl;
        delete(img_filter);
    }

    return 0;
}



