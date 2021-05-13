#include <stdio.h>
#include <stdlib.h>

#include <CL/cl.h>

#define MAX_SOURCE_SIZE (0x100000)

int main(void) {
	//create 2 input Vectors
	int i;
	const int LIST_SIZE = 1024;
	int *A = (int*)malloc(sizeof(int) * LIST_SIZE);
	int *B = (int*)malloc(sizeof(int) * LIST_SIZE);

	for (i = 0; i < LIST_SIZE; i++) {
		A[i] = i;
		B[i] = LIST_SIZE - i;
	}

	//Load Kernel source code intro the array source_str
	FILE* fp;
	char* src_str;
	size_t src_size;

	fp = fopen("kernel.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load Kernel.\n");
		exit(1);
	}

	src_str = (char*)malloc(MAX_SOURCE_SIZE);
	src_size = fread(src_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	//get Plattform & Device Info
	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);

	//create OpenCL Context
	cl_context ctx = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

	//create Command Queue
	cl_command_queue cmd_queue = clCreateCommandQueueWithProperties(ctx, device_id, 0, &ret);

	//create Memory Buffer on device for each Vector
	cl_mem a_mem_obj = clCreateBuffer(ctx, CL_MEM_READ_ONLY, LIST_SIZE * sizeof(int), NULL, &ret);
	cl_mem b_mem_obj = clCreateBuffer(ctx, CL_MEM_READ_ONLY, LIST_SIZE * sizeof(int), NULL, &ret);
	cl_mem c_mem_obj = clCreateBuffer(ctx, CL_MEM_READ_ONLY, LIST_SIZE * sizeof(int), NULL, &ret);

	//Copy Lists into their buffers
	ret = clEnqueueWriteBuffer(cmd_queue, a_mem_obj, CL_TRUE, 0, LIST_SIZE * sizeof(int), A, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(cmd_queue, b_mem_obj, CL_TRUE, 0, LIST_SIZE * sizeof(int), B, 0, NULL, NULL);

	//create a Program from Kernel Source
	cl_program program = clCreateProgramWithSource(ctx, 1, (const char**)&src_str, (const size_t*)&src_size, &ret);

	//Build Program
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

	//Create OpenCL Kernel
	cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);

	//Set Kernel args
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a_mem_obj);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&b_mem_obj);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&c_mem_obj);

	// Execute the OpenCL kernel on the list
	size_t global_item_size = LIST_SIZE; // Process the entire lists
	size_t local_item_size = 64; // Divide work items into groups of 64
	ret = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);

	// Read the memory buffer C on the device to the local variable C
	int* C = (int*)malloc(sizeof(int) * LIST_SIZE);
	ret = clEnqueueReadBuffer(cmd_queue, c_mem_obj, CL_TRUE, 0,
		LIST_SIZE * sizeof(int), C, 0, NULL, NULL);

	// Display the result
	for (i = 0; i < LIST_SIZE; i++)
		printf("%d + %d = %d\n", A[i], B[i], C[i]);

	

	// Clean up
	ret = clFlush(cmd_queue);
	ret = clFinish(cmd_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(a_mem_obj);
	ret = clReleaseMemObject(b_mem_obj);
	ret = clReleaseMemObject(c_mem_obj);
	ret = clReleaseCommandQueue(cmd_queue);
	ret = clReleaseContext(ctx);
	free(A);
	free(B);
	free(C);
	return 0; 
}




