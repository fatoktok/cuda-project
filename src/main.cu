#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void invertColors(unsigned char *image, int width, int height, int channels)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Make sure we don't access memory outside the image.
    if (x < width && y < height)
    {
        int idx = (y * width + x) * channels;  // Compute the index for the pixel.


        for (int c = 0; c < channels; ++c)
        {
            image[idx + c] = 255 - image[idx + c];
        }
    }
}

int main(int argc, char *argv[])
{

    int width = 512;
    int height = 512;
    int channels = 3; // RGB image.
    size_t imageSize = width * height * channels * sizeof(unsigned char);
    
    // Allocate host memory for the image.
    unsigned char *h_image = (unsigned char*)malloc(imageSize);
    if (h_image == NULL) {
        fprintf(stderr, "Error: Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize the dummy image with a constant value (e.g., 100 for each channel).
    for (int i = 0; i < width * height * channels; i++) {
        h_image[i] = 100;
    }
    
    // Allocate device memory.
    unsigned char *d_image;
    cudaError_t err = cudaMalloc((void**)&d_image, imageSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Failed to allocate device memory: %s\n", cudaGetErrorString(err));
        free(h_image);
        return EXIT_FAILURE;
    }
    
    // Copy the image data from the host to the device.
    err = cudaMemcpy(d_image, h_image, imageSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Failed to copy image from host to device: %s\n", cudaGetErrorString(err));
        cudaFree(d_image);
        free(h_image);
        return EXIT_FAILURE;
    }
    
    // Define the block and grid dimensions.
    // Using 16x16 threads per block is a common choice.
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    // Launch the kernel.
    invertColors<<<gridSize, blockSize>>>(d_image, width, height, channels);
    
    // Check if the kernel launch resulted in an error.
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Failed to launch kernel: %s\n", cudaGetErrorString(err));
        cudaFree(d_image);
        free(h_image);
        return EXIT_FAILURE;
    }
    
    // Copy the processed image data back from the device to the host.
    err = cudaMemcpy(h_image, d_image, imageSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Failed to copy result from device to host: %s\n", cudaGetErrorString(err));
        cudaFree(d_image);
        free(h_image);
        return EXIT_FAILURE;
    }
    

    printf("First 10 pixels (R, G, B):\n");
    for (int i = 0; i < 10; i++) {
        int index = i * channels;
        printf("Pixel %d: %3u, %3u, %3u\n", i, h_image[index], h_image[index+1], h_image[index+2]);
    }
    
    // Clean up device and host memory.
    cudaFree(d_image);
    free(h_image);
    
    printf("Processing complete.\n");
    return EXIT_SUCCESS;
}
