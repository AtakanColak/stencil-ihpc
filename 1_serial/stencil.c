
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Define output file name
#define OUTPUT_FILE "stencil.pgm"

void stencil(const int nx, const int ny, float *restrict  image, float *restrict  tmp_image);
void init_image(const int nx, const int ny, float *restrict  image, float *restrict  tmp_image);
void output_image(const char * file_name, const int nx, const int ny, float *restrict image);
double wtime(void);

int main(int argc, char *argv[]) {

  // Check usage
  if (argc != 4) {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // Initiliase problem dimensions from command line arguments
  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int niters = atoi(argv[3]);

  // Allocate the image
  float *restrict image = malloc(sizeof(float)*nx*ny);
  float *restrict tmp_image = malloc(sizeof(float)*nx*ny);

  // Set the input image
  init_image(nx, ny, image, tmp_image);

  // Call the stencil kernel
  double tic = wtime();
  for (int t = 0; t < niters; ++t) {
    stencil(nx, ny, image, tmp_image);
    stencil(nx, ny, tmp_image, image);
  }
  double toc = wtime();


  // Output
  //FILE *f = fopen("stencil.out","w");
  printf("------------------------------------\n");
  printf(" runtime: %lf s\n", toc-tic);
  printf("------------------------------------\n");
  //fclose(f);

  output_image(OUTPUT_FILE, nx, ny, image);
  free(image);
}

void stencil(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {
//case j = 0
//case i = 0
    //tmp_image[0]  = image[0] * 6;
    //tmp_image[0] += image[1];
    //tmp_image[0] += image[nx];

    //for (int i = 1; i < nx - 1; ++i) {
     // tmp_image[i] = image[i] * 6;
      //tmp_image[i] += image[i-1];
      //tmp_image[i] += image[i+1];
      //tmp_image[i] += image[nx + i];
    //}
    // case i = nx -1
    //tmp_image[ nx - 1] = image[ nx -1] * 6;
    //tmp_image[ nx - 1] += image[ nx- 2];
    //tmp_image[ nx -1] += image[ (2*nx) - 1];
//case j [1,ny]
    for (int j = 0; j < ny; ++j) {
      //case i = 0
      //tmp_image[j*nx]  = image[j*nx] * 6;
      //tmp_image[j*nx] += image[j*nx + 1];
      //tmp_image[ (j*nx)] += image[ ((j-1)*nx)];
      //tmp_image[ (j*nx)] += image[ ((j+1)*nx)];

      for (int i = 0; i < nx; ++i) {
	
        tmp_image[ (j*nx) + i] = image[ (j*nx) + i] * 0.6f;
        if (i > 0)     tmp_image[ (j*nx) + i] += image[ (j*nx) + i-1] * 0.1f;
        if (i < nx -1) tmp_image[ (j*nx) + i] += image[ (j*nx) + i+1] * 0.1f;
        if (j > 0)      tmp_image[ (j*nx) + i] += image[ ((j-1)*nx) + i] * 0.1f;
        if (j < ny -1) tmp_image[ (j*nx) + i] += image[ ((j+1)*nx) + i] *0.1f;
    }
      // case i = nx -1
      //tmp_image[ ((j+1)*nx) - 1] = image[ ((j+1)*nx) -1] * 6;
      //tmp_image[ ((j+1)*nx) - 1] += image[ ((j+1)*nx)- 2];
      //tmp_image[ ((j+1)*nx) -1] += image[ ((j)*nx) - 1];
      //tmp_image[ ((j+1)*nx) -1] += image[ ((j+2)*nx) - 1];
  }
    //for j = (ny-1)

    //tmp_image[ (ny-1)*nx]  = image[ (ny-1)*nx] * 6;
    //tmp_image[ (ny-1)*nx] += image[ (ny-1)*nx + 1];
    //tmp_image[ ( (ny-1)*nx)] += image[ ((ny-2)*nx)];

    //for (int i = 1; i < nx - 1; ++i) {
      //tmp_image[ ( (ny-1)*nx) + i] = image[ ( (ny-1)*nx) + i] * 6;
      //tmp_image[ ( (ny-1)*nx) + i] += image[ ( (ny-1)*nx) + i-1];
      //tmp_image[ ( (ny-1)*nx) + i] += image[ ( (ny-1)*nx) + i+1];
    //  tmp_image[ ( (ny-1)*nx) + i] += image[ ((ny-2)*nx) + i];
    //}
    // case i = nx -1
    //tmp_image[ (ny*nx) - 1] = image[ (ny*nx) -1] * 6;
    //tmp_image[ (ny*nx) - 1] += image[ (ny*nx)- 2];
    //tmp_image[ (ny*nx) -1] += image[ (( (ny-1))*nx) - 1];

}

// Create the input image
void init_image(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {
  // Zero everything
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      image[j+i*ny] = 0.0f;
      tmp_image[j+i*ny] = 0.0f;
    }
  }

  // Checkerboard
  for (int j = 0; j < 8; ++j) {
    for (int i = 0; i < 8; ++i) {
      for (int jj = j*ny/8; jj < (j+1)*ny/8; ++jj) {
        for (int ii = i*nx/8; ii < (i+1)*nx/8; ++ii) {
          if ((i+j)%2)
          image[jj+ii*ny] = 100.0f;
        }
      }
    }
  }
}

// Routine to output the image in Netpbm grayscale binary image format
void output_image(const char * file_name, const int nx, const int ny, float * restrict image) {

  // Open output file
  FILE *fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }

  // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", nx, ny);

  // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for output
  float maximum = 0.0f;
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      if (image[j+i*ny] > maximum)
        maximum = image[j+i*ny];
    }
  }

  // Output image, converting to numbers 0-255
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      fputc((char)(255.0f*image[j+i*ny]/maximum), fp);
    }
  }

  // Close the file
  fclose(fp);

}

// Get the current time in seconds since the Epoch
double wtime(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec*1e-6;
}
