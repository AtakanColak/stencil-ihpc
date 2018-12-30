#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"

// Define output file name
#define OUTPUT_FILE "stencil.pgm"
#define TRUE 1

int core_rank, core_count, message_flag, nx, ny, niters;

double tic, toc;

#define MASTER 0
#define LAST (core_count - 1)

#define IS_MASTER (core_rank == MASTER)
#define IS_LAST (core_rank == LAST)

#define ROW_PER_CORE (ny / core_count)
#define ROW_REM_LAST (ny % core_count)

#define ROW_OF_CORE(X) ((X == LAST) ? (ROW_PER_CORE + ROW_REM_LAST) : (ROW_PER_CORE))
#define ROW_CUR_CORE (ROW_OF_CORE(core_rank) + 2)

#define ABOVE (core_rank - 1)
#define BELOW (core_rank + 1)

#define NOT_TOP (core_rank != MASTER)
#define NOT_BOTTOM (core_rank != LAST)

#define ROW_RECV_ABOVE 0
#define ROW_SEND_ABOVE 1

#define ROW_RECV_BELOW (ROW_CUR_CORE - 1)
#define ROW_SEND_BELOW (ROW_CUR_CORE - 2)

#define TAG_DIST 0

#define FLOAT_PER_BUF (BUFSIZ / 4)

int col_per_row, col_per_last, col_num;

#define COL_LENGTH(X) (((X == col_num - 1) && (col_per_last != 0)) ? col_per_last : FLOAT_PER_BUF)

void send_tic(double *tic);
void recv_tic(double *tic, int rank, MPI_Status status);

void send_row(int dest_core_index, int row_index, float *restrict source, float *restrict message);
void recv_row(int src_core_index, int row_index, float *restrict dest, float *restrict message, MPI_Status status);

void send_recv_row(int core_index, int recv_row_index, int send_row_index, float *restrict sender, float *restrict receiver, float *restrict img_frag, MPI_Status status);
void send_recv_overlaps(float *restrict sender, float *restrict receiver, float *restrict img_frag, MPI_Status status);

void stencil(float *restrict dest, float *restrict source);
void stencil_mpi(float *restrict img_frag, float *restrict tmp_img_frag, float *restrict sender, float *restrict receiver, MPI_Status status);

void init_image(float *restrict image);
void output_image(const char *file_name, float *restrict image);

double wtime(void);

int main(int argc, char *argv[])
{
  MPI_Status status;

  if (argc != 4)
  {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  MPI_Init(&argc, &argv);
  MPI_Initialized(&message_flag);
  if (message_flag != TRUE)
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  MPI_Comm_size(MPI_COMM_WORLD, &core_count);
  MPI_Comm_rank(MPI_COMM_WORLD, &core_rank);

  nx = atoi(argv[1]);
  ny = atoi(argv[2]);
  niters = atoi(argv[3]);

  col_per_row = (nx / FLOAT_PER_BUF);
  col_per_last = (nx % FLOAT_PER_BUF);
  col_num = ((col_per_last == 0) ? col_per_row : col_per_row + 1);

  // printf("Core #%d active...\n", core_rank);
  // printf("Core #%d ROW_CUR_CORE:%d \n", core_rank, ROW_CUR_CORE);
  float *restrict img_frag = malloc(sizeof(float) * nx * ROW_CUR_CORE);
  float *restrict tmp_img_frag = malloc(sizeof(float) * nx * ROW_CUR_CORE);
  float *restrict message = malloc(sizeof(float) * nx);
  float *restrict receiver = malloc(sizeof(float) * nx);
  float *restrict sender = malloc(sizeof(float) * nx);

  if (IS_MASTER)
  {
    // printf("Buffer float size is %d \n", FLOAT_PER_BUF);
    // printf("Column count is %d \n", col_num);
    // printf("Image size is %d x %d\n", nx, ny);
    float *restrict image = malloc(sizeof(float) * nx * ny);
    float *restrict tmp_image = malloc(sizeof(float) * nx * ny);
    double *restrict tics = malloc(sizeof(double) * core_count);
    init_image(image);

    //printf("Core #%d distributing...\n", core_rank);
    for (int j = 1; j < core_count; ++j)
      for (int i = 0; i < ROW_OF_CORE(j); ++i)
        send_row(j, j * ROW_PER_CORE + i, image, message);

    for (int j = 0; j < ROW_PER_CORE; ++j)
      for (int i = 0; i < nx; ++i)
        img_frag[((j + 1) * nx) + i] = image[(j * nx) + i];

    tic = wtime();
    stencil_mpi(img_frag, tmp_img_frag, sender, receiver, status);
    toc = wtime();

    //printf("Core #%d receiving...\n", core_rank);
    // Gather sections
    for (int j = 1; j < core_count; ++j)
    {
      for (int i = 0; i < ROW_OF_CORE(j); ++i)
        recv_row(j, j * ROW_PER_CORE + i, image, message, status);
      double elapsed;
      recv_tic(&elapsed, j, status);
      tics[j] = elapsed;
    }
    tics[0] = toc - tic;
    double max_elapse = tics[0];
    //printf(" runtime of core %d: %lf s\n", core_rank, max_elapse);
    for (int j = 1; j < core_count; ++j)
      if (tics[j] > max_elapse)
        max_elapse = tics[j];

    //Push own section
    for (int j = 0; j < ROW_PER_CORE; ++j)
      for (int i = 0; i < nx; ++i)
        image[(j * nx) + i] = img_frag[((j + 1) * nx) + i];
    //printf(" Number of cores used : %d\n", core_count);
    // Output
    printf("------------------------------------\n");
    printf(" runtime: %lf s\n", max_elapse);
    printf("------------------------------------\n");

    output_image(OUTPUT_FILE, image);
    free(image);
    free(tmp_image);
    free(tics);
  }
  else
  {
    //printf("Core #%d receiving...\n", core_rank);
    for (int i = 0; i < ROW_OF_CORE(core_rank); ++i)
      recv_row(MASTER, i + 1, img_frag, message, status);

    tic = wtime();
    stencil_mpi(img_frag, tmp_img_frag, sender, receiver, status);
    toc = wtime();

    //printf("Core #%d sending...\n", core_rank);
    for (int i = 0; i < ROW_OF_CORE(core_rank); ++i)
      send_row(MASTER, i + 1, img_frag, message);

    double elapsed_time = toc - tic;
    send_tic(&elapsed_time);
    //printf(" runtime of core %d: %lf s\n", core_rank, elapsed_time);
    //printf("Core #%d terminating...\n", core_rank);
  }
  free(message);
  free(img_frag);
  free(tmp_img_frag);
  free(receiver);
  free(sender);
  MPI_Finalize();
  return EXIT_SUCCESS;
}

void edge_case(float *restrict tmp_image, float *restrict image, int c, int v1, int v2, int v3)
{
  tmp_image[c] = image[c] * 0.6f;
  tmp_image[c] += image[c + v1] * 0.1f;
  tmp_image[c] += image[c + v2] * 0.1f;
  tmp_image[c] += image[c + v3] * 0.1f;
}

void send_tic(double *tic)
{
  MPI_Send(tic, 1, MPI_DOUBLE, MASTER, TAG_DIST, MPI_COMM_WORLD);
}

void recv_tic(double *tic, int rank, MPI_Status status)
{
  MPI_Recv(tic, 1, MPI_DOUBLE, rank, TAG_DIST, MPI_COMM_WORLD, &status);
}

void send_row(int dest_core_index, int row_index, float *restrict source, float *restrict message)
{
  for (int i = 0; i < nx; ++i)
    message[i] = source[(row_index * nx) + i];
  MPI_Send(message, nx, MPI_FLOAT, dest_core_index, TAG_DIST, MPI_COMM_WORLD);
}

void recv_row(int src_core_index, int row_index, float *restrict dest, float *restrict message, MPI_Status status)
{
  MPI_Recv(message, nx, MPI_FLOAT, src_core_index, TAG_DIST, MPI_COMM_WORLD, &status);
  for (int i = 0; i < nx; ++i)
    dest[(row_index * nx) + i] = message[i];
}

void send_recv_row(int core_index, int recv_row_index, int send_row_index, float *restrict sender, float *restrict receiver, float *restrict img_frag, MPI_Status status)
{
  for (int i = 0; i < nx; ++i)
    sender[i] = img_frag[(send_row_index * nx) + i];
  MPI_Sendrecv(sender, nx, MPI_FLOAT, core_index, TAG_DIST, receiver, nx, MPI_FLOAT, core_index, TAG_DIST, MPI_COMM_WORLD, &status);
  for (int i = 0; i < nx; ++i)
    img_frag[(recv_row_index * nx) + i] = receiver[i];
}

void send_recv_overlaps(float *restrict sender, float *restrict receiver, float *restrict img_frag, MPI_Status status)
{
  if (core_rank % 2 == 0)
  {
    if (NOT_BOTTOM)
      send_recv_row(BELOW, ROW_RECV_BELOW, ROW_SEND_BELOW, sender, receiver, img_frag, status);
    if (NOT_TOP)
      send_recv_row(ABOVE, ROW_RECV_ABOVE, ROW_SEND_ABOVE, sender, receiver, img_frag, status);
  }
  else
  {
    if (NOT_TOP)
      send_recv_row(ABOVE, ROW_RECV_ABOVE, ROW_SEND_ABOVE, sender, receiver, img_frag, status);
    if (NOT_BOTTOM)
      send_recv_row(BELOW, ROW_RECV_BELOW, ROW_SEND_BELOW, sender, receiver, img_frag, status);
  }
}


void stencil(float *restrict dest, float *restrict source)
{
  //For Corners and horizontal edges just have them as 0

  for (int j = ROW_SEND_ABOVE; j < ROW_RECV_BELOW; ++j)
  {
    edge_case(dest, source, j * nx, -nx, +1, +nx);
    edge_case(dest, source, (j + 1) * nx - 1, -nx, -1, +nx);
    for (int i = 1; i < nx - 1; ++i)
    {
      dest[(j * nx) + i] = source[(j * nx) + i] * 0.6f;
      dest[(j * nx) + i] += source[(j * nx) + i - 1] * 0.1f;
      dest[(j * nx) + i] += source[(j * nx) + i + 1] * 0.1f;
      dest[(j * nx) + i] += source[((j - 1) * nx) + i] * 0.1f;
      dest[(j * nx) + i] += source[((j + 1) * nx) + i] * 0.1f;
    }
  }
}

void stencil_mpi(float *restrict img_frag, float *restrict tmp_img_frag, float *restrict sender, float *restrict receiver, MPI_Status status)
{
  //printf("Core #%d stencil start...\n", core_rank);
  // Set top and bottom to zeroes
  if (IS_MASTER)
    for (int i = nx; i < nx; ++i)
    {
      img_frag[i] = 0.0f;
      tmp_img_frag[i] = 0.0f;
    }
  if (IS_LAST)
    for (int i = nx; i < nx; ++i)
    {
      img_frag[ROW_RECV_BELOW * nx + i] = 0.0f;
      tmp_img_frag[ROW_RECV_BELOW * nx + i] = 0.0f;
    }

  //printf("Core %d\nROW_PER_CORE %d\nROW_CUR_CORE %d\n",core_rank, ROW_PER_CORE, ROW_CUR_CORE);
  for (int t = 0; t < niters; ++t)
  {
    // printf("Core #%d send_rec %d\n", core_rank, 2*t);
    send_recv_overlaps(sender, receiver, img_frag, status);
    //printf("Core #%d stencil %d\n", core_rank, 2*t);
    stencil(tmp_img_frag, img_frag);
    //printf("Core #%d send_rec %d\n", core_rank, 2*t + 1);
    send_recv_overlaps(sender, receiver, tmp_img_frag, status);
    //printf("Core #%d stencil %d\n", core_rank, 2*t + 1);
    stencil(img_frag, tmp_img_frag);
    //test whether send or stencil is faulty
  }
  //printf("Core #%d stencil end...\n", core_rank);
}

// Create the input image
void init_image(float *restrict image)
{
  // Checkerboard
  for (int j = 0; j < 8; ++j)
  {
    for (int i = 0; i < 8; ++i)
    {
      for (int jj = j * ny / 8; jj < (j + 1) * ny / 8; ++jj)
      {
        for (int ii = i * nx / 8; ii < (i + 1) * nx / 8; ++ii)
        {
          if ((i + j) % 2)
            image[jj + ii * ny] = 100.0f;
          else
            image[jj + ii * ny] = 0.0f;
        }
      }
    }
  }
}

// Routine to output the image in Netpbm grayscale binary image format
void output_image(const char *file_name, float *restrict image)
{

  // Open output file
  FILE *fp = fopen(file_name, "w");
  if (!fp)
  {
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }

  // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", nx, ny);

  // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for output
  float maximum = 0.0f;
  for (int j = 0; j < ny; ++j)
  {
    for (int i = 0; i < nx; ++i)
    {
      if (image[j + i * ny] > maximum)
        maximum = image[j + i * ny];
    }
  }

  // Output image, converting to numbers 0-255
  for (int j = 0; j < ny; ++j)
  {
    for (int i = 0; i < nx; ++i)
    {
      fputc((char)(255.0f * image[j + i * ny] / maximum), fp);
    }
  }

  // Close the file
  fclose(fp);
}

// Get the current time in seconds since the Epoch
double wtime(void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

// void corner_case(float *restrict tmp_image, float *restrict image, int c, int v1, int v2)
// {
//   tmp_image[c] = image[c] * 0.6f;
//   tmp_image[c] += image[c + v1] * 0.1f;
//   tmp_image[c] += image[c + v2] * 0.1f;
// }

// void stencil(const int nx, const int ny, float *restrict image, float *restrict tmp_image)
// {
//   //Corner cases
//   corner_case(tmp_image, image, 0, 1, nx);
//   corner_case(tmp_image, image, (nx - 1), -1, nx);
//   corner_case(tmp_image, image, ((ny - 1) * nx), 1, -nx);
//   corner_case(tmp_image, image, (ny * nx - 1), -1, -nx);

//   //Horizontal Edges
//   for (int i = 1; i < (nx - 1); ++i)
//   {
//     edge_case(tmp_image, image, i, -1, +1, nx);
//     edge_case(tmp_image, image, (ny - 1) * nx + i, -1, +1, -nx);
//   }

//   //Vertical Edges
//   for (int i = 1; i < (ny - 1); ++i)
//   {
//     edge_case(tmp_image, image, i * nx, -nx, +1, nx);
//     edge_case(tmp_image, image, (i + 1) * nx - 1, -nx, -1, nx);
//   }

//   //Inner pixels

//   for (int j = 1; j < (ny - 1); ++j)
//   {
//     for (int i = 1; i < (nx - 1); ++i)
//     {
//       tmp_image[(j * nx) + i] = image[(j * nx) + i] * 0.6f;
//       tmp_image[(j * nx) + i] += image[(j * nx) + i - 1] * 0.1f;
//       tmp_image[(j * nx) + i] += image[(j * nx) + i + 1] * 0.1f;
//       tmp_image[(j * nx) + i] += image[((j - 1) * nx) + i] * 0.1f;
//       tmp_image[(j * nx) + i] += image[((j + 1) * nx) + i] * 0.1f;
//     }
//   }
// }

// for(int j = 0; j < ny * nx; ++j) {
//   if(tmp_image[j] != image[j]) {
//     printf("Mismatch in (j:%d,i:%d): Expected %f, got %f\n",j / nx, j % nx, tmp_image[j],image[j]);
//     break;
//   }
//   if(j == ny*nx - 1) printf("Transfer successfull...\n");
// }

//this code is ugly man

// void send_overlaps(float *restrict source, float * restrict message)
// {
//   if (NOT_BOTTOM) {
//     //printf("Core #%d sending core below.\n", core_rank);
//     send_row(BELOW, ROW_SEND_BELOW, source, message);
//   }

//   if (NOT_TOP) {
//     //printf("Core #%d sending core above.\n", core_rank);
//     send_row(ABOVE, ROW_SEND_ABOVE, source, message);
//   }
// }

// void recv_overlaps(float *restrict dest, float * restrict message, MPI_Status status)
// {
//   if (NOT_TOP) {
//    // printf("Core #%d recv from core above.\n", core_rank);
//     recv_row(ABOVE, ROW_RECV_ABOVE, dest, message, status);
//   }
//   if (NOT_BOTTOM) {
//     //printf("Core #%d recv from bottom core.\n", core_rank);
//     recv_row(BELOW, ROW_RECV_BELOW, dest, message, status);
//   }
// }

// void send_rec(float *restrict image, float *restrict message, MPI_Status status)
// {
//   if (core_rank % 2 == 0)
//   {
//     send_overlaps(image, message);
//     recv_overlaps(image, message, status);
//   }
//   else
//   {
//     recv_overlaps(image, message, status);
//     send_overlaps(image, message);
//   }
// }