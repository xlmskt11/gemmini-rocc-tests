// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils_all.h"

#define gemmini_num 4
#define RAND rand()

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

  printf("Flush Gemmini TLB of stale virtual addresses\n");
  gemmini_flush(custom0, 0);
  gemmini_flush(custom1, 0);
  gemmini_flush(custom2, 0);
  gemmini_flush(custom3, 0);

  printf("Initialize our input and output matrices in main memory\n");
  elem_t In[DIM][DIM];
  for (size_t i = 0; i < DIM; ++i)
  {
    for (size_t j = 0; j < DIM; ++j)
    {
      In[i][j] = RAND % 2;
    }
  }
  elem_t Out[DIM][DIM];

  elem_t Identity[DIM][DIM];
  for (size_t i = 0; i < DIM; i++)
    for (size_t j = 0; j < DIM; j++)
      Identity[i][j] = i == j;

  printf("Calculate the scratchpad addresses of all our matrices\n");
  printf("  Note: The scratchpad is \"row-addressed\", where each address contains one matrix row\n");
  size_t In_sp_addr = 0;
  size_t Out_sp_addr = DIM;
  size_t Identity_sp_addr = 2*DIM;

  printf("Move \"In\" matrix from main memory into Gemmini's scratchpad\n");
  gemmini_config_ld(custom0, DIM * sizeof(elem_t));
  // gemmini_config_ld(custom1, DIM * sizeof(elem_t));
  // gemmini_config_ld(custom2, DIM * sizeof(elem_t));
  // gemmini_config_ld(custom3, DIM * sizeof(elem_t));

  gemmini_config_st(custom0, DIM * sizeof(elem_t));
  // gemmini_config_st(custom1, DIM * sizeof(elem_t));
  // gemmini_config_st(custom2, DIM * sizeof(elem_t));
  // gemmini_config_st(custom3, DIM * sizeof(elem_t));

  gemmini_extended_mvin(custom0, (elem_t *)In + DIM * DIM * custom0 / 4, In_sp_addr + DIM * custom0 / 4, DIM, DIM / 4);
  // gemmini_extended_mvin(custom1, (elem_t *)In + DIM * DIM * custom1 / 4, In_sp_addr + DIM * custom1 / 4, DIM, DIM / 4);
  // gemmini_extended_mvin(custom2, (elem_t *)In + DIM * DIM * custom2 / 4, In_sp_addr + DIM * custom2 / 4, DIM, DIM / 4);
  // gemmini_extended_mvin(custom3, (elem_t *)In + DIM * DIM * custom3 / 4, In_sp_addr + DIM * custom3 / 4, DIM, DIM / 4);

  printf("Move \"Identity\" matrix from main memory into Gemmini's scratchpad\n");
  gemmini_extended_mvin(custom0, (elem_t *)Identity + DIM * DIM * custom0 / 4, Identity_sp_addr + DIM * custom0 / 4, DIM, DIM / 4);
  // gemmini_extended_mvin(custom1, (elem_t *)Identity + DIM * DIM * custom1 / 4, Identity_sp_addr + DIM * custom1 / 4, DIM, DIM / 4);
  // gemmini_extended_mvin(custom2, (elem_t *)Identity + DIM * DIM * custom2 / 4, Identity_sp_addr + DIM * custom2 / 4, DIM, DIM / 4);
  // gemmini_extended_mvin(custom3, (elem_t *)Identity + DIM * DIM * custom3 / 4, Identity_sp_addr + DIM * custom3 / 4, DIM, DIM / 4);

  printf("Multiply \"In\" matrix with \"Identity\" matrix with a bias of 0\n");
  gemmini_config_ex(custom0, OUTPUT_STATIONARY, 0, 0);
  // gemmini_config_ex(custom1, OUTPUT_STATIONARY, 0, 0);
  // gemmini_config_ex(custom2, OUTPUT_STATIONARY, 0, 0);
  // gemmini_config_ex(custom3, OUTPUT_STATIONARY, 0, 0);

  gemmini_extended_preload(custom0, GARBAGE_ADDR, Out_sp_addr + DIM * custom0 / 4, DIM, DIM / 4, DIM, DIM / 4);
  // gemmini_extended_preload(custom1, GARBAGE_ADDR, Out_sp_addr + DIM * custom1 / 4, DIM, DIM / 4, DIM, DIM / 4);
  // gemmini_extended_preload(custom2, GARBAGE_ADDR, Out_sp_addr + DIM * custom2 / 4, DIM, DIM / 4, DIM, DIM / 4);
  // gemmini_extended_preload(custom3, GARBAGE_ADDR, Out_sp_addr + DIM * custom3 / 4, DIM, DIM / 4, DIM, DIM / 4);

  gemmini_extended_compute_preloaded(custom0, In_sp_addr + DIM * custom0 / 4, Identity_sp_addr, DIM, DIM / 4, DIM, DIM);
  // gemmini_extended_compute_preloaded(custom1, In_sp_addr + DIM * custom1 / 4, Identity_sp_addr, DIM, DIM / 4, DIM, DIM);
  // gemmini_extended_compute_preloaded(custom2, In_sp_addr + DIM * custom2 / 4, Identity_sp_addr, DIM, DIM / 4, DIM, DIM);
  // gemmini_extended_compute_preloaded(custom3, In_sp_addr + DIM * custom3 / 4, Identity_sp_addr, DIM, DIM / 4, DIM, DIM);

  printf("Move \"Out\" matrix from Gemmini's scratchpad into main memory\n");
  gemmini_config_st(custom0, DIM * sizeof(elem_t));
  // gemmini_config_st(custom1, DIM * sizeof(elem_t));
  // gemmini_config_st(custom2, DIM * sizeof(elem_t));
  // gemmini_config_st(custom3, DIM * sizeof(elem_t));

  gemmini_extended_mvout(custom0, Out + DIM * DIM * custom0 / 4, Out_sp_addr + DIM * custom0 / 4, DIM, DIM / 4);
  // gemmini_extended_mvout(custom1, Out + DIM * DIM * custom1 / 4, Out_sp_addr + DIM * custom1 / 4, DIM, DIM / 4);
  // gemmini_extended_mvout(custom2, Out + DIM * DIM * custom2 / 4, Out_sp_addr + DIM * custom2 / 4, DIM, DIM / 4);
  // gemmini_extended_mvout(custom3, Out + DIM * DIM * custom3 / 4, Out_sp_addr + DIM * custom3 / 4, DIM, DIM / 4);

  printf("Fence till Gemmini completes all memory operations\n");
  gemmini_fence();

  printf("Check whether \"In\" and \"Out\" matrices are identical\n");
  if (!is_equal(In, Out)) {
    printf("Input and output matrices are different!\n");
    printf("\"In\" matrix:\n");
    printMatrix(In);
    printf("\"Out\" matrix:\n");
    printMatrix(Out);
    printf("\n");

    exit(1);
  }

  printf("Input and output matrices are identical, as expected\n");
  exit(0);
}

