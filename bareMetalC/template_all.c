// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "template_op0.h"
#include "template_op1.h"
#include "template_op2.h"
#include "template_op3.h"

int main() {
  printf("Starting Gemmini_op0\n");
  temp_op0();
  printf("Gemmini_op0 END\n");
  printf("Starting Gemmini_op1\n");
  temp_op1();
  printf("Gemmini_op1 END\n");
  printf("Starting Gemmini_op2\n");
  temp_op2();
  printf("Gemmini_op2 END\n");
  printf("Starting Gemmini_op3\n");
  temp_op3();
  printf("Gemmini_op3 END\n");
  exit(0);
}

