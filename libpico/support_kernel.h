#ifndef SUPPORT_KERNEL_H
#define SUPPORT_KERNEL_H

#ifdef __cplusplus
extern "C" {
#endif

void reduce_wrapper(void* inbuff, void* inoutbuff, int count);

#ifdef __cplusplus
}
#endif
#endif