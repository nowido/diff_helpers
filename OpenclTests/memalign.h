#ifndef MEMALIGN_H
#define MEMALIGN_H
 
//-------------------------------------------------------------
// aligned memory allocation cross-stuff
//-------------------------------------------------------------

#ifdef _WIN32

#include <malloc.h>

#define aligned_alloc(alignment, size) _aligned_malloc((size), (alignment))
#define aligned_free _aligned_free
#define align_as(alignment) __declspec(align(alignment))

#else

void* aligned_alloc(size_t alignment, size_t size);

void* aligned_alloc(size_t alignment, size_t size)
{
    void* p = NULL;
    posix_memalign (&p, alignment, size);
    return p;
}

#define aligned_free free
#define align_as(alignment) __attribute__((aligned((alignment))))

#endif

#define CACHE_LINE 64
#define SSE_ALIGNMENT 16
#define SSE_BASE_COUNT 4

#endif

