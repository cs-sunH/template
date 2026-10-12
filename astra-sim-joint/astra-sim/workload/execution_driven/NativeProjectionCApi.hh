/******************************************************************************
Stable C ABI for Python ctypes access to the native projection engine.
******************************************************************************/

#ifndef EXECUTION_DRIVEN_NATIVE_PROJECTION_C_API_HH
#define EXECUTION_DRIVEN_NATIVE_PROJECTION_C_API_HH

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(JOINT_PROJECTION_BUILD)
#    define JPROJ_EXPORT __declspec(dllexport)
#  else
#    define JPROJ_EXPORT __declspec(dllimport)
#  endif
#else
#  define JPROJ_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

JPROJ_EXPORT uint32_t jproj_abi_version(void);
JPROJ_EXPORT int jproj_create(const char* snapshot, size_t len,
                              void** out_context);
JPROJ_EXPORT int jproj_project(void* context,
                               const char* candidate, size_t candidate_len,
                               const char* options, size_t options_len,
                               char** out_result, size_t* out_len);
/// Stateful compatibility surface for ProjectionWorld constructor/step/run.
/// This session is owned by the opaque context handle and never mutates the
/// immutable context consumed by jproj_project.
JPROJ_EXPORT int jproj_world_operation(void* context,
                                      const char* request, size_t request_len,
                                      char** out_result, size_t* out_len);
JPROJ_EXPORT void jproj_free_result(char* result);
JPROJ_EXPORT void jproj_destroy(void* context);
JPROJ_EXPORT const char* jproj_last_error(void* context);

#ifdef __cplusplus
}
#endif

#endif  // EXECUTION_DRIVEN_NATIVE_PROJECTION_C_API_HH
