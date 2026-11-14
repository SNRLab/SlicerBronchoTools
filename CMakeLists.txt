cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(BronchoTools)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "")
set(EXTENSION_CATEGORY "IGT")
set(EXTENSION_CONTRIBUTORS "Franklin King (SNR)")
set(EXTENSION_DESCRIPTION "Tools supporting image-guided bronchoscopy")
set(EXTENSION_ICONURL "")
set(EXTENSION_SCREENSHOTURLS "")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(VideoInput)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
