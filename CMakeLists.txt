cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(BronchoTools)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/SNRLab/SlicerBronchoTools/wiki")
set(EXTENSION_CATEGORY "Endoscopy")
set(EXTENSION_CONTRIBUTORS "Franklin King (SNR)")
set(EXTENSION_DESCRIPTION "Tools supporting image-guided bronchoscopy")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/SNRLab/SlicerBronchoTools/main/Images/BronchoToolsLogo.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/SNRLab/SlicerBronchoTools/main/Images/BronchoToolsScreenshot_1.png https://raw.githubusercontent.com/SNRLab/SlicerBronchoTools/main/Images/BronchoToolsScreenshot_2.png https://raw.githubusercontent.com/SNRLab/SlicerBronchoTools/main/Images/BronchoToolsScreenshot_3.jpg")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(VideoInput)
add_subdirectory(BronchoscopicDepthMapGeneration)
add_subdirectory(EndoluminalPhantomBuilder)

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
