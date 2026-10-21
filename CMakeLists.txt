cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SlicerHaversack)

#-----------------------------------------------------------------------------
# Extension metadata
#-----------------------------------------------------------------------------
set(EXTENSION_HOMEPAGE "https://github.com/pieper/SlicerHaversack")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Steve Pieper (Isomics)")
set(EXTENSION_DESCRIPTION "Run haversack's nnU-Net-family segmentation models (TotalSegmentator, MOOSE, MRSegmentator, CADS, ...), the LNQ lymph-node models, and license-gated extra models such as PanTS on a volume in the scene, in a managed Python environment, on Apple Silicon, CUDA or CPU.")
set(EXTENSION_ICONURL "")
set(EXTENSION_SCREENSHOTURLS "")
set(EXTENSION_DEPENDS "NA")

#-----------------------------------------------------------------------------
# Extension dependencies
#-----------------------------------------------------------------------------
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
#-----------------------------------------------------------------------------
add_subdirectory(HaversackSegment)

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
