
# 
# Configuration for all sub-projects
# 

# Generate version-header
string(TOUPPER ${META_PROJECT_NAME} META_PROJECT_NAME_UPPER)
configure_file(version.h.in ${CMAKE_CURRENT_BINARY_DIR}/include/${META_PROJECT_NAME}/${META_PROJECT_NAME}-version.h)


# 
# Sub-projects
# 

# Libraries
set(IDE_FOLDER "")
add_subdirectory(common)

# Examples
set(IDE_FOLDER "Examples")
add_subdirectory(attributedvertexclouds-blockworld)
add_subdirectory(attributedvertexclouds-cuboids)
add_subdirectory(attributedvertexclouds-arcs)
add_subdirectory(attributedvertexclouds-polygons)
add_subdirectory(attributedvertexclouds-trajectories)

# Tools
set(IDE_FOLDER "Tools")
add_subdirectory(noise-generator)

# 
# Deployment
# 

# Deploy generated headers
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/${META_PROJECT_NAME} DESTINATION include COMPONENT dev)
