set(srcs            app_main.cpp
                    ../model/mnist_coefficient.cpp)

set(src_dirs        ../model)

set(include_dirs    ../model
                    ../../include/dl)

idf_component_register(SRCS ${srcs} SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires})

set(lib     libdl.a)

if(${IDF_TARGET} STREQUAL "esp32s3")
    set(links   "-L ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/esp32s3")
endif()


target_link_libraries(${COMPONENT_TARGET} ${links})
target_link_libraries(${COMPONENT_TARGET} ${lib})



